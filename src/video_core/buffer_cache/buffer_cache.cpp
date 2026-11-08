// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <Zydis/Zydis.h>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/ab_switch.h"
#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    memory_semaphore.SetName("sparse binds");
    if (const char* env = std::getenv("SHADPS4_REUSE_STREAM_COPIES")) {
        reuse_stream_copies = env[0] != '0';
    }
    if (const char* env = std::getenv("SHADPS4_AB");
        env && std::string_view{env} == "stream-copies") {
        stream_copies_ab = Common::AbSwitch::Register("Copy every read", "Reuse stream copies",
                                                      reuse_stream_copies);
    }
    const u32 readbacks_mode = EmulatorSettings.GetReadbacksMode();
    tracks_gpu_pages =
        readbacks_mode == GpuReadbacksMode::Precise || readbacks_mode == GpuReadbacksMode::Fenced;
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = SupportedUsage(instance, ARENA_USAGE),
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    ProcessLandings();
    LogReadbackStats();
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks, VAddr ahead,
                                   u64 ahead_size) {
    memory_tracker->InvalidateRegion(device_addr, size, ahead, ahead_size,
                                     [this, device_addr, size, assume_locks] {
                                         ReadMemory(device_addr, size, true, false, assume_locks);
                                     });
}

void BufferCache::UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write) {
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, is_write);
    gpu_modified_ranges.Subtract(device_addr, size);
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool async,
                             bool assume_locks) {
    if (assume_locks) {
        ReadMemoryNow(device_addr, size, is_write, async);
    } else {
        liverpool->SendCommand<true>([this, device_addr, size, is_write, async] {
            ReadMemoryNow(device_addr, size, is_write, async);
        });
    }
}

bool BufferCache::ReadMemoryForLoad(VAddr device_addr, u64 size, VAddr load_addr, u32 load_size,
                                    u16 mnemonic, u8* data, bool assume_locks) {
    bool served = false;
    const auto read = [&] {
        if (load_size == 0) {
            ++unsupported_loads[mnemonic];
        } else if (!GpuBytesPending(load_addr, load_size) && TakeFaultLoadBudget(load_addr) &&
                   memory->ReadBackingPages(load_addr, data, load_size)) {
            // Only bytes the CPU has the current data of: guest memory has them already.
            ++readback_stats.served_loads;
            served = true;
            return;
        } else if (GpuBytesPending(load_addr, load_size)) {
            ++readback_stats.gpu_byte_loads;
        }
        ReadMemoryNow(device_addr, size, false, false);
    };
    if (assume_locks) {
        read();
    } else {
        liverpool->SendCommand<true>(read);
    }
    return served;
}

bool BufferCache::CopyMemoryForLoad(VAddr device_addr, u64 size, VAddr src, VAddr dst, u64 bytes,
                                    const std::function<bool(const u8*)>& write_dst,
                                    bool assume_locks) {
    bool served = false;
    const auto copy = [&] {
        // A pending landing into the destination would overwrite the copy later.
        if (!GpuBytesPending(src, bytes) && !GpuBytesPending(dst, bytes) &&
            TakeFaultLoadBudget(device_addr)) {
            copy_scratch.resize(bytes);
            if (memory->ReadBackingPages(src, copy_scratch.data(), bytes) &&
                write_dst(copy_scratch.data())) {
                ++readback_stats.served_loads;
                ++readback_stats.served_copies;
                readback_stats.served_copy_bytes += bytes;
                served = true;
                return;
            }
        } else if (GpuBytesPending(src, bytes) || GpuBytesPending(dst, bytes)) {
            ++readback_stats.gpu_byte_loads;
        }
        ReadMemoryNow(device_addr, size, false, false);
    };
    if (assume_locks) {
        copy();
    } else {
        liverpool->SendCommand<true>(copy);
    }
    return served;
}

bool BufferCache::TakeFaultLoadBudget(VAddr address) {
    static const u32 budget = [] {
        const char* env = std::getenv("SHADPS4_FAULT_LOAD_BUDGET");
        return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 64u;
    }();
    constexpr u64 WindowNs = 16'000'000;
    const u64 now = VideoCore::PerfStats::NowNs();
    auto& page = fault_load_budgets[address >> 12];
    if (now - page.window_start_ns > WindowNs) {
        page = {now, 0};
    }
    if (page.served >= budget) {
        ++readback_stats.over_budget_loads;
        return false;
    }
    ++page.served;
    return true;
}

void BufferCache::NoteGpuThreadReadFault(u64 instruction, bool served) {
    auto& [count, served_count] = gpu_thread_fault_sites[instruction];
    ++count;
    served_count += served ? 1 : 0;
}

namespace {
/// "symbol+offset (module+offset)" for a code address, for logs.
std::string DescribeCodeAddress(u64 address) {
#ifndef _WIN32
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(address), &info) != 0 && info.dli_fbase != nullptr) {
        const u64 module_offset = address - reinterpret_cast<u64>(info.dli_fbase);
        const char* module = info.dli_fname ? std::strrchr(info.dli_fname, '/') : nullptr;
        module = module ? module + 1 : (info.dli_fname ? info.dli_fname : "?");
        if (info.dli_sname != nullptr) {
            return fmt::format("{}+{:#x} ({}+{:#x})", info.dli_sname,
                               address - reinterpret_cast<u64>(info.dli_saddr), module,
                               module_offset);
        }
        return fmt::format("{}+{:#x}", module, module_offset);
    }
#endif
    return fmt::format("{:#x}", address);
}
} // namespace

bool BufferCache::GpuBytesPending(VAddr address, u64 size) const {
    // Pending bytes keep their pages marked GPU-modified until they land or are read back, so
    // an unmarked page (most shader code) answers with a bit test instead of range lookups.
    if (!memory_tracker->IsRegionGpuModified(address, size)) {
        return false;
    }
    return gpu_modified_ranges.Intersects(address, size) ||
           async_downloads.Intersects(address, size) ||
           std::ranges::any_of(landings, [&](const auto& landing) {
               return landing->ranges.Intersects(address, size);
           });
}

void BufferCache::ReadMemoryNow(VAddr device_addr, u64 size, bool is_write, bool async) {
    read_reason = is_write ? ReadReason::Write : ReadReason::Other;
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const u64 page_size = Common::AlignUp(device_addr + size, BYTES_PER_PAGE) - page_start;
    if (tracks_gpu_pages && !is_write && !async) {
        // A CPU read fault: remembered so fenced readbacks land this memory early from now
        // on. Unmaps and GC evictions also come here, but don't count.
        read_reason = land_early_ranges.Intersects(page_start, page_size) ? ReadReason::RepeatRead
                                                                          : ReadReason::FirstRead;
        land_early_ranges.Add(page_start, page_size);
        if (read_reason == ReadReason::RepeatRead) {
            NoteRepeatRead(device_addr, size);
        }
    } else if (tracks_gpu_pages && is_write && !async) {
        // The CPU writing a page that holds GPU data has to wait for that data first, so the
        // page lands early from now on too: a CPU filling a ring the GPU also writes would
        // otherwise stall on every page.
        NoteWriteFault(device_addr, size);
        land_early_ranges.Add(page_start, page_size);
    }
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);

    // GPU-modified ranges come as many small scattered islands,
    // so the download is widened to a window around the request
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    DownloadMemory(arena, window_start, window_end - window_start, async);
    read_reason = ReadReason::Other;
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
}

void BufferCache::NoteWriteFault(VAddr address, u64 size) {
    const VAddr page = Common::AlignDown(address, BYTES_PER_PAGE);
    const auto pending = [&](VAddr start, u64 length) {
        return gpu_modified_ranges.Intersects(start, length) ||
               std::ranges::any_of(landings, [&](const auto& landing) {
                   return landing->ranges.Intersects(start, length);
               });
    };
    u64 gpu_bytes = 0;
    gpu_modified_ranges.ForEachInRange(page, BYTES_PER_PAGE,
                                       [&](VAddr start, VAddr end) { gpu_bytes += end - start; });
    for (const auto& landing : landings) {
        landing->ranges.ForEachInRange(page, BYTES_PER_PAGE,
                                       [&](VAddr start, VAddr end) { gpu_bytes += end - start; });
    }
    auto& entry = write_fault_pages[page];
    ++entry.faults;
    entry.gpu_bytes = std::max<u32>(entry.gpu_bytes, std::min<u64>(gpu_bytes, BYTES_PER_PAGE));
    if (pending(address, size)) {
        ++entry.onto_gpu_bytes;
    }
}

void BufferCache::NoteRepeatRead(VAddr address, u64 size) {
    const VAddr page = Common::AlignDown(address, BYTES_PER_PAGE);
    const auto in_flight = [&](VAddr start, u64 length) {
        return std::ranges::any_of(landings, [&](const auto& landing) {
            return landing->ranges.Intersects(start, length);
        });
    };
    if (gpu_modified_ranges.Intersects(address, size) || in_flight(address, size)) {
        ++readback_stats.repeat_reads_of_gpu_bytes;
    }
    if (in_flight(page, BYTES_PER_PAGE)) {
        ++readback_stats.repeat_reads_in_flight;
    }
    u64 gpu_bytes = 0;
    gpu_modified_ranges.ForEachInRange(page, BYTES_PER_PAGE,
                                       [&](VAddr start, VAddr end) { gpu_bytes += end - start; });
    for (const auto& landing : landings) {
        landing->ranges.ForEachInRange(page, BYTES_PER_PAGE,
                                       [&](VAddr start, VAddr end) { gpu_bytes += end - start; });
    }
    auto& entry = fault_pages[page];
    ++entry.faults;
    entry.gpu_bytes = std::max<u32>(entry.gpu_bytes, std::min<u64>(gpu_bytes, BYTES_PER_PAGE));
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size, bool async) {
    if (!landings.empty()) {
        // Data already on its way back must land first, or it would overwrite what the CPU
        // writes next. Then pages whose landing is done can simply be unprotected.
        WaitForLandings(device_addr, size);
        ProcessLandings();
    }
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        if (tracks_gpu_pages && !async_downloads.Intersects(device_addr, size)) {
            // Everything here has landed; nothing is left to protect. Unless an async download
            // took it and hasn't written guest memory yet: then the fault retries until it has.
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
        }
        return;
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 16, async);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    if (!async) {
        ++readback_stats.sync_downloads;
        readback_stats.sync_download_bytes += total_size_bytes;
        ++readback_stats.sync_by_reason[static_cast<u32>(read_reason)];
    }

    const auto write_data = [this, arena_base, device_addr, size, download, async,
                             copies = std::move(copies)] {
        download.Invalidate();
        for (const auto& copy : copies) {
            auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
            memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                    copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);

        if (async) {
            staging_pool.FreeDeferred(download);
            if (--num_async_downloads == 0) {
                async_downloads.Clear();
            }
        }
    };

    if (async) {
        // Pending operations run on this thread, at a submit.
        async_downloads.Add(device_addr, size);
        ++num_async_downloads;
        scheduler.DeferOperation(std::move(write_data));
    } else {
        const auto stall_start = std::chrono::steady_clock::now();
        scheduler.Finish();
        const u64 stall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - stall_start)
                                 .count();
        readback_stats.stall_ns += stall_ns;
        readback_stats.stall_ns_by_reason[static_cast<u32>(read_reason)] += stall_ns;
        write_data();
    }
}

bool BufferCache::LandGpuModifiedMemory() {
    ProcessLandings();

    // Ranges above these sizes stay GPU-side and are read back by the fault path if the CPU
    // ever touches them, so a big GPU-only buffer isn't copied at every fence.
    constexpr u64 MaxRangeSize = 16_MB;
    constexpr u64 MaxLandingSize = LandingRingSize / 4;
    constexpr u64 CopyAlign = 64;

    struct ArenaCopies {
        const Buffer* arena;
        boost::container::small_vector<vk::BufferCopy, 16> copies;
    };
    struct Write {
        VAddr address;
        u64 offset;
        u64 size;
    };
    boost::container::small_vector<ArenaCopies, 2> arena_copies;
    std::vector<Write> writes;
    auto landing = std::make_shared<Landing>();
    u64 total_size_bytes = 0;
    const auto add_range = [&](VAddr start, VAddr end) {
        while (start < end) {
            const Buffer* arena = address_space[start >> ARENA_PAGE_BITS];
            if (!arena) {
                return;
            }
            const VAddr chunk_end = std::min<VAddr>(end, arena->cpu_addr + arena->size_bytes);
            const u64 size = chunk_end - start;
            if (size <= MaxRangeSize && total_size_bytes + size <= MaxLandingSize) {
                if (arena_copies.empty() || arena_copies.back().arena != arena) {
                    arena_copies.push_back({arena, {}});
                }
                arena_copies.back().copies.push_back(vk::BufferCopy{
                    .srcOffset = start - arena->cpu_addr,
                    .dstOffset = total_size_bytes,
                    .size = size,
                });
                writes.push_back({start, total_size_bytes, size});
                landing->ranges.Add(start, size);
                total_size_bytes += Common::AlignUp(size, CopyAlign);
            }
            start = chunk_end;
        }
    };
    // Only memory the CPU has faulted on before lands early: most of what the GPU writes is never
    // read back, and copying it all costs more than the occasional fault on the rest.
    land_early_ranges.ForEach([&](VAddr start, VAddr end) {
        gpu_modified_ranges.ForEachInRange(start, end - start, add_range);
    });
    if (total_size_bytes == 0) {
        return false;
    }

    // Staging space comes from a ring of host memory owned by the landings. Landings finish in
    // order, so the ring frees from its tail. When it's full, this fence skips landing: its
    // ranges stay protected and are read back by the fault path if the CPU touches them.
    if (!landing_ring) {
        landing_ring = std::make_unique<Buffer>(instance, 0, LandingRingSize,
                                                MemoryType::HostCached, "Readback Landing Ring");
    }
    u64 ring_start = landing_ring_head;
    if (ring_start % LandingRingSize + total_size_bytes > LandingRingSize) {
        ring_start = Common::AlignUp(ring_start, LandingRingSize); // wrap past the ring's end
    }
    const u64 ring_end = ring_start + total_size_bytes;
    if (ring_end - landing_ring_tail.load(std::memory_order_acquire) > LandingRingSize) {
        ++readback_stats.full_ring_landings;
        return false;
    }
    landing_ring_head = ring_end;
    const u64 ring_offset = ring_start % LandingRingSize;

    landing->ranges.ForEach(
        [&](VAddr start, VAddr end) { gpu_modified_ranges.Subtract(start, end - start); });
    scheduler.EndRendering();
    for (auto& [arena, copies] : arena_copies) {
        for (auto& copy : copies) {
            copy.dstOffset += ring_offset;
        }
        runtime.CopyBuffer(arena, landing_ring.get(), copies);
    }
    ++readback_stats.landings;
    readback_stats.landing_bytes += total_size_bytes;
    landing->tick = scheduler.CurrentTick();
    landings.push_back(landing);
    scheduler.DeferPriorityOperation([this, ring_offset, total_size_bytes, ring_end,
                                      landing = std::move(landing), writes = std::move(writes)] {
        landing_ring->Invalidate(ring_offset, total_size_bytes);
        const u8* mapped = landing_ring->mapped_data.data() + ring_offset;
        for (const auto& write : writes) {
            memory->TryWriteBacking(std::bit_cast<u8*>(write.address), mapped + write.offset,
                                    write.size);
        }
        landing_ring_tail.store(ring_end, std::memory_order_release);
        landing->done.store(true, std::memory_order_release);
        landing->done.notify_all();
    });
    return true;
}

void BufferCache::ProcessLandings() {
    while (!landings.empty() && landings.front()->done.load(std::memory_order_acquire)) {
        const auto landing = std::move(landings.front());
        landings.pop_front();
        // Tracking is per page. A page is unprotected only if the GPU hasn't written it again
        // since the copy was recorded and no later landing still has to fill it.
        const auto is_current = [this](VAddr page) {
            if (gpu_modified_ranges.Intersects(page, BYTES_PER_PAGE)) {
                return false;
            }
            return std::ranges::none_of(landings, [page](const auto& pending) {
                return pending->ranges.Intersects(page, BYTES_PER_PAGE);
            });
        };
        landing->ranges.ForEach([&](VAddr start, VAddr end) {
            const VAddr page_end = Common::AlignUp(end, BYTES_PER_PAGE);
            std::optional<VAddr> run_start;
            for (VAddr page = Common::AlignDown(start, BYTES_PER_PAGE); page < page_end;
                 page += BYTES_PER_PAGE) {
                const bool current = is_current(page);
                if (current && !run_start) {
                    run_start = page;
                } else if (!current && run_start) {
                    memory_tracker->UnmarkRegionAsGpuModified(*run_start, page - *run_start, false);
                    run_start.reset();
                }
            }
            if (run_start) {
                memory_tracker->UnmarkRegionAsGpuModified(*run_start, page_end - *run_start, false);
            }
        });
    }
}

void BufferCache::LogReadbackStats() {
    constexpr u64 IntervalNs = 5'000'000'000ULL;
    const u64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    if (readback_stats.last_log_ns == 0) {
        readback_stats =
            ReadbackStats{.last_log_ns = now, .ab_switches = Common::AbSwitch::Switches()};
        return;
    }
    // A window that spans an A/B switch would mix both variants: log what came before it.
    if (now - readback_stats.last_log_ns >= IntervalNs ||
        readback_stats.ab_switches != Common::AbSwitch::Switches()) {
        FlushReadbackStats();
    }
}

void BufferCache::FlushReadbackStats() {
    const u64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    auto& stats = readback_stats;
    const double seconds = double(now - stats.last_log_ns) / 1e9;
    // The fault lines below are logged without readbacks too (Disabled mode), to compare.
    const bool any_readbacks =
        stats.sync_downloads != 0 || stats.landings != 0 || stats.deferred_fences != 0;
    if (seconds > 0.5) {
        // While an A/B switch flips, the variant is the one before the switch.
        const bool switched = stats.ab_switches != Common::AbSwitch::Switches();
        const char* variant = Common::AbSwitch::Label();
        if (variant && switched) {
            variant = Common::AbSwitch::LabelOf(!Common::AbSwitch::IsB());
        }
        const auto& n = stats.sync_by_reason;
        const auto& ns = stats.stall_ns_by_reason;
        if (any_readbacks) {
            // Milliseconds of each second the GPU thread spent waiting.
            LOG_INFO(
                Render,
                "Readbacks per second [{}] over {:.1f} s: {:.1f} stalling downloads ({:.1f} KB, "
                "{:.2f} ms stalled), {:.1f} fence landings ({:.1f} KB), fences {:.1f} deferred / "
                "{:.1f} immediate, {:.2f} ms throttled",
                variant ? variant : "-", seconds, stats.sync_downloads / seconds,
                stats.sync_download_bytes / 1024.0 / seconds, stats.stall_ns / 1e6 / seconds,
                stats.landings / seconds, stats.landing_bytes / 1024.0 / seconds,
                stats.deferred_fences / seconds, stats.immediate_fences / seconds,
                stats.throttle_ns / 1e6 / seconds);
            LOG_INFO(
                Render,
                "Stalling downloads per second [{}]: first reads {:.1f} ({:.2f} ms), repeat reads "
                "{:.1f} ({:.2f} ms), writes {:.1f} ({:.2f} ms), other {:.1f} ({:.2f} ms); "
                "landings skipped {:.1f} as too soon, {:.1f} for a full ring",
                variant ? variant : "-", n[0] / seconds, ns[0] / 1e6 / seconds, n[1] / seconds,
                ns[1] / 1e6 / seconds, n[2] / seconds, ns[2] / 1e6 / seconds, n[3] / seconds,
                ns[3] / 1e6 / seconds, stats.skipped_landings / seconds,
                stats.full_ring_landings / seconds);
        }
        if (stats.served_loads != 0 || stats.gpu_byte_loads != 0 || stats.over_budget_loads != 0 ||
            !unsupported_loads.empty()) {
            std::vector<std::pair<u16, u32>> loads(unsupported_loads.begin(),
                                                   unsupported_loads.end());
            const size_t shown = std::min<size_t>(loads.size(), 5);
            std::partial_sort(loads.begin(), loads.begin() + shown, loads.end(),
                              [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string top;
            for (size_t i = 0; i < shown; ++i) {
                const char* name =
                    ZydisMnemonicGetString(static_cast<ZydisMnemonic>(loads[i].first));
                top += fmt::format("{}{} {:.1f}", i ? ", " : "", name ? name : "?",
                                   loads[i].second / seconds);
            }
            LOG_INFO(Render,
                     "Read faults per second [{}]: {:.1f} served from guest memory (no GPU bytes; "
                     "{:.1f} movs copies, {:.1f} KB), {:.1f} read GPU bytes, {:.1f} read back "
                     "over their page's budget; not decoded: {}",
                     variant ? variant : "-", stats.served_loads / seconds,
                     stats.served_copies / seconds, stats.served_copy_bytes / 1024.0 / seconds,
                     stats.gpu_byte_loads / seconds, stats.over_budget_loads / seconds,
                     top.empty() ? "none" : top);
        }
        if (n[1] != 0) {
            std::vector<std::pair<VAddr, FaultPage>> pages(fault_pages.begin(), fault_pages.end());
            const size_t shown = std::min<size_t>(pages.size(), 6);
            std::partial_sort(
                pages.begin(), pages.begin() + shown, pages.end(),
                [](const auto& a, const auto& b) { return a.second.faults > b.second.faults; });
            std::string top;
            for (size_t i = 0; i < shown; ++i) {
                top += fmt::format("{}{:#x} {} ({} B GPU-written)", i ? ", " : "", pages[i].first,
                                   pages[i].second.faults, pages[i].second.gpu_bytes);
            }
            LOG_INFO(Render,
                     "Repeat read faults per second [{}]: {:.1f} read bytes the GPU wrote, {:.1f} "
                     "hit a page a landing was filling; {} pages, most faulted: {}",
                     variant ? variant : "-", stats.repeat_reads_of_gpu_bytes / seconds,
                     stats.repeat_reads_in_flight / seconds, pages.size(), top);
        }
        if (!gpu_thread_fault_sites.empty()) {
            std::vector<std::pair<u64, std::pair<u32, u32>>> sites(gpu_thread_fault_sites.begin(),
                                                                   gpu_thread_fault_sites.end());
            const size_t shown = std::min<size_t>(sites.size(), 6);
            std::partial_sort(
                sites.begin(), sites.begin() + shown, sites.end(),
                [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
            std::string top;
            for (size_t i = 0; i < shown; ++i) {
                const auto [instruction, counts] = sites[i];
                top += fmt::format("{}{} {:.1f} ({:.1f} served)", i ? ", " : "",
                                   DescribeCodeAddress(instruction), counts.first / seconds,
                                   counts.second / seconds);
            }
            LOG_INFO(Render, "GPU-thread read faults per second [{}] by instruction: {}",
                     variant ? variant : "-", top);
        }
        if (!write_fault_pages.empty()) {
            u64 faults = 0;
            u64 onto_gpu_bytes = 0;
            for (const auto& [page, entry] : write_fault_pages) {
                faults += entry.faults;
                onto_gpu_bytes += entry.onto_gpu_bytes;
            }
            std::vector<std::pair<VAddr, WriteFaultPage>> pages(write_fault_pages.begin(),
                                                                write_fault_pages.end());
            const size_t shown = std::min<size_t>(pages.size(), 6);
            std::partial_sort(
                pages.begin(), pages.begin() + shown, pages.end(),
                [](const auto& a, const auto& b) { return a.second.faults > b.second.faults; });
            std::string top;
            for (size_t i = 0; i < shown; ++i) {
                top += fmt::format("{}{:#x} {} ({} B GPU-written, {} onto them)", i ? ", " : "",
                                   pages[i].first, pages[i].second.faults,
                                   pages[i].second.gpu_bytes, pages[i].second.onto_gpu_bytes);
            }
            LOG_INFO(Render,
                     "Write faults on GPU data per second [{}]: {:.1f} ({:.1f} wrote bytes the GPU "
                     "wrote); {} pages, most faulted: {}",
                     variant ? variant : "-", faults / seconds, onto_gpu_bytes / seconds,
                     pages.size(), top);
        }
        if (const auto reads = TakeReadFaultTotals(); reads.count != 0) {
            LOG_INFO(Render,
                     "All read faults per second [{}]: {:.1f}, {:.2f} ms ({:.1f} us each); on the "
                     "GPU thread {:.1f}, {:.2f} ms",
                     variant ? variant : "-", reads.count / seconds, reads.ns / 1e6 / seconds,
                     reads.ns / 1e3 / reads.count, reads.gpu_thread_count / seconds,
                     reads.gpu_thread_ns / 1e6 / seconds);
        }
        if (const auto w = TakeWriteFaultTotals(); w.count != 0) {
            // Each fault's time (which includes unprotecting ahead), split: the lock waits are
            // only where a lock was contended; "rest" is the handler's own work (tracking bits,
            // watchers) with uncontended locks.
            const auto& p = w.phases;
            const u64 named = p.region_lock_ns + p.page_lock_ns + p.protect_ns + p.texture_ns;
            const auto us = [&](u64 ns) { return ns / 1e3 / w.count; };
            LOG_INFO(Render,
                     "Write faults per second [{}]: {:.1f}, {:.2f} ms ({:.1f} us each: region lock "
                     "{:.2f}, page locks {:.2f}, mprotect {:.2f} in {:.2f} calls of {:.1f} pages, "
                     "textures {:.2f}, rest {:.2f}); {:.1f} also unprotected {:.1f} pages ahead",
                     variant ? variant : "-", w.count / seconds, w.ns / 1e6 / seconds,
                     w.ns / 1e3 / w.count, us(p.region_lock_ns), us(p.page_lock_ns),
                     us(p.protect_ns), double(p.protect_calls) / w.count,
                     p.protect_calls ? double(p.protect_pages) / p.protect_calls : 0.0,
                     us(p.texture_ns), w.ns > named ? us(w.ns - named) : 0.0,
                     w.ahead_count / seconds,
                     w.ahead_count ? double(w.ahead_pages) / w.ahead_count : 0.0);
        }
        if (auto sites = TakeSlowWriteFaultSites(); !sites.empty()) {
            const size_t shown = std::min<size_t>(sites.size(), 6);
            std::partial_sort(sites.begin(), sites.begin() + shown, sites.end(),
                              [](const auto& a, const auto& b) { return a.count > b.count; });
            std::string top;
            for (size_t i = 0; i < shown; ++i) {
                const auto& site = sites[i];
                const auto us = [&](u64 ns) { return ns / 1e3 / site.count; };
                top += fmt::format("{}{}{} {:.1f} ({:.0f} us: region lock {:.0f}, page locks "
                                   "{:.0f}, mprotect {:.0f}, textures {:.0f})",
                                   i ? ", " : "", DescribeCodeAddress(site.instruction),
                                   site.gpu_thread ? " [GPU thread]" : "", site.count / seconds,
                                   us(site.total_ns), us(site.phases.region_lock_ns),
                                   us(site.phases.page_lock_ns), us(site.phases.protect_ns),
                                   us(site.phases.texture_ns));
            }
            LOG_INFO(Render, "Slow write faults per second [{}] by instruction: {}",
                     variant ? variant : "-", top);
        }
        if (VideoCore::RegionLockProfiling()) {
            static constexpr std::array<const char*, size_t(VideoCore::RegionLockOp::Count)> names{
                "CPU set",           "CPU clear",         "GPU set",       "GPU clear",
                "CPU set+GPU clear", "CPU clear+GPU set", "upload ranges", "upload ranges written",
                "download ranges",   "explicit (faults)", "other"};
            std::string holds;
            for (size_t i = 0; i < names.size(); ++i) {
                auto& entry = VideoCore::region_lock_holds[i];
                const u64 count = entry.count.exchange(0);
                const u64 hold_ns = entry.hold_ns.exchange(0);
                const u64 protect_ns = entry.protect_ns.exchange(0);
                const u64 max_ns = entry.max_ns.exchange(0);
                if (count != 0) {
                    holds +=
                        fmt::format("{}{} {:.1f} (avg {:.0f} us, mprotect {:.0f} us, max "
                                    "{:.0f} us)",
                                    holds.empty() ? "" : ", ", names[i], count / seconds,
                                    hold_ns / 1e3 / count, protect_ns / 1e3 / count, max_ns / 1e3);
                }
            }
            LOG_INFO(Render, "Region tracker locks held over 20 us per second [{}]: {}",
                     variant ? variant : "-", holds.empty() ? "none" : holds);
        }
    }
    gpu_thread_fault_sites.clear();
    fault_pages.clear();
    write_fault_pages.clear();
    fault_load_budgets.clear();
    unsupported_loads.clear();
    stats = ReadbackStats{.last_log_ns = now, .ab_switches = Common::AbSwitch::Switches()};
}

void BufferCache::WaitForLandings(VAddr device_addr, u64 size) {
    for (const auto& landing : landings) {
        if (!landing->done.load(std::memory_order_acquire) &&
            landing->ranges.Intersects(device_addr, size)) {
            // Fenced signals submit lazily, so the copy may still be unsubmitted. The priority
            // op only runs once the GPU is done with it, and this thread is the one that
            // submits: Wait() flushes first if needed, or this would deadlock.
            scheduler.Wait(landing->tick);
            landing->done.wait(false, std::memory_order_acquire);
        }
    }
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        ObtainBufferFlags flags) {
    // Thousands of calls a frame: timed whole only with SHADPS4_DRAW_COSTS=1. Otherwise only
    // the slow parts below are (making memory resident, uploading it).
    const VideoCore::PerfStats::ScopedCost cost{VideoCore::PerfStats::Cost::Buffers, true};
    const bool is_written = True(flags & ObtainBufferFlags::IsWritten);
    const bool is_texel_buffer = True(flags & ObtainBufferFlags::IsTexelBuffer);
    const bool skip_stream_buffer = True(flags & ObtainBufferFlags::IgnoreStreamBuffer);
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && !skip_stream_buffer && size <= STREAM_THRESHOLD &&
        !IsRegionGpuModified(device_addr, size)) {
        // Only within one command buffer: the stream buffer frees regions by the tick they
        // were copied in, so a later tick reading an old region could see it overwritten.
        const u64 tick = scheduler.CurrentTick();
        auto& copy =
            stream_copies[((device_addr >> 4) ^ (device_addr >> 14) ^ size) % stream_copies.size()];
        const bool reuse = stream_copies_ab ? Common::AbSwitch::IsB() : reuse_stream_copies;
        if (reuse && copy.address == device_addr && copy.size == size &&
            copy.epoch == stream_copy_epoch && copy.tick == tick &&
            copy.wraps == stream_buffer.Wraps()) {
            VideoCore::PerfStats::frames.AddGpuCount(VideoCore::PerfStats::Cost::BufferBytesReused,
                                                     size);
            return {&stream_buffer, copy.offset};
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        VideoCore::PerfStats::frames.AddGpuCount(VideoCore::PerfStats::Cost::BufferBytes, size);
        copy = StreamCopy{
            .address = device_addr,
            .size = size,
            .epoch = stream_copy_epoch,
            .tick = tick,
            .wraps = stream_buffer.Wraps(),
            .offset = offset,
        };
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    if (is_written) {
        texture_cache.MarkAsMaybeGpuDirty(device_addr, size);
    }
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size,
                                                                bool device_local) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    if (!device_local) {
        return {staging.buffer, staging.offset};
    }
    const auto local = staging_pool.Request(size, VideoCore::MemoryType::DeviceLocal,
                                            instance.StorageMinAlignment());
    const vk::BufferCopy copy{staging.offset, local.offset, size};
    runtime.CopyBuffer(staging.buffer, local.buffer, std::span{&copy, 1});
    return {local.buffer, local.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::MarkRegionAsGpuModified(VAddr addr, size_t size) {
    texture_cache.MarkAsMaybeGpuDirty(addr, size);
    memory_tracker->MarkRegionAsGpuModified(addr, size);
    gpu_modified_ranges.Add(addr, size);
}

void BufferCache::ReadEdgeImagePages(const Image& image) {
    // May happen that after downloading the image and invalidating region,
    // that there were GPU modified ranges that are lost due to CPU reuploading.
    // This doesn't change tracker state and it is spected to call DownloadImageMemory after this.
    const VAddr image_addr = image.info.guest_address;
    const u64 image_size = image.info.guest_size;
    const VAddr image_end = image_addr + image_size;
    const VAddr page_start = PageManager::GetPageAddr(image_addr);
    const VAddr page_end = PageManager::GetNextPageAddr(image_end - 1);
    boost::container::small_vector<vk::BufferCopy, 2> copies;
    u64 total_size_bytes = 0;
    const auto [buffer, offset] = ObtainBuffer(page_start, page_end - page_start);
    const auto add_download = [&](VAddr start, VAddr end) {
        const u64 new_offset = start - buffer->CpuAddr();
        const u64 new_size = end - start;
        copies.push_back(vk::BufferCopy{
            .srcOffset = new_offset,
            .dstOffset = total_size_bytes,
            .size = new_size,
        });
        // Align up to avoid cache conflicts
        constexpr u64 align = 64ULL;
        constexpr u64 mask = ~(align - 1ULL);
        total_size_bytes += (new_size + align - 1) & mask;
    };
    gpu_modified_ranges.ForEachInRange(page_start, image_addr - page_start, add_download);
    gpu_modified_ranges.ForEachInRange(image_end, page_end - image_end, add_download);
    gpu_modified_ranges.Subtract(page_start, page_end - page_start);
    if (total_size_bytes == 0) {
        return;
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 16, true);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += download.offset;
    }
    scheduler.EndRendering();
    runtime.CopyBuffer(buffer, download.buffer, copies);
    scheduler.DeferOperation(
        [this, base = buffer->CpuAddr(), download, copies = std::move(copies)]() {
            download.Invalidate();
            auto* memory = Core::Memory::Instance();
            for (const auto& copy : copies) {
                const VAddr copy_device_addr = base + copy.srcOffset;
                const u64 dst_offset = copy.dstOffset - download.offset;
                memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr),
                                        download.mapped + dst_offset, copy.size);
            }
            staging_pool.FreeDeferred(download);
        });
}

void BufferCache::SynchronizeDmaBuffers() {
    const PerfStats::ScopedCost cost{PerfStats::Cost::Dma};
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    const VideoCore::PerfStats::ScopedCost cost{VideoCore::PerfStats::Cost::Buffers};

    const bool has_bda = instance.IsBufferDeviceAddressSupported();
    const vk::MemoryAllocateFlagsInfo alloc_flags = {
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info = {
        .pNext = has_bda ? &alloc_flags : nullptr,
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        if (!has_bda) {
            continue;
        }
        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    if (has_bda) {
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    }
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (!copies.empty()) {
        const VideoCore::PerfStats::ScopedCost cost{VideoCore::PerfStats::Cost::Buffers};
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
        VideoCore::PerfStats::frames.AddGpuCount(VideoCore::PerfStats::Cost::BufferBytes,
                                                 total_size_bytes);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
