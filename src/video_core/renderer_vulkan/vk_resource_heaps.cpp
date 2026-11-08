// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "common/alignment.h"
#include "common/assert.h"
#include "shader_recompiler/info.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_resource_heaps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/sampler.h"

namespace Vulkan {

namespace {

/// Resource slots for draws to write into; at 32 bytes a slot this is 16 MiB.
constexpr u32 MaxResourceSlots = 512 * 1024;

/// Of those, slots kept by image views (and null descriptors) for as long as they live.
constexpr u32 KeptResourceSlots = 64 * 1024;

constexpr u32 NoBorder = ~0u;

u32 AlignedSize(vk::DeviceSize size, vk::DeviceSize alignment) {
    return static_cast<u32>(Common::AlignUp(size, std::max<vk::DeviceSize>(alignment, 1)));
}

vk::DescriptorSetAndBindingMappingEXT IndirectMapping(u32 set, u32 binding, u32 stage,
                                                      u32 index_slot, u32 stride, bool array) {
    vk::DescriptorSetAndBindingMappingEXT mapping{
        .descriptorSet = set,
        .firstBinding = binding,
        .bindingCount = 1,
        .resourceMask = vk::SpirvResourceTypeFlagBitsEXT::eAll,
    };
    const u32 push_offset = Shader::HeapPushData::StageTableOffset(stage);
    const u32 address_offset =
        Shader::StageTable::IndicesOffset + index_slot * static_cast<u32>(sizeof(u32));
    if (array) {
        // Each element of a mip array is its own descriptor with its own index.
        mapping.source = vk::DescriptorMappingSourceEXT::eHeapWithIndirectIndexArray;
        mapping.sourceData.indirectIndexArray = vk::DescriptorMappingSourceIndirectIndexArrayEXT{
            .heapOffset = 0,
            .pushOffset = push_offset,
            .addressOffset = address_offset,
            .heapIndexStride = stride,
        };
    } else {
        mapping.source = vk::DescriptorMappingSourceEXT::eHeapWithIndirectIndex;
        mapping.sourceData.indirectIndex = vk::DescriptorMappingSourceIndirectIndexEXT{
            .heapOffset = 0,
            .pushOffset = push_offset,
            .addressOffset = address_offset,
            .heapIndexStride = stride,
            .heapArrayStride = stride,
        };
    }
    return mapping;
}

} // Anonymous namespace

HeapGeometry HeapGeometry::Of(const Instance& instance) {
    const auto& props = instance.DescriptorHeapProperties();
    const vk::DeviceSize resource_size =
        std::max(props.bufferDescriptorSize, props.imageDescriptorSize);
    const vk::DeviceSize resource_alignment =
        std::max(props.bufferDescriptorAlignment, props.imageDescriptorAlignment);
    return HeapGeometry{
        .resource_stride = AlignedSize(resource_size, resource_alignment),
        .sampler_stride =
            AlignedSize(props.samplerDescriptorSize, props.samplerDescriptorAlignment),
    };
}

void StageHeapMappings::Build(const Shader::Info& stage, const HeapGeometry& geometry) {
    // Bindings are numbered per stage in this order, and each one's index sits in the table at
    // the same position as its binding number.
    mappings.clear();
    const u32 logical_stage = static_cast<u32>(stage.sw_stage);
    u32 binding = 0;
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        mappings.push_back(
            IndirectMapping(0, binding, logical_stage, binding, geometry.resource_stride, false));
        ++binding;
    }
    for (const auto& image : stage.images) {
        const u32 num_bindings = image.NumBindings(stage);
        mappings.push_back(IndirectMapping(0, binding, logical_stage, binding,
                                           geometry.resource_stride, num_bindings > 1));
        binding += num_bindings;
    }
    for (size_t i = 0; i < stage.samplers.size(); ++i) {
        mappings.push_back(
            IndirectMapping(0, binding, logical_stage, binding, geometry.sampler_stride, false));
        ++binding;
    }
    vk::DescriptorSetAndBindingMappingEXT table{
        .descriptorSet = Shader::StageTable::DescriptorSet,
        .firstBinding = Shader::StageTable::Binding,
        .bindingCount = 1,
        .resourceMask = vk::SpirvResourceTypeFlagBitsEXT::eUniformBuffer,
        .source = vk::DescriptorMappingSourceEXT::ePushAddress,
    };
    table.sourceData.pushAddressOffset = Shader::HeapPushData::StageTableOffset(logical_stage);
    mappings.push_back(table);
    info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT{
        .mappingCount = static_cast<u32>(mappings.size()),
        .pMappings = mappings.data(),
    };
}

ResourceHeaps::ResourceHeaps(const Instance& instance_, Scheduler& scheduler_,
                             VideoCore::StreamBuffer& table_buffer_)
    : instance{instance_}, scheduler{scheduler_}, table_buffer{table_buffer_},
      geometry{HeapGeometry::Of(instance_)} {
    const auto& props = instance.DescriptorHeapProperties();

    // Each heap keeps the driver's reserved range at its end, after the slots we write.
    const u64 resource_reserved = Common::AlignUp(props.minResourceHeapReservedRange,
                                                  std::max<u64>(props.resourceHeapAlignment, 1));
    const u64 max_resource_slots =
        (props.maxResourceHeapSize - resource_reserved) / geometry.resource_stride;
    const u32 total_slots = static_cast<u32>(std::min<u64>(max_resource_slots, MaxResourceSlots));
    const u64 resource_size = u64(total_slots) * geometry.resource_stride;
    const char* keep_env = std::getenv("SHADPS4_KEEP_DESCRIPTORS");
    keep_descriptors =
        (keep_env == nullptr || keep_env[0] != '0') && total_slots >= 2 * KeptResourceSlots;
    // The ring is what's left after the kept slots, which are handed out lowest first.
    resource_slots = keep_descriptors ? total_slots - KeptResourceSlots : total_slots;
    if (keep_descriptors) {
        free_kept.reserve(KeptResourceSlots);
        for (u32 slot = total_slots; slot-- > resource_slots;) {
            free_kept.push_back(slot);
        }
    }
    resource_heap = std::make_unique<VideoCore::Buffer>(
        instance, 0, resource_size + resource_reserved, VideoCore::MemoryType::Stream,
        "Resource heap", vk::BufferUsageFlagBits::eDescriptorHeapEXT);
    ASSERT_MSG(!resource_heap->mapped_data.empty(), "Resource heap isn't host visible");
    resource_bind = vk::BindHeapInfoEXT{
        .heapRange =
            vk::DeviceAddressRangeEXT{
                .address = resource_heap->BufferDeviceAddress(),
                .size = resource_size + resource_reserved,
            },
        .reservedRangeOffset = resource_size,
        .reservedRangeSize = resource_reserved,
    };

    const u64 sampler_reserved = Common::AlignUp(props.minSamplerHeapReservedRange,
                                                 std::max<u64>(props.samplerHeapAlignment, 1));
    const u32 sampler_slots =
        static_cast<u32>((props.maxSamplerHeapSize - sampler_reserved) / geometry.sampler_stride);
    const u64 sampler_size = u64(sampler_slots) * geometry.sampler_stride;
    sampler_heap = std::make_unique<VideoCore::Buffer>(
        instance, 0, sampler_size + sampler_reserved, VideoCore::MemoryType::Stream, "Sampler heap",
        vk::BufferUsageFlagBits::eDescriptorHeapEXT);
    ASSERT_MSG(!sampler_heap->mapped_data.empty(), "Sampler heap isn't host visible");
    sampler_bind = vk::BindHeapInfoEXT{
        .heapRange =
            vk::DeviceAddressRangeEXT{
                .address = sampler_heap->BufferDeviceAddress(),
                .size = sampler_size + sampler_reserved,
            },
        .reservedRangeOffset = sampler_size,
        .reservedRangeSize = sampler_reserved,
    };
    // Handed out lowest first.
    free_samplers.reserve(sampler_slots);
    for (u32 slot = sampler_slots; slot-- > 0;) {
        free_samplers.push_back(slot);
    }

    LOG_INFO(Render_Vulkan,
             "Descriptor heap: {} resource slots of {} bytes ({} kept by image views), {} sampler "
             "slots of {} bytes",
             total_slots, geometry.resource_stride, keep_descriptors ? KeptResourceSlots : 0,
             sampler_slots, geometry.sampler_stride);
}

ResourceHeaps::~ResourceHeaps() = default;

u32 ResourceHeaps::AllocateResources(u32 count) {
    ASSERT_MSG(count <= resource_slots, "Draw needs {} descriptors, the heap has {}", count,
               resource_slots);
    while (!resource_segments.empty() && scheduler.IsFree(resource_segments.front().tick)) {
        resource_segments.pop_front();
    }
    if (resource_cursor + count > resource_slots) {
        // Wrapping leaves the end unused this lap. Segments from the cursor on are from the last
        // lap, so they're the oldest, and they have to be done before the start can be reused.
        while (!resource_segments.empty() && resource_segments.front().begin >= resource_cursor) {
            scheduler.Wait(resource_segments.front().tick);
            resource_segments.pop_front();
        }
        resource_cursor = 0;
    }
    const u32 begin = resource_cursor;
    const u32 end = begin + count;
    // Whatever is still in use just ahead of the cursor is the oldest work.
    while (!resource_segments.empty() && resource_segments.front().begin < end &&
           resource_segments.front().end > begin) {
        scheduler.Wait(resource_segments.front().tick);
        resource_segments.pop_front();
    }
    const u64 tick = scheduler.CurrentTick();
    if (!resource_segments.empty() && resource_segments.back().tick == tick &&
        resource_segments.back().end == begin) {
        resource_segments.back().end = end;
    } else {
        resource_segments.push_back({tick, begin, end});
    }
    resource_cursor = end;
    return begin;
}

u32 ResourceHeaps::SamplerSlot(VideoCore::Sampler& sampler) {
    if (sampler.heap_slot.Valid()) {
        return sampler.heap_slot.Index();
    }
    if (free_samplers.empty()) {
        LOG_ERROR(Render_Vulkan, "Sampler heap is full, reusing the first sampler");
        return 0;
    }
    const u32 slot = free_samplers.back();
    free_samplers.pop_back();

    const auto device = instance.GetDevice();
    vk::SamplerCreateInfo ci{};
    vk::SamplerCustomBorderColorCreateInfoEXT custom{};
    sampler.HeapCreateInfo(ci, custom);
    u32 border = NoBorder;
    vk::SamplerCustomBorderColorIndexCreateInfoEXT border_index{};
    if (sampler.CustomBorder()) {
        const auto [result, index] = device.registerCustomBorderColorEXT(custom, false);
        if (result == vk::Result::eSuccess) {
            border = index;
            border_index.index = index;
            border_index.pNext = &custom;
            ci.pNext = &border_index;
        } else {
            LOG_ERROR(Render_Vulkan, "Failed to register a custom border color: {}",
                      vk::to_string(result));
            ci.pNext = nullptr;
            ci.borderColor = vk::BorderColor::eFloatOpaqueBlack;
        }
    }
    const vk::HostAddressRangeEXT range{
        .address = sampler_heap->mapped_data.data() + u64(slot) * geometry.sampler_stride,
        .size = geometry.sampler_stride,
    };
    const auto result = device.writeSamplerDescriptorsEXT(1, &ci, &range);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to write a sampler descriptor: {}",
               vk::to_string(result));
    if (!sampler_heap->is_coherent) {
        sampler_heap->Flush(u64(slot) * geometry.sampler_stride, geometry.sampler_stride);
    }
    sampler.heap_slot = VideoCore::SamplerHeapSlot{this, slot, border};
    return slot;
}

void ResourceHeaps::ReleaseSampler(u32 index, u32 border) {
    // Samplers are only destroyed long after their last use, so the slot is free on the GPU.
    free_samplers.push_back(index);
    if (border != NoBorder) {
        instance.GetDevice().unregisterCustomBorderColorEXT(border);
    }
}

void ResourceHeaps::ReleaseKeptSlot(u32 index) {
    // Views are only destroyed once the GPU is done with them.
    std::scoped_lock lock{kept_mutex};
    free_kept.push_back(index);
}

u32 ResourceHeaps::TakeKeptSlot() {
    std::scoped_lock lock{kept_mutex};
    if (free_kept.empty()) {
        return NoSlot;
    }
    const u32 slot = free_kept.back();
    free_kept.pop_back();
    return slot;
}

void ResourceHeaps::WriteSlot(u32 slot, const vk::ResourceDescriptorInfoEXT& info) {
    const vk::HostAddressRangeEXT range{
        .address = resource_heap->mapped_data.data() + u64(slot) * geometry.resource_stride,
        .size = geometry.resource_stride,
    };
    const auto result = instance.GetDevice().writeResourceDescriptorsEXT(1, &info, &range);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to write a resource descriptor: {}",
               vk::to_string(result));
    if (!resource_heap->is_coherent) {
        resource_heap->Flush(u64(slot) * geometry.resource_stride, geometry.resource_stride);
    }
}

u32 ResourceHeaps::KeptSlot(const HeapDraw& draw, const vk::WriteDescriptorSet& write,
                            u32 element) {
    if (!keep_descriptors) {
        return NoSlot;
    }
    const auto type = write.descriptorType;
    const auto take_null = [&](u32 kind) {
        u32& slot = null_slots[kind];
        if (slot == NoSlot && (slot = TakeKeptSlot()) != NoSlot) {
            vk::ResourceDescriptorInfoEXT info{.type = type};
            if (type == vk::DescriptorType::eStorageBuffer) {
                info.data.pAddressRange = nullptr;
            } else {
                info.data.pImage = nullptr;
            }
            WriteSlot(slot, info);
        }
        return slot;
    };
    if (type == vk::DescriptorType::eStorageBuffer) {
        // Buffer ranges change from draw to draw, so only null ones are kept.
        const size_t buffer_index = (write.pBufferInfo + element) - draw.buffer_infos.data();
        return draw.buffer_addresses[buffer_index] == 0 ? take_null(0) : NoSlot;
    }
    const bool is_storage = type == vk::DescriptorType::eStorageImage;
    const size_t image_index = (write.pImageInfo + element) - draw.image_infos.data();
    const auto* view = draw.image_views[image_index];
    if (!view) {
        return take_null(is_storage ? 2 : 1);
    }
    const auto layout = draw.image_infos[image_index].imageLayout;
    auto& kept = view->heap_slots;
    if (const u32 slot = kept.Find(type, layout); slot != NoSlot) {
        return slot;
    }
    if (kept.Full()) {
        return NoSlot;
    }
    const u32 slot = TakeKeptSlot();
    if (slot == NoSlot) {
        return NoSlot;
    }
    ViewInfo v;
    const vk::ImageDescriptorInfoEXT image{
        .pView = &view->HeapCreateInfo(v.ci, v.usage, v.min_lod),
        .layout = layout,
    };
    vk::ResourceDescriptorInfoEXT info{.type = type};
    info.data.pImage = &image;
    WriteSlot(slot, info);
    kept.Add(this, slot, type, layout);
    return slot;
}

void ResourceHeaps::Bind(const HeapDraw& draw, const Shader::PushData& push_data) {
    Shader::HeapPushData heap_push{.data = push_data, .stage_tables = {}};
    const auto device = instance.GetDevice();
    const u64 table_alignment = instance.UniformMinAlignment();

    for (const auto& stage : draw.stages) {
        const auto writes = draw.writes.subspan(stage.first_write, stage.num_writes);
        // Image views and null descriptors point at their kept slots; the rest go in the ring.
        kept_slots.clear();
        u32 num_resources = 0;
        for (const auto& write : writes) {
            if (write.descriptorType == vk::DescriptorType::eSampler) {
                continue;
            }
            const bool is_buffer = write.descriptorType == vk::DescriptorType::eStorageBuffer;
            for (u32 j = 0; j < write.descriptorCount; ++j) {
                // Bound buffers are most of them and never kept (only null ones are).
                const bool bound_buffer =
                    is_buffer &&
                    draw.buffer_addresses[(write.pBufferInfo + j) - draw.buffer_infos.data()] != 0;
                const u32 kept = bound_buffer ? NoSlot : KeptSlot(draw, write, j);
                kept_slots.push_back(kept);
                num_resources += kept == NoSlot ? 1 : 0;
            }
        }

        // Before mapping the table: making room in the ring can submit work.
        const u32 first_slot = num_resources != 0 ? AllocateResources(num_resources) : 0u;

        const u32 table_size =
            Shader::StageTable::IndicesOffset + stage.num_bindings * static_cast<u32>(sizeof(u32));
        const auto [table, table_offset] = table_buffer.Map(table_size, table_alignment);
        ASSERT_MSG(table, "Failed to map a stage table of {} bytes", table_size);
        std::memcpy(table, stage.buf_offsets.data(), stage.buf_offsets.size());
        static_assert(sizeof(stage.strides) == Shader::StageTable::StridesSize);
        std::memcpy(table + Shader::StageTable::StridesOffset, stage.strides.data(),
                    sizeof(stage.strides));
        static_assert(sizeof(stage.formats) == Shader::StageTable::FormatsSize);
        std::memcpy(table + Shader::StageTable::FormatsOffset, stage.formats.data(),
                    stage.num_formats * sizeof(Shader::RuntimeBufferFormat));
        static_assert(sizeof(stage.image_swizzles) == Shader::StageTable::ImageSwizzlesSize);
        std::memcpy(table + Shader::StageTable::ImageSwizzlesOffset, stage.image_swizzles.data(),
                    sizeof(stage.image_swizzles));
        static_assert(sizeof(stage.target_swizzles) == Shader::StageTable::TargetSwizzlesSize);
        std::memcpy(table + Shader::StageTable::TargetSwizzlesOffset, stage.target_swizzles.data(),
                    sizeof(stage.target_swizzles));
        static_assert(sizeof(stage.vertex_formats) == Shader::StageTable::VertexFormatsSize);
        std::memcpy(table + Shader::StageTable::VertexFormatsOffset, stage.vertex_formats.data(),
                    stage.num_vertex_formats * sizeof(Shader::RuntimeVertexFormat));
        u32* indices = reinterpret_cast<u32*>(table + Shader::StageTable::IndicesOffset);
        resource_infos.resize(num_resources);
        resource_ranges.resize(num_resources);
        address_ranges.resize(num_resources);
        image_descs.resize(num_resources);
        view_infos.resize(num_resources);

        u32 next = 0;
        size_t next_kept = 0;
        for (const auto& write : writes) {
            for (u32 j = 0; j < write.descriptorCount; ++j) {
                const u32 binding = write.dstBinding + j;
                ASSERT(binding < stage.num_bindings);
                if (write.descriptorType == vk::DescriptorType::eSampler) {
                    const size_t image_index = (write.pImageInfo + j) - draw.image_infos.data();
                    indices[binding] = draw.sampler_slots[image_index];
                    continue;
                }
                if (const u32 kept = kept_slots[next_kept++]; kept != NoSlot) {
                    indices[binding] = kept;
                    continue;
                }
                const u32 slot = first_slot + next;
                indices[binding] = slot;
                auto& info = resource_infos[next];
                info = vk::ResourceDescriptorInfoEXT{.type = write.descriptorType};
                if (write.descriptorType == vk::DescriptorType::eStorageBuffer) {
                    const size_t buffer_index = (write.pBufferInfo + j) - draw.buffer_infos.data();
                    const vk::DeviceAddress address = draw.buffer_addresses[buffer_index];
                    if (address != 0) {
                        address_ranges[next] = vk::DeviceAddressRangeEXT{
                            .address = address,
                            .size = draw.buffer_infos[buffer_index].range,
                        };
                        info.data.pAddressRange = &address_ranges[next];
                    } else {
                        info.data.pAddressRange = nullptr;
                    }
                } else {
                    const size_t image_index = (write.pImageInfo + j) - draw.image_infos.data();
                    const auto* view = draw.image_views[image_index];
                    if (view) {
                        auto& v = view_infos[next];
                        image_descs[next] = vk::ImageDescriptorInfoEXT{
                            .pView = &view->HeapCreateInfo(v.ci, v.usage, v.min_lod),
                            .layout = draw.image_infos[image_index].imageLayout,
                        };
                        info.data.pImage = &image_descs[next];
                    } else {
                        info.data.pImage = nullptr;
                    }
                }
                resource_ranges[next] = vk::HostAddressRangeEXT{
                    .address =
                        resource_heap->mapped_data.data() + u64(slot) * geometry.resource_stride,
                    .size = geometry.resource_stride,
                };
                ++next;
            }
        }
        if (num_resources != 0) {
            const auto result = device.writeResourceDescriptorsEXT(
                num_resources, resource_infos.data(), resource_ranges.data());
            ASSERT_MSG(result == vk::Result::eSuccess, "Failed to write resource descriptors: {}",
                       vk::to_string(result));
            if (!resource_heap->is_coherent) {
                resource_heap->Flush(u64(first_slot) * geometry.resource_stride,
                                     u64(num_resources) * geometry.resource_stride);
            }
        }
        table_buffer.Commit();
        heap_push.stage_tables[stage.stage] = table_buffer.BufferDeviceAddress() + table_offset;
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindResourceHeapEXT(resource_bind);
    cmdbuf.bindSamplerHeapEXT(sampler_bind);
    const vk::PushDataInfoEXT push_info{
        .offset = 0,
        .data =
            vk::HostAddressRangeConstEXT{
                .address = &heap_push,
                .size = sizeof(heap_push),
            },
    };
    cmdbuf.pushDataEXT(push_info);
}

} // namespace Vulkan
