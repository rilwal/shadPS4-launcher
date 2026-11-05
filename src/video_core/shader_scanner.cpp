// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

#include "common/logging/log.h"
#include "common/thread.h"
#include "core/memory.h"
#include "video_core/amdgpu/regs_shader.h"
#include "video_core/perf_stats.h"
#include "video_core/shader_scanner.h"

namespace VideoCore {

namespace {

/// The start of every BinaryInfo.
constexpr std::array<u8, 7> Signature{'O', 'r', 'b', 'S', 'h', 'd', 'r'};
/// s_mov_b32 vcc_hi, literal: the first instruction of every shader. Its literal is the
/// distance from the code to its BinaryInfo, in 8-byte units minus one.
constexpr u32 TokenMovVccHi = 0xBEEB03FF;
constexpr u64 ChunkSize = 4_MB;
/// Shaders are at most a few tens of KB; anything larger isn't one.
constexpr u32 MaxCodeLength = 1_MB;
constexpr auto ScanInterval = std::chrono::seconds{3};
constexpr u32 FullPassEvery = 8;
constexpr u64 UnreadablePieceSize = 64_KB;

} // Anonymous namespace

ShaderScanner::ShaderScanner() {
#ifdef __linux__
    fd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        LOG_WARNING(Render, "Shader scan unavailable: can't open /proc/self/mem");
        return;
    }
    thread = std::jthread([this](std::stop_token stop) { Run(stop); });
#else
    LOG_INFO(Render, "Shader scan is only available on Linux for now");
#endif
}

ShaderScanner::~ShaderScanner() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
#ifdef __linux__
    if (fd >= 0) {
        close(fd);
    }
#endif
}

bool ShaderScanner::Read(u64 addr, void* dst, u64 size) const {
#ifdef __linux__
    auto* out = static_cast<u8*>(dst);
    while (size > 0) {
        const ssize_t got = pread(fd, out, size, static_cast<off_t>(addr));
        if (got <= 0) {
            return false;
        }
        out += got;
        addr += got;
        size -= got;
    }
    return true;
#else
    return false;
#endif
}

void ShaderScanner::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:ShaderScan");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    std::mutex wait_mutex;
    std::condition_variable_any cv;
    while (!stop.stop_requested()) {
        ScanOnce();
        std::unique_lock lock{wait_mutex};
        cv.wait_for(lock, stop, ScanInterval, [] { return false; });
    }
}

void ShaderScanner::ScanOnce() {
    const u64 start = PerfStats::NowNs();
    auto ranges = Core::Memory::Instance()->MappedRanges();
    // Most passes only read ranges mapped since the last one; every few passes read everything,
    // for shaders loaded into memory that was already mapped.
    const bool full = pass_index++ % FullPassEvery == 0;
    std::vector<u8> chunk;
    u64 scanned = 0;
    pass_new = 0;
    pass_unreadable = 0;
    for (const auto& range : ranges) {
        if (!full && std::ranges::binary_search(last_ranges, range)) {
            continue;
        }
        const u64 unreadable_before = pass_unreadable;
        ScanRange(range.first, range.second, chunk);
        scanned += range.second;
        if (pass_index == 1 && pass_unreadable != unreadable_before) {
            LOG_INFO(Render, "Shader scan can't read {} KB of {:#x}..{:#x}",
                     (pass_unreadable - unreadable_before) >> 10, range.first,
                     range.first + range.second);
        }
    }
    last_ranges = std::move(ranges);
    const u64 elapsed = PerfStats::NowNs() - start;
    scan_ns.fetch_add(elapsed, std::memory_order_relaxed);
    scan_bytes.fetch_add(scanned, std::memory_order_relaxed);
    if (pass_new > 0 || full) {
        std::scoped_lock lock{mutex};
        LOG_INFO(Render,
                 "Shader scan ({}) read {} MB in {:.1f} ms ({} MB unreadable): {} new shaders, "
                 "{} known",
                 full ? "full" : "new ranges", scanned >> 20, elapsed / 1e6, pass_unreadable >> 20,
                 pass_new, shaders.size());
    }
}

void ShaderScanner::ScanRange(u64 base, u64 size, std::vector<u8>& chunk) {
    // Shaders in executables are often only 2-byte aligned, so this searches bytes. Chunks
    // overlap by a signature's length so one across a boundary is still seen.
    for (u64 offset = 0; offset < size; offset += ChunkSize - Signature.size()) {
        const u64 len = std::min(ChunkSize, size - offset);
        if (len < sizeof(AmdGpu::BinaryInfo)) {
            break;
        }
        chunk.resize(len);
        if (!Read(base + offset, chunk.data(), len)) {
            // Part of the chunk can't be read (never backed, or unmapped meanwhile): read what
            // can be, page by page, and leave the rest zero.
            for (u64 page = 0; page < len; page += UnreadablePieceSize) {
                const u64 piece = std::min(UnreadablePieceSize, len - page);
                if (!Read(base + offset + page, chunk.data() + page, piece)) {
                    std::memset(chunk.data() + page, 0, piece);
                    pass_unreadable += piece;
                }
            }
        }
        // memchr is vectorised, and the signature's first byte is rare enough in most data
        // that checking the rest at each hit costs little.
        const u8* data = chunk.data();
        const u8* end = data + len - Signature.size() + 1;
        for (const u8* it = data;
             (it = static_cast<const u8*>(std::memchr(it, Signature[0], end - it))) != nullptr;
             ++it) {
            if (std::memcmp(it, Signature.data(), Signature.size()) == 0) {
                TryAdd(base + offset + (it - data));
            }
        }
        if (len < ChunkSize) {
            break;
        }
    }
}

void ShaderScanner::TryAdd(u64 info_addr) {
    AmdGpu::BinaryInfo info;
    if (!Read(info_addr, &info, sizeof(info)) || !info.Valid() || info.length == 0 ||
        info.length > MaxCodeLength || info.length % sizeof(u32) != 0) {
        return;
    }
    {
        std::scoped_lock lock{mutex};
        if (shaders.contains(info.shader_hash)) {
            return;
        }
    }
    // The code ends a little before its info (usage metadata sits between), and starts with
    // s_mov_b32 vcc_hi whose literal points at the info.
    const u64 window = info.length + 512;
    if (window > info_addr) {
        return;
    }
    std::vector<u32> before(window / sizeof(u32));
    const u64 window_base = info_addr - window;
    if (!Read(window_base, before.data(), window)) {
        return;
    }
    for (u64 k = 1; k * 8 <= window; ++k) {
        const u64 code_addr = info_addr - k * 8;
        const u64 index = (code_addr - window_base) / sizeof(u32);
        if (before[index] != TokenMovVccHi || before[index + 1] + 1 != k) {
            continue;
        }
        if (code_addr + info.length > info_addr) {
            return;
        }
        Shader shader{
            .code = {before.begin() + index, before.begin() + index + info.length / sizeof(u32)},
            .type = static_cast<u8>(info.type),
            .found_ns = PerfStats::NowNs(),
        };
        std::scoped_lock lock{mutex};
        if (shaders.try_emplace(info.shader_hash, std::move(shader)).second) {
            ++pass_new;
        }
        return;
    }
}

std::optional<ShaderScanner::Shader> ShaderScanner::Find(u64 hash) {
    std::scoped_lock lock{mutex};
    const auto it = shaders.find(hash);
    if (it == shaders.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ShaderScanner::NoteFirstUse(u64 hash) {
    const u64 now = PerfStats::NowNs();
    std::scoped_lock lock{mutex};
    ++first_uses;
    const auto it = shaders.find(hash);
    if (it != shaders.end()) {
        ++found_ahead;
        leads.push_back((now - it->second.found_ns) / 1e9);
    } else {
        LOG_INFO(Render, "Shader scan hadn't found program {:#x} before its first use", hash);
    }
}

ShaderScanner::Counts ShaderScanner::TakeCounts() {
    std::scoped_lock lock{mutex};
    Counts counts{
        .first_uses = std::exchange(first_uses, 0),
        .found_ahead = std::exchange(found_ahead, 0),
        .known = static_cast<u32>(shaders.size()),
        .scan_ms = scan_ns.exchange(0, std::memory_order_relaxed) / 1e6,
        .scanned_mb = scan_bytes.exchange(0, std::memory_order_relaxed) >> 20,
    };
    if (!leads.empty()) {
        std::ranges::nth_element(leads, leads.begin() + leads.size() / 2);
        counts.median_lead_s = leads[leads.size() / 2];
        leads.clear();
    }
    return counts;
}

} // namespace VideoCore
