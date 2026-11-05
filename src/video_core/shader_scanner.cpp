// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
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

/// "OrbS" as a little-endian word: BinaryInfo is word aligned.
constexpr u32 SignatureWord = 0x5362724F;
/// s_mov_b32 vcc_hi, literal: the first instruction of every shader. Its literal is the
/// distance from the code to its BinaryInfo, in 8-byte units minus one.
constexpr u32 TokenMovVccHi = 0xBEEB03FF;
constexpr u64 ChunkSize = 4_MB;
/// Shaders are at most a few tens of KB; anything larger isn't one.
constexpr u32 MaxCodeLength = 1_MB;
constexpr auto ScanInterval = std::chrono::seconds{3};

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
    const auto ranges = Core::Memory::Instance()->ShaderCodeRanges();
    std::vector<u8> chunk;
    u64 scanned = 0;
    pass_new = 0;
    for (const auto& range : ranges) {
        ScanRange(range.base, range.size, chunk);
        scanned += range.size;
    }
    if (pass_new > 0) {
        std::scoped_lock lock{mutex};
        LOG_INFO(Render, "Shader scan read {} MB in {:.1f} ms: {} new shaders, {} known",
                 scanned >> 20, (PerfStats::NowNs() - start) / 1e6, pass_new, shaders.size());
    }
}

void ShaderScanner::ScanRange(u64 base, u64 size, std::vector<u8>& chunk) {
    // Chunks overlap by a word so a signature across a boundary is still seen.
    for (u64 offset = 0; offset < size; offset += ChunkSize - sizeof(u32)) {
        const u64 len = std::min(ChunkSize, size - offset);
        if (len < sizeof(AmdGpu::BinaryInfo)) {
            break;
        }
        chunk.resize(len);
        if (!Read(base + offset, chunk.data(), len)) {
            // Unmapped meanwhile, or never backed: skip the rest of the range.
            return;
        }
        const auto* words = reinterpret_cast<const u32*>(chunk.data());
        const u64 num_words = len / sizeof(u32);
        for (u64 i = 0; i < num_words; ++i) {
            if (words[i] != SignatureWord) [[likely]] {
                continue;
            }
            const u8* bytes = reinterpret_cast<const u8*>(&words[i]);
            if (i * sizeof(u32) + 8 > len || bytes[4] != 'h' || bytes[5] != 'd' ||
                bytes[6] != 'r') {
                continue;
            }
            TryAdd(base + offset + i * sizeof(u32));
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
    }
}

ShaderScanner::Counts ShaderScanner::TakeCounts() {
    std::scoped_lock lock{mutex};
    Counts counts{
        .first_uses = std::exchange(first_uses, 0),
        .found_ahead = std::exchange(found_ahead, 0),
        .known = static_cast<u32>(shaders.size()),
    };
    if (!leads.empty()) {
        std::ranges::nth_element(leads, leads.begin() + leads.size() / 2);
        counts.median_lead_s = leads[leads.size() / 2];
        leads.clear();
    }
    return counts;
}

} // namespace VideoCore
