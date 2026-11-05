// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <tsl/robin_map.h>

#include "common/types.h"

namespace VideoCore {

/// Looks through the game's memory for shader binaries while it runs, so shaders are known
/// before their first draw. Runs on its own low-priority thread and reads guest memory through
/// /proc/self/mem, so page protection (write tracking, readbacks) never faults and a range unmapped
/// meanwhile just fails to read.
class ShaderScanner {
public:
    struct Shader {
        std::vector<u32> code;
        u8 type;
        /// When the scan found it.
        u64 found_ns;
    };

    ShaderScanner();
    ~ShaderScanner();

    ShaderScanner(const ShaderScanner&) = delete;
    ShaderScanner& operator=(const ShaderScanner&) = delete;

    /// Whether the scanner could start on this platform.
    bool Running() const {
        return fd >= 0;
    }

    /// The shader with this hash, if the scan found it.
    std::optional<Shader> Find(u64 hash);

    /// Notes the first draw or dispatch with a program, for the summary: whether the scan had
    /// found it, and how long before.
    void NoteFirstUse(u64 hash);

    struct Counts {
        u32 first_uses{};
        u32 found_ahead{};
        /// Median time from finding to first use, over the found ones.
        double median_lead_s{};
        u32 known{};
        /// Time and memory the scan spent reading.
        double scan_ms{};
        u64 scanned_mb{};
    };
    /// Counts since the last call (known is the running total).
    Counts TakeCounts();

private:
    void Run(std::stop_token stop);
    void ScanOnce();
    void ScanRange(u64 base, u64 size, std::vector<u8>& chunk);
    void TryAdd(u64 info_addr);
    bool Read(u64 addr, void* dst, u64 size) const;

    int fd{-1};
    std::mutex mutex;
    tsl::robin_map<u64, Shader> shaders;
    std::vector<double> leads;
    u32 first_uses{};
    u32 found_ahead{};
    /// Scan totals, for the per-pass log.
    u32 pass_new{};
    u32 pass_index{};
    u64 pass_unreadable{};
    /// Ranges as of the last pass, sorted.
    std::vector<std::pair<VAddr, u64>> last_ranges;
    std::atomic<u64> scan_ns{};
    std::atomic<u64> scan_bytes{};
    std::jthread thread;
};

} // namespace VideoCore
