// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
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
        /// The PSSL shader struct in front of the code, when the scan found one: the common
        /// word, a word of embedded constant and scratch sizes, then the stage registers from
        /// PgmLo, then input usage slots and (for pixel shaders) input semantics, up to the code.
        /// Empty when the shader came without it.
        std::vector<u8> header;
        u8 type;
        /// When the scan found it.
        u64 found_ns;
        /// The hardware stage it was first used as (a Shader::Stage), NoStage until then.
        /// Saved with it, so later launches know it before any draw.
        u8 stage = NoStage;
    };
    static constexpr u8 NoStage = 0xff;

    ShaderScanner();
    ~ShaderScanner();

    ShaderScanner(const ShaderScanner&) = delete;
    ShaderScanner& operator=(const ShaderScanner&) = delete;

    /// Whether the scanner could start on this platform.
    bool Running() const {
        return fd >= 0;
    }

    /// Called after the game reads a file into memory, to look through what it read for
    /// shaders at once instead of at the next pass. `data` is a host copy of the `size` bytes
    /// now at guest address `addr`. Cheap, and does nothing while no scanner runs.
    static void NoteRead(const void* data, u64 addr, u64 size);
    /// Called when guest memory was filled some other way, such as a module loading: scans the
    /// range at once.
    static void NoteFilled(u64 addr, u64 size);

    /// The shader with this hash, if the scan found it.
    std::optional<Shader> Find(u64 hash);

    /// The binary type of the shader with this hash, if the scan found it.
    std::optional<u8> TypeOf(u64 hash);

    /// Hashes of the shaders found with their stage registers, in the order found, from
    /// `cursor` on; moves `cursor` past them.
    std::vector<u64> TakeFound(size_t& cursor);

    /// Notes the first draw or dispatch with a program, for the summary: whether the scan had
    /// found it, and how long before. Also remembers the stage it was used as.
    void NoteFirstUse(u64 hash, u8 stage);

    /// The stage a shader was first used as, in this run or an earlier one.
    std::optional<u8> StageOf(u64 hash);

    /// Waits until the shaders saved by earlier runs are loaded.
    void WaitUntilLoaded() const;

    /// Hashes of the known shaders with a stage struct and a known stage.
    std::vector<u64> KnownWithStage();

    struct Counts {
        u32 first_uses{};
        u32 found_ahead{};
        /// Median time from finding to first use, over the found ones.
        double median_lead_s{};
        u32 known{};
        /// How many of the known ones came with their stage registers.
        u32 with_registers{};
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
    /// Scans what NoteRead and NoteFilled queued.
    void ScanPending();
    /// Returns whether the shader was new.
    bool TryAdd(u64 info_addr);
    /// Finds the shader struct that the code at `code_addr` belongs to: its PgmLo is the
    /// distance from the struct to the code and its PgmHi is all ones.
    std::vector<u8> ReadHeader(u64 code_addr) const;
    bool Read(u64 addr, void* dst, u64 size) const;
    /// The shaders found in earlier runs of this game, saved so that the next launch knows them
    /// from the start instead of waiting for the scan to find them again.
    void Load();
    void Save();

    std::filesystem::path save_path;
    /// Whether shaders were found since the last save. Guarded by `mutex`.
    bool dirty{};
    u64 last_save_ns{};
    u32 loaded{};
    std::atomic<bool> load_done{false};

    int fd{-1};
    std::mutex mutex;
    tsl::robin_map<u64, Shader> shaders;
    std::vector<double> leads;
    u32 with_registers{};
    /// Hashes of the shaders with stage registers, in the order found.
    std::vector<u64> found_order;
    u32 first_uses{};
    u32 found_ahead{};
    /// Scan totals, for the per-pass log.
    u32 pass_new{};
    /// Shaders found through file reads and loads, between passes.
    u32 pending_new{};
    u32 pass_index{};
    u64 start_ns{};
    u64 pass_unreadable{};
    /// Ranges as of the last pass, sorted.
    std::vector<std::pair<VAddr, u64>> last_ranges;
    std::atomic<u64> scan_ns{};
    std::atomic<u64> scan_bytes{};
    std::jthread thread;
};

} // namespace VideoCore
