// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <tsl/robin_map.h>

#include "common/types.h"
#include "video_core/shader_list.h"

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
        /// The PSSL reflection's resource records after the BinaryInfo (24 bytes each: api slot,
        /// size, type word with the dimension in byte 0, constant count, unknown, name), when
        /// the copy with the struct had them. Empty otherwise.
        std::vector<u8> reflection;
        u8 type;
        /// When the scan found it.
        u64 found_ns;
        /// Guest address of the copy with the shader struct, 0 until the scan finds it in this
        /// run (shaders from the saved list start without one).
        u64 addr{};
        /// The hardware stage it was first used as (a Shader::Stage), NoStage until then.
        /// Saved with it, so later launches know it before any draw.
        u8 stage = NoStage;
        /// Its place in the order of first uses over this and earlier runs, from 1; 0 if the
        /// game hasn't used it yet. Saved with it.
        u32 use_order{};
    };
    static constexpr u8 NoStage = ShaderFiles::NoStage;

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

    struct Place {
        /// Guest address of the copy with the shader struct, 0 when not known.
        u64 addr;
        /// Shader::use_order.
        u32 use_order;
    };
    /// Where the shader with this hash was found, and when it was first used.
    Place PlaceOf(u64 hash);

    /// Number of recent first uses remembered by address.
    static constexpr size_t NumRecentUses = 8;
    /// Guest addresses of the latest first uses of found shaders, 0 where none yet. Games keep
    /// the shaders of a level or material together, so what is found near them is likely
    /// drawn soon.
    std::array<u64, NumRecentUses> RecentUses();

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

    /// The pixel shader semantic base (see PipelineCache::LearnSemanticBase) an earlier run
    /// learned, saved with the list.
    std::optional<int> SemanticBase();
    void SetSemanticBase(int base);
    /// Tiling pipelines (TileManager's packed keys) the game used, saved with the list so they
    /// can be built ahead next time.
    std::vector<u64> TilingKeys();
    void NoteTilingKey(u64 key);

    /// Draw states recorded at first uses, as ParseDrawState's blobs, so later launches can
    /// compile shaders for exactly what they were drawn with. Saved apart from the list, in a
    /// file with no shader code, so it can be shared (see ShaderFiles::StateKeyBytes).
    void NoteState(u64 hash, std::vector<u8> state);
    std::vector<std::vector<u8>> StatesOf(u64 hash);
    /// The hardware stage a shader's recorded states were drawn as, if it has any.
    std::optional<u8> StateStageOf(u64 hash);

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

    /// Events for the run's timeline file (<cache>/<serial>_timeline/<start time>.csv), from
    /// which shader_tool's simulate replays when shaders became known and were used:
    ///   F found (a: guest address, b: 1 with struct, 2 with reflection)
    ///   H found again with the struct it lacked (a: address, b: as F)
    ///   L loaded from the saved list (a: saved stage, b: use order, high bit with struct)
    ///   Q queued for a worker (a: stage)
    ///   W worker finished it (a: start, ns since the scan started; b: 1 if it compiled)
    ///   U first use (a: stage, b: 's' served, 'd' compiled early but differed, 'p' found but
    ///     not compiled yet, 'n' not found)
    ///   D a worker finished it after its first use
    ///   J the first use waited for the worker compiling it (a: ns waited)
    ///   B semantic base learned (a: base)
    void NoteEvent(char event, u64 hash, u64 a = 0, u64 b = 0);
    /// Nanoseconds since the scanner started, the timeline's clock.
    u64 Clock() const;

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
    /// The reflection's resource records after the BinaryInfo at `info_addr`, if they look
    /// real: a sane count and a first record whose name is a printable string.
    std::vector<u8> ReadReflection(u64 info_addr) const;
    bool Read(u64 addr, void* dst, u64 size) const;
    /// The shaders found in earlier runs of this game, saved so that the next launch knows them
    /// from the start instead of waiting for the scan to find them again.
    void Load();
    void LoadStates();
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
    std::array<u64, NumRecentUses> recent_uses{};
    /// The last use_order given.
    u32 use_counter{};
    int semantic_base{-1};
    std::vector<u64> tiling_keys;
    ShaderFiles::ShaderStates states;
    std::filesystem::path states_path;
    size_t recent_use_index{};
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

    struct Event {
        u64 t;
        u64 hash;
        u64 a;
        u64 b;
        char event;
    };
    /// Appends the events noted since the last call to the timeline file.
    void FlushTimeline();
    std::mutex timeline_mutex;
    std::vector<Event> timeline;
    std::filesystem::path timeline_path;
    bool timeline_started{};

    std::jthread thread;
};

} // namespace VideoCore
