// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

#include <zstd.h>

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
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
/// Beyond this many queued signatures or ranges, the next pass picks the rest up.
constexpr size_t MaxPending = 64 * 1024;
/// How far before its code a shader struct may start: the struct, up to 255 usage slots and
/// the input semantics.
constexpr u64 MaxHeaderLength = 2_KB;
/// The common word, then the embedded constant buffer and scratch sizes, then PgmLo.
constexpr u64 HeaderPgmLoOffset = 8;
/// The saved list: magic, version, the uncompressed size, then the entries compressed.
constexpr u32 SaveMagic = 0x4e435353; // "SSCN"
constexpr u32 SaveVersion = 3;
constexpr u64 SaveInterval = 30'000'000'000ULL;
/// Bigger than any real list, so a corrupt size can't ask for a huge allocation.
constexpr u64 MaxSaveSize = 1_GB;

/// What file reads and loads queued for the scanner. Static, so a game thread can queue while
/// the scanner is being destroyed.
struct Pending {
    std::atomic<bool> enabled{false};
    std::mutex mutex;
    std::condition_variable_any cv;
    std::vector<u64> infos;
    std::vector<std::pair<u64, u64>> ranges;
};
Pending pending;

} // Anonymous namespace

void ShaderScanner::NoteRead(const void* data, u64 addr, u64 size) {
    if (!pending.enabled.load(std::memory_order_relaxed) || size < sizeof(AmdGpu::BinaryInfo)) {
        return;
    }
    const auto* bytes = static_cast<const u8*>(data);
    const u8* end = bytes + size - Signature.size() + 1;
    std::vector<u64> found;
    for (const u8* it = bytes;
         (it = static_cast<const u8*>(std::memchr(it, Signature[0], end - it))) != nullptr; ++it) {
        if (std::memcmp(it, Signature.data(), Signature.size()) == 0) {
            found.push_back(addr + (it - bytes));
        }
    }
    if (found.empty()) {
        return;
    }
    {
        std::scoped_lock lock{pending.mutex};
        if (pending.infos.size() + found.size() > MaxPending) {
            return;
        }
        pending.infos.insert(pending.infos.end(), found.begin(), found.end());
    }
    pending.cv.notify_one();
}

void ShaderScanner::NoteFilled(u64 addr, u64 size) {
    if (!pending.enabled.load(std::memory_order_relaxed) || size == 0) {
        return;
    }
    {
        std::scoped_lock lock{pending.mutex};
        if (pending.ranges.size() >= MaxPending) {
            return;
        }
        pending.ranges.emplace_back(addr, size);
    }
    pending.cv.notify_one();
}

ShaderScanner::ShaderScanner() {
#ifdef __linux__
    fd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        LOG_WARNING(Render, "Shader scan unavailable: can't open /proc/self/mem");
        return;
    }
    pending.enabled = true;
    start_ns = PerfStats::NowNs();
    if (const auto serial = Common::ElfInfo::Instance().GameSerial(); !serial.empty()) {
        save_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                    (std::string{serial} + "_shaders.bin");
    }
    thread = std::jthread([this](std::stop_token stop) { Run(stop); });
#else
    LOG_INFO(Render, "Shader scan is only available on Linux for now");
#endif
}

ShaderScanner::~ShaderScanner() {
    pending.enabled = false;
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
        Save();
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
    Load();
    load_done = true;
    load_done.notify_all();
    last_save_ns = PerfStats::NowNs();
    auto next_pass = std::chrono::steady_clock::now();
    while (!stop.stop_requested()) {
        ScanPending();
        if (std::chrono::steady_clock::now() >= next_pass) {
            ScanOnce();
            next_pass = std::chrono::steady_clock::now() + ScanInterval;
            if (PerfStats::NowNs() - last_save_ns >= SaveInterval) {
                Save();
            }
        }
        std::unique_lock lock{pending.mutex};
        pending.cv.wait_until(lock, stop, next_pass,
                              [] { return !pending.infos.empty() || !pending.ranges.empty(); });
    }
}

void ShaderScanner::ScanPending() {
    std::vector<u64> infos;
    std::vector<std::pair<u64, u64>> ranges;
    {
        std::scoped_lock lock{pending.mutex};
        infos.swap(pending.infos);
        ranges.swap(pending.ranges);
    }
    if (infos.empty() && ranges.empty()) {
        return;
    }
    const u64 start = PerfStats::NowNs();
    u32 found = 0;
    for (const u64 info : infos) {
        found += TryAdd(info) ? 1 : 0;
    }
    const u32 before = pass_new;
    std::vector<u8> chunk;
    u64 scanned = 0;
    for (const auto& [addr, size] : ranges) {
        ScanRange(addr, size, chunk);
        scanned += size;
    }
    found += pass_new - before;
    pass_new = before;
    pending_new += found;
    scan_ns.fetch_add(PerfStats::NowNs() - start, std::memory_order_relaxed);
    scan_bytes.fetch_add(scanned, std::memory_order_relaxed);
}

void ShaderScanner::ScanOnce() {
    const u64 start = PerfStats::NowNs();
    auto ranges = Core::Memory::Instance()->MappedRanges();
    // Most passes only read ranges mapped since the last one; every few passes read everything,
    // for shaders loaded into memory that was already mapped.
    // Measurement for now (SHADPS4_SCAN_EARLY_FULL_S): every pass reads everything for this many
    // seconds after start, for shaders the game unpacks itself into memory mapped earlier.
    static const u64 early_full_ns = [] {
        const char* value = std::getenv("SHADPS4_SCAN_EARLY_FULL_S");
        return value ? std::strtoull(value, nullptr, 10) * 1'000'000'000ULL : 0;
    }();
    const bool full = pass_index++ % FullPassEvery == 0 || start - start_ns < early_full_ns;
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
    if (pass_new > 0 || pending_new > 0 || full) {
        std::scoped_lock lock{mutex};
        LOG_INFO(Render,
                 "Shader scan ({}) read {} MB in {:.1f} ms ({} MB unreadable): {} new shaders, "
                 "{} found from file reads and loads since the last pass, {} known",
                 full ? "full" : "new ranges", scanned >> 20, elapsed / 1e6, pass_unreadable >> 20,
                 pass_new, pending_new, shaders.size());
        pending_new = 0;
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

bool ShaderScanner::TryAdd(u64 info_addr) {
    AmdGpu::BinaryInfo info;
    if (!Read(info_addr, &info, sizeof(info)) || !info.Valid() || info.length == 0 ||
        info.length > MaxCodeLength || info.length % sizeof(u32) != 0) {
        return false;
    }
    {
        // A copy found again is only of use if the first had no shader struct in front: games
        // copy the code alone into GPU memory, and that copy can be found before the file's.
        std::scoped_lock lock{mutex};
        if (const auto it = shaders.find(info.shader_hash);
            it != shaders.end() && !it->second.header.empty()) {
            return false;
        }
    }
    // The code ends a little before its info (usage metadata sits between), and starts with
    // s_mov_b32 vcc_hi whose literal points at the info.
    const u64 window = info.length + 512;
    if (window > info_addr) {
        return false;
    }
    std::vector<u32> before(window / sizeof(u32));
    const u64 window_base = info_addr - window;
    if (!Read(window_base, before.data(), window)) {
        return false;
    }
    for (u64 k = 1; k * 8 <= window; ++k) {
        const u64 code_addr = info_addr - k * 8;
        const u64 index = (code_addr - window_base) / sizeof(u32);
        if (before[index] != TokenMovVccHi || before[index + 1] + 1 != k) {
            continue;
        }
        if (code_addr + info.length > info_addr) {
            return false;
        }
        Shader shader{
            .code = {before.begin() + index, before.begin() + index + info.length / sizeof(u32)},
            .header = ReadHeader(code_addr),
            .type = static_cast<u8>(info.type),
            .found_ns = PerfStats::NowNs(),
        };
        const bool has_header = !shader.header.empty();
        std::scoped_lock lock{mutex};
        if (const auto it = shaders.find(info.shader_hash); it != shaders.end()) {
            if (!has_header || !it.value().header.empty()) {
                return false;
            }
            it.value().header = std::move(shader.header);
            ++with_registers;
            found_order.push_back(info.shader_hash);
            dirty = true;
            return false;
        }
        if (shaders.try_emplace(info.shader_hash, std::move(shader)).second) {
            ++pass_new;
            dirty = true;
            if (has_header) {
                ++with_registers;
                found_order.push_back(info.shader_hash);
            }
            return true;
        }
        return false;
    }
    return false;
}

std::vector<u8> ShaderScanner::ReadHeader(u64 code_addr) const {
    const u64 window = std::min(MaxHeaderLength, code_addr);
    std::vector<u8> bytes(window);
    if (window < HeaderPgmLoOffset + 8 || !Read(code_addr - window, bytes.data(), window)) {
        return {};
    }
    for (u64 len = HeaderPgmLoOffset + 8; len <= window; ++len) {
        const u8* start = bytes.data() + window - len;
        u32 pgm_lo;
        u32 pgm_hi;
        std::memcpy(&pgm_lo, start + HeaderPgmLoOffset, sizeof(pgm_lo));
        std::memcpy(&pgm_hi, start + HeaderPgmLoOffset + 4, sizeof(pgm_hi));
        if (pgm_lo == len && pgm_hi == 0xFFFFFFFF) {
            return {start, start + len};
        }
    }
    return {};
}

namespace {

template <typename T>
void Put(std::vector<u8>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
bool Get(std::span<const u8>& in, T& value) {
    if (in.size() < sizeof(T)) {
        return false;
    }
    std::memcpy(&value, in.data(), sizeof(T));
    in = in.subspan(sizeof(T));
    return true;
}

} // Anonymous namespace

void ShaderScanner::Load() {
    if (save_path.empty()) {
        return;
    }
    std::ifstream file{save_path, std::ios::binary};
    if (!file) {
        return;
    }
    u32 magic{};
    u32 version{};
    u64 raw_size{};
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&raw_size), sizeof(raw_size));
    if (!file || magic != SaveMagic || version != SaveVersion || raw_size > MaxSaveSize) {
        LOG_WARNING(Render, "Ignoring the saved shader list {}: unknown format",
                    save_path.string());
        return;
    }
    const std::vector<char> packed((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
    std::vector<u8> raw(raw_size);
    const size_t got = ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size());
    if (ZSTD_isError(got) || got != raw_size) {
        LOG_WARNING(Render, "Ignoring the saved shader list {}: corrupt", save_path.string());
        return;
    }
    std::span<const u8> in{raw};
    u32 count{};
    if (!Get(in, count)) {
        return;
    }
    const u64 now = PerfStats::NowNs();
    std::scoped_lock lock{mutex};
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 type{};
        u8 stage{};
        u32 code_dwords{};
        u32 header_bytes{};
        if (!Get(in, hash) || !Get(in, type) || !Get(in, stage) || !Get(in, code_dwords) ||
            !Get(in, header_bytes) ||
            code_dwords > MaxCodeLength / sizeof(u32) || header_bytes > MaxHeaderLength ||
            in.size() < code_dwords * sizeof(u32) + header_bytes) {
            LOG_WARNING(Render, "Saved shader list {} is cut short after {} shaders",
                        save_path.string(), i);
            break;
        }
        if (code_dwords == 0) {
            break;
        }
        Shader shader{.type = type, .found_ns = now, .stage = stage};
        shader.code.resize(code_dwords);
        std::memcpy(shader.code.data(), in.data(), code_dwords * sizeof(u32));
        in = in.subspan(code_dwords * sizeof(u32));
        shader.header.assign(in.begin(), in.begin() + header_bytes);
        in = in.subspan(header_bytes);
        const bool has_header = !shader.header.empty();
        if (shaders.try_emplace(hash, std::move(shader)).second) {
            ++loaded;
            if (has_header) {
                ++with_registers;
                found_order.push_back(hash);
            }
        }
    }
    LOG_INFO(Render, "Shader scan: {} shaders from earlier runs ({} with stage registers)",
             loaded, with_registers);
}

void ShaderScanner::Save() {
    if (save_path.empty()) {
        return;
    }
    std::vector<u8> raw;
    u32 count{};
    {
        std::scoped_lock lock{mutex};
        if (!dirty) {
            return;
        }
        dirty = false;
        count = static_cast<u32>(shaders.size());
        Put(raw, count);
        for (const auto& [hash, shader] : shaders) {
            Put(raw, hash);
            Put(raw, shader.type);
            Put(raw, shader.stage);
            Put(raw, static_cast<u32>(shader.code.size()));
            Put(raw, static_cast<u32>(shader.header.size()));
            const auto* code = reinterpret_cast<const u8*>(shader.code.data());
            raw.insert(raw.end(), code, code + shader.code.size() * sizeof(u32));
            raw.insert(raw.end(), shader.header.begin(), shader.header.end());
        }
    }
    last_save_ns = PerfStats::NowNs();
    std::vector<char> packed(ZSTD_compressBound(raw.size()));
    const size_t size = ZSTD_compress(packed.data(), packed.size(), raw.data(), raw.size(), 3);
    if (ZSTD_isError(size)) {
        return;
    }
    // Written aside and renamed over, so a run stopped mid-write keeps the last good list.
    auto temp = save_path;
    temp += ".tmp";
    {
        std::ofstream file{temp, std::ios::binary | std::ios::trunc};
        const u64 raw_size = raw.size();
        file.write(reinterpret_cast<const char*>(&SaveMagic), sizeof(SaveMagic));
        file.write(reinterpret_cast<const char*>(&SaveVersion), sizeof(SaveVersion));
        file.write(reinterpret_cast<const char*>(&raw_size), sizeof(raw_size));
        file.write(packed.data(), static_cast<std::streamsize>(size));
        if (!file) {
            LOG_WARNING(Render, "Can't save the shader list to {}", temp.string());
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(temp, save_path, ec);
    if (ec) {
        LOG_WARNING(Render, "Can't save the shader list to {}: {}", save_path.string(),
                    ec.message());
        return;
    }
    LOG_INFO(Render, "Shader scan: saved {} shaders ({} KB) to {}", count, size >> 10,
             save_path.string());
}

std::optional<ShaderScanner::Shader> ShaderScanner::Find(u64 hash) {
    std::scoped_lock lock{mutex};
    const auto it = shaders.find(hash);
    if (it == shaders.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<u8> ShaderScanner::StageOf(u64 hash) {
    std::scoped_lock lock{mutex};
    const auto it = shaders.find(hash);
    if (it == shaders.end() || it->second.stage == NoStage) {
        return std::nullopt;
    }
    return it->second.stage;
}

void ShaderScanner::WaitUntilLoaded() const {
    if (fd >= 0) {
        load_done.wait(false);
    }
}

std::vector<u64> ShaderScanner::KnownWithStage() {
    std::scoped_lock lock{mutex};
    std::vector<u64> hashes;
    for (const auto& [hash, shader] : shaders) {
        if (shader.stage != NoStage && !shader.header.empty()) {
            hashes.push_back(hash);
        }
    }
    std::ranges::sort(hashes);
    return hashes;
}

std::optional<u8> ShaderScanner::TypeOf(u64 hash) {
    std::scoped_lock lock{mutex};
    const auto it = shaders.find(hash);
    if (it == shaders.end()) {
        return std::nullopt;
    }
    return it->second.type;
}

std::vector<u64> ShaderScanner::TakeFound(size_t& cursor) {
    std::scoped_lock lock{mutex};
    if (cursor >= found_order.size()) {
        return {};
    }
    std::vector<u64> found(found_order.begin() + cursor, found_order.end());
    cursor = found_order.size();
    return found;
}

void ShaderScanner::NoteFirstUse(u64 hash, u8 stage) {
    const u64 now = PerfStats::NowNs();
    std::scoped_lock lock{mutex};
    ++first_uses;
    const auto it = shaders.find(hash);
    if (it != shaders.end()) {
        ++found_ahead;
        leads.push_back((now - it->second.found_ns) / 1e9);
        // The draw knows the stage; a saved one from an older launch can be wrong.
        if (it->second.stage != stage) {
            it.value().stage = stage;
            dirty = true;
        }
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
        .with_registers = with_registers,
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
