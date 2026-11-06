// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <fstream>
#include <string_view>
#include <zstd.h>

#include "common/logging/log.h"
#include "video_core/shader_list.h"

namespace VideoCore::ShaderFiles {

namespace {

/// The saved list: magic, version, the uncompressed size, then the entries compressed.
constexpr u32 SaveMagic = 0x4e435353; // "SSCN"
constexpr u32 SaveVersion = 10;
/// Bigger than any real list, so a corrupt size can't ask for a huge allocation.
constexpr u64 MaxSaveSize = (u64{1} << 30);
/// The recorded draw states, in their own file with no shader code: magic, version, size,
/// then compressed per shader its hash, a count and the states, each a size and its bytes.
constexpr u32 StatesMagic = 0x41545353; // "SSTA"
constexpr u32 StatesVersion = 2;

/// Reads a file Save wrote: its entries, uncompressed, if the magic and version match.
std::optional<std::vector<u8>> ReadSaved(const std::filesystem::path& path, u32 want_magic,
                                         u32 want_version, std::string_view what) {
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        return std::nullopt;
    }
    u32 magic{};
    u32 version{};
    u64 raw_size{};
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&raw_size), sizeof(raw_size));
    if (!file || magic != want_magic || version != want_version || raw_size > MaxSaveSize) {
        LOG_WARNING(Render, "Ignoring the saved {} {}: unknown format", what, path.string());
        return std::nullopt;
    }
    const std::vector<char> packed((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
    std::vector<u8> raw(raw_size);
    const size_t got = ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size());
    if (ZSTD_isError(got) || got != raw_size) {
        LOG_WARNING(Render, "Ignoring the saved {} {}: corrupt", what, path.string());
        return std::nullopt;
    }
    return raw;
}

/// Compresses and writes entries, aside and renamed over, so a run stopped mid-write keeps the
/// last good file. Returns the compressed size.
std::optional<size_t> WriteSaved(const std::filesystem::path& path, u32 magic, u32 version,
                                 std::span<const u8> raw, std::string_view what) {
    std::vector<char> packed(ZSTD_compressBound(raw.size()));
    const size_t size = ZSTD_compress(packed.data(), packed.size(), raw.data(), raw.size(), 3);
    if (ZSTD_isError(size)) {
        return std::nullopt;
    }
    auto temp = path;
    temp += ".tmp";
    {
        std::ofstream file{temp, std::ios::binary | std::ios::trunc};
        const u64 raw_size = raw.size();
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&raw_size), sizeof(raw_size));
        file.write(packed.data(), static_cast<std::streamsize>(size));
        if (!file) {
            LOG_WARNING(Render, "Can't save the {} to {}", what, temp.string());
            return std::nullopt;
        }
    }
    std::error_code ec;
    std::filesystem::rename(temp, path, ec);
    if (ec) {
        LOG_WARNING(Render, "Can't save the {} to {}: {}", what, path.string(), ec.message());
        return std::nullopt;
    }
    return size;
}

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

std::optional<ShaderList> ReadShaderList(const std::filesystem::path& path) {
    const auto saved = ReadSaved(path, SaveMagic, SaveVersion, "shader list");
    if (!saved) {
        return std::nullopt;
    }
    std::span<const u8> in{*saved};
    u32 count{};
    s32 base{};
    u32 num_tiling_keys{};
    if (!Get(in, count) || !Get(in, base) || !Get(in, num_tiling_keys) ||
        num_tiling_keys > MaxTilingKeys) {
        return std::nullopt;
    }
    ShaderList list{.semantic_base = base};
    list.tiling_keys.resize(num_tiling_keys);
    for (u64& key : list.tiling_keys) {
        if (!Get(in, key)) {
            return std::nullopt;
        }
    }
    list.shaders.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        ListedShader shader{};
        u32 code_dwords{};
        u32 header_bytes{};
        u32 reflection_bytes{};
        if (!Get(in, shader.hash) || !Get(in, shader.type) || !Get(in, shader.stage) ||
            !Get(in, shader.use_order) || !Get(in, code_dwords) || !Get(in, header_bytes) ||
            !Get(in, reflection_bytes) || code_dwords > MaxCodeLength / sizeof(u32) ||
            header_bytes > MaxHeaderLength || reflection_bytes > MaxReflectionLength ||
            in.size() < code_dwords * sizeof(u32) + header_bytes + reflection_bytes) {
            LOG_WARNING(Render, "Saved shader list {} is cut short after {} shaders", path.string(),
                        i);
            break;
        }
        if (code_dwords == 0) {
            break;
        }
        shader.code.resize(code_dwords);
        std::memcpy(shader.code.data(), in.data(), code_dwords * sizeof(u32));
        in = in.subspan(code_dwords * sizeof(u32));
        shader.header.assign(in.begin(), in.begin() + header_bytes);
        in = in.subspan(header_bytes);
        shader.reflection.assign(in.begin(), in.begin() + reflection_bytes);
        in = in.subspan(reflection_bytes);
        list.shaders.push_back(std::move(shader));
    }
    return list;
}

std::optional<ShaderStates> ReadShaderStates(const std::filesystem::path& path) {
    const auto saved = ReadSaved(path, StatesMagic, StatesVersion, "shader draw states");
    if (!saved) {
        return std::nullopt;
    }
    std::span<const u8> in{*saved};
    u32 count{};
    if (!Get(in, count)) {
        return std::nullopt;
    }
    ShaderStates states;
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u32 num{};
        if (!Get(in, hash) || !Get(in, num) || num > MaxStatesPerShader) {
            break;
        }
        auto& list = states[hash];
        for (u32 k = 0; k < num; ++k) {
            u32 size{};
            if (!Get(in, size) || size > MaxStateSize || in.size() < size) {
                LOG_WARNING(Render, "Saved shader draw states {} are cut short", path.string());
                return states;
            }
            if (size > StateKeyBytes) {
                list.emplace_back(in.begin(), in.begin() + size);
            }
            in = in.subspan(size);
        }
    }
    return states;
}

ShaderListWriter::ShaderListWriter(u32 count, int semantic_base, std::span<const u64> tiling_keys) {
    Put(raw, count);
    Put(raw, static_cast<s32>(semantic_base));
    Put(raw, static_cast<u32>(tiling_keys.size()));
    for (const u64 key : tiling_keys) {
        Put(raw, key);
    }
}

void ShaderListWriter::Add(u64 hash, u8 type, u8 stage, u32 use_order, std::span<const u32> code,
                           std::span<const u8> header, std::span<const u8> reflection) {
    Put(raw, hash);
    Put(raw, type);
    Put(raw, stage);
    Put(raw, use_order);
    Put(raw, static_cast<u32>(code.size()));
    Put(raw, static_cast<u32>(header.size()));
    Put(raw, static_cast<u32>(reflection.size()));
    const auto* bytes = reinterpret_cast<const u8*>(code.data());
    raw.insert(raw.end(), bytes, bytes + code.size_bytes());
    raw.insert(raw.end(), header.begin(), header.end());
    raw.insert(raw.end(), reflection.begin(), reflection.end());
}

std::optional<size_t> ShaderListWriter::Write(const std::filesystem::path& path) const {
    return WriteSaved(path, SaveMagic, SaveVersion, raw, "shader list");
}

std::optional<size_t> WriteShaderStates(const std::filesystem::path& path,
                                        const ShaderStates& states) {
    std::vector<u8> raw;
    Put(raw, static_cast<u32>(states.size()));
    for (const auto& [hash, list] : states) {
        Put(raw, hash);
        Put(raw, static_cast<u32>(list.size()));
        for (const auto& state : list) {
            Put(raw, static_cast<u32>(state.size()));
            raw.insert(raw.end(), state.begin(), state.end());
        }
    }
    return WriteSaved(path, StatesMagic, StatesVersion, raw, "shader draw states");
}

} // namespace VideoCore::ShaderFiles
