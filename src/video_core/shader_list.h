// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <vector>
#include <tsl/robin_map.h>

#include "common/types.h"

/// The files the shader scanner keeps per game: the list of shaders it found (with their code)
/// and the draw states recorded at first uses (without code). Read by the scanner at launch and
/// by the shader tool.
namespace VideoCore::ShaderFiles {

/// Shaders are at most a few tens of KB; anything larger isn't one.
constexpr u32 MaxCodeLength = 1 << 20;
/// How far before its code a shader struct may start: the struct, up to 255 usage slots and
/// the input semantics.
constexpr u64 MaxHeaderLength = 2 << 10;
/// 256 resource records of 24 bytes.
constexpr u64 MaxReflectionLength = 6 << 10;
constexpr u32 MaxTilingKeys = 1024;
constexpr u32 MaxStatesPerShader = 8;
constexpr u32 MaxStateSize = 64 << 10;
/// A state blob starts with this many bytes that identify what it records (one with the same
/// start isn't kept twice), then its hardware stage as a byte.
constexpr size_t StateKeyBytes = 12;
constexpr u8 NoStage = 0xff;

struct ListedShader {
    u64 hash;
    u8 type;
    /// The hardware stage it was first used as, NoStage if never.
    u8 stage;
    /// Its place in the order of first uses, from 1; 0 if never used.
    u32 use_order;
    std::vector<u32> code;
    std::vector<u8> header;
    std::vector<u8> reflection;
};

struct ShaderList {
    int semantic_base{-1};
    std::vector<u64> tiling_keys;
    std::vector<ListedShader> shaders;
};

using ShaderStates = tsl::robin_map<u64, std::vector<std::vector<u8>>>;

/// What a file holds, if its format is this build's; a list cut short keeps what came before.
std::optional<ShaderList> ReadShaderList(const std::filesystem::path& path);
std::optional<ShaderStates> ReadShaderStates(const std::filesystem::path& path);

/// The list's entries are added one at a time, so the scanner needn't copy its shaders.
class ShaderListWriter {
public:
    ShaderListWriter(u32 count, int semantic_base, std::span<const u64> tiling_keys);
    void Add(u64 hash, u8 type, u8 stage, u32 use_order, std::span<const u32> code,
             std::span<const u8> header, std::span<const u8> reflection);
    /// Returns the compressed size.
    std::optional<size_t> Write(const std::filesystem::path& path) const;

private:
    std::vector<u8> raw;
};

std::optional<size_t> WriteShaderStates(const std::filesystem::path& path,
                                        const ShaderStates& states);

} // namespace VideoCore::ShaderFiles
