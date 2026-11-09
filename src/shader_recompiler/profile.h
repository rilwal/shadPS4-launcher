// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader {

struct Profile {
    u32 max_viewport_width{};
    u32 max_viewport_height{};
    u32 max_shared_memory_size{};
    u32 supported_spirv{0x00010000};
    u32 subgroup_size{};
    u32 sparse_page_shift{};
    bool support_int8{};
    bool support_int16{};
    bool support_int64{};
    bool support_float16{};
    bool support_float64{};
    bool supports_denorm_behavior_independence{};
    bool supports_rounding_mode_independence{};
    bool support_fp16_denorm_preserve{};
    bool support_fp16_denorm_flush{};
    bool support_fp16_round_to_zero{};
    bool support_fp32_denorm_preserve{};
    bool support_fp32_denorm_flush{};
    bool support_fp32_round_to_zero{};
    bool support_fp64_denorm_preserve{};
    bool support_fp64_denorm_flush{};
    bool support_fp64_round_to_zero{};
    bool support_fp16_signed_zero_inf_nan_preserve{};
    bool support_fp32_signed_zero_inf_nan_preserve{};
    bool support_fp64_signed_zero_inf_nan_preserve{};
    bool supports_image_load_store_lod{};
    bool supports_native_cube_calc{};
    bool supports_trinary_minmax{};
    bool supports_buffer_fp32_atomic_min_max{};
    bool supports_image_fp32_atomic_min_max{};
    bool supports_buffer_int64_atomics{};
    bool supports_shared_int64_atomics{};
    bool supports_workgroup_explicit_memory_layout{};
    bool supports_amd_shader_explicit_vertex_parameter{};
    bool supports_fragment_shader_barycentric{};
    bool supports_shader_subgroup_clock{};
    bool has_broken_spirv_clamp{};
    bool lower_left_origin_mode{};
    bool needs_manual_interpolation{};
    bool needs_lds_barriers{};
    bool needs_buffer_offsets{};
    bool needs_unorm_fixup{};
    bool needs_clip_distance_emulation{};
    bool supports_shader_stencil_export{};
    bool fast_shader_math{};
    bool support_bda{};
    /// Resources come from VK_EXT_descriptor_heap: each stage numbers its own bindings, and
    /// buffer offsets are read from the stage's table instead of push constants.
    bool uses_descriptor_heap{};
    /// Leaves out of specialisations what doesn't change the code: the exact format of
    /// render targets (only its number class), and sRGB where no sampler forces degamma.
    bool fewer_shader_variants{};
    /// Formatted buffer loads read the buffer's format from the stage table, so shaders
    /// aren't compiled again for each format. Needs the descriptor heap.
    bool runtime_buffer_formats{};
    /// Structured buffer accesses read the buffer's stride from the stage table, so shaders
    /// aren't compiled again for each stride. Needs the descriptor heap.
    bool runtime_buffer_strides{};
    /// Storage image loads and stores read the image's swizzle from the stage table. Needs
    /// the descriptor heap.
    bool runtime_storage_swizzles{};
    /// Color exports read each render target's swizzle from the stage table, and export to
    /// every target the shader's export format names whether or not the draw binds it. Needs
    /// the descriptor heap.
    bool runtime_target_swizzles{};
    /// Vertex attributes read their V#'s format from the stage table: the vertex input hands
    /// the shader each element's raw bits and the shader converts them, so one shader serves
    /// any format, number class and component order. Needs the descriptor heap.
    bool runtime_vertex_formats{};

    bool operator==(const Profile&) const = default;
};

} // namespace Shader
