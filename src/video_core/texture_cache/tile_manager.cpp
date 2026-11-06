// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/tile_manager.h"

#include "video_core/host_shaders/tiling_macro_128_comp.h"
#include "video_core/host_shaders/tiling_macro_16_comp.h"
#include "video_core/host_shaders/tiling_macro_32_comp.h"
#include "video_core/host_shaders/tiling_macro_64_comp.h"
#include "video_core/host_shaders/tiling_macro_8_comp.h"
#include "video_core/host_shaders/tiling_macro_96_comp.h"
#include "video_core/host_shaders/tiling_micro_128_comp.h"
#include "video_core/host_shaders/tiling_micro_16_comp.h"
#include "video_core/host_shaders/tiling_micro_32_comp.h"
#include "video_core/host_shaders/tiling_micro_64_comp.h"
#include "video_core/host_shaders/tiling_micro_8_comp.h"
#include "video_core/host_shaders/tiling_micro_96_comp.h"

#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>

namespace VideoCore {

struct TilingInfo {
    u32 bank_swizzle;
    u32 micro_tiled_mips;
    u32 num_mips;
    std::array<MipInfo, MAX_MIPS> mips;
};

TileManager::TileManager(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, StreamBuffer& stream_buffer_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, stream_buffer{stream_buffer_} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto desc_layout_result = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(desc_layout_result.result));
    desc_layout = std::move(desc_layout_result.value);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 0U,
        .pPushConstantRanges = nullptr,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    pl_layout = std::move(layout);
}

TileManager::~TileManager() = default;

vk::Pipeline TileManager::GetTilingPipeline(const ImageInfo& info, bool is_tiler) {
    const TilingKey key{
        .tile_mode = info.tile_mode,
        .num_bits = info.num_bits,
        .num_samples = info.num_samples,
        .is_tiler = is_tiler,
    };
    {
        std::scoped_lock lock{pipelines_mutex};
        if (const auto it = tiling_pipelines.find(key); it != tiling_pipelines.end()) {
            return *it->second;
        }
    }
    LOG_INFO(Render_Vulkan, "Creating tiling pipeline {}_{} {}",
             magic_enum::enum_name(key.tile_mode), key.num_bits,
             key.is_tiler ? "tiler" : "detiler");
    auto pipeline = CreateTilingPipeline(key);
    if (key_observer) {
        key_observer(PackKey(key));
    }
    std::scoped_lock lock{pipelines_mutex};
    // Prepare's thread may have built it meanwhile; either serves.
    return *tiling_pipelines.try_emplace(key, std::move(pipeline)).first->second;
}

u64 TileManager::PackKey(const TilingKey& key) {
    return u64(key.tile_mode) | u64(key.num_bits) << 8 | u64(key.num_samples) << 16 |
           u64(key.is_tiler) << 24;
}

std::optional<TileManager::TilingKey> TileManager::UnpackKey(u64 packed) {
    const TilingKey key{
        .tile_mode = static_cast<AmdGpu::TileMode>(packed & 0xff),
        .num_bits = u32(packed >> 8) & 0xff,
        .num_samples = u32(packed >> 16) & 0xff,
        .is_tiler = ((packed >> 24) & 1) != 0,
    };
    // Saved keys come from a file, so only ones a game could have used are built.
    const bool bits_ok = key.num_bits == 8 || key.num_bits == 16 || key.num_bits == 32 ||
                         key.num_bits == 64 || key.num_bits == 96 || key.num_bits == 128;
    const bool samples_ok = key.num_samples == 1 || key.num_samples == 2 || key.num_samples == 4 ||
                            key.num_samples == 8;
    if (!bits_ok || !samples_ok || !magic_enum::enum_contains(key.tile_mode) ||
        (packed >> 25) != 0) {
        return std::nullopt;
    }
    return key;
}

void TileManager::SetKeyObserver(std::function<void(u64)> observer) {
    key_observer = std::move(observer);
}

/// Tiling pipelines every game measured so far needs (Bloodborne and The Last Guardian use 18
/// between them, the same in every run), built ahead on a first launch too.
static std::vector<u64> CommonKeys() {
    using AmdGpu::TileMode;
    struct Common {
        TileMode mode;
        u32 bits;
        bool tiler;
    };
    static constexpr std::array<Common, 18> Keys{{
        {TileMode::Thin1DThin, 8, false},
        {TileMode::Thin1DThin, 16, false},
        {TileMode::Thin1DThin, 32, false},
        {TileMode::Thin1DThin, 64, false},
        {TileMode::Thin1DThin, 128, false},
        {TileMode::Display2DThin, 32, false},
        {TileMode::Thin2DThin, 8, false},
        {TileMode::Thin2DThin, 32, false},
        {TileMode::Thin2DThin, 32, true},
        {TileMode::Thin2DThin, 64, false},
        {TileMode::Thin2DThin, 64, true},
        {TileMode::Depth2DThin64, 16, false},
        {TileMode::Depth2DThin64, 32, false},
        {TileMode::Depth2DThin64, 32, true},
        {TileMode::Thick1DThick, 8, false},
        {TileMode::Thick1DThick, 16, false},
        {TileMode::Thick1DThick, 32, false},
        {TileMode::Thick1DThick, 64, false},
    }};
    std::vector<u64> keys;
    for (const auto& key : Keys) {
        keys.push_back(u64(key.mode) | u64(key.bits) << 8 | u64(1) << 16 | u64(key.tiler) << 24);
    }
    return keys;
}

void TileManager::Prepare(std::function<std::vector<u64>()> keys) {
    prepare_thread = std::jthread([this, keys = std::move(keys)](std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:PrepareTiling");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        u32 built = 0;
        // The game's own first, then the common ones.
        auto all = keys();
        const auto common = CommonKeys();
        all.insert(all.end(), common.begin(), common.end());
        for (const u64 packed : all) {
            if (stop.stop_requested()) {
                return;
            }
            const auto key = UnpackKey(packed);
            if (!key) {
                continue;
            }
            {
                std::scoped_lock lock{pipelines_mutex};
                if (tiling_pipelines.contains(*key)) {
                    continue;
                }
            }
            auto pipeline = CreateTilingPipeline(*key);
            std::scoped_lock lock{pipelines_mutex};
            built += tiling_pipelines.try_emplace(*key, std::move(pipeline)).second ? 1 : 0;
        }
        LOG_INFO(Render_Vulkan, "Prepared {} tiling pipelines ahead", built);
    });
}

void TileManager::StopPrepare() {
    if (prepare_thread.joinable()) {
        prepare_thread.request_stop();
        prepare_thread.join();
    }
}

vk::UniquePipeline TileManager::CreateTilingPipeline(const TilingKey& key) const {
    const auto device = instance.GetDevice();
    const auto array_mode = AmdGpu::GetArrayMode(key.tile_mode);
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(key.tile_mode);
    const bool is_macro = AmdGpu::IsMacroTiled(array_mode);
    std::array<u32, 12> spec_data{
        key.num_samples,
        u32(micro_tile_mode),
        AmdGpu::GetMicroTileThickness(array_mode),
        u32(key.is_tiler),
    };
    if (is_macro) {
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(key.tile_mode, key.num_bits, key.num_samples);
        spec_data[4] = u32(array_mode);
        spec_data[5] = u32(AmdGpu::GetPipeConfig(key.tile_mode));
        spec_data[6] = AmdGpu::GetBankWidth(macro_tile_mode);
        spec_data[7] = AmdGpu::GetBankHeight(macro_tile_mode);
        spec_data[8] = AmdGpu::GetNumBanks(macro_tile_mode);
        spec_data[9] = std::bit_width(spec_data[8]) - 1;
        spec_data[10] =
            AmdGpu::CalculateTileSplit(key.tile_mode, array_mode, micro_tile_mode, key.num_bits);
        spec_data[11] = AmdGpu::GetMacrotileAspect(macro_tile_mode);
    }

    std::span<const u32> code;
    switch (key.num_bits) {
    case 8:
        code = is_macro ? std::span<const u32>{TILING_MACRO_8_COMP}
                        : std::span<const u32>{TILING_MICRO_8_COMP};
        break;
    case 16:
        code = is_macro ? std::span<const u32>{TILING_MACRO_16_COMP}
                        : std::span<const u32>{TILING_MICRO_16_COMP};
        break;
    case 32:
        code = is_macro ? std::span<const u32>{TILING_MACRO_32_COMP}
                        : std::span<const u32>{TILING_MICRO_32_COMP};
        break;
    case 64:
        code = is_macro ? std::span<const u32>{TILING_MACRO_64_COMP}
                        : std::span<const u32>{TILING_MICRO_64_COMP};
        break;
    case 96:
        code = is_macro ? std::span<const u32>{TILING_MACRO_96_COMP}
                        : std::span<const u32>{TILING_MICRO_96_COMP};
        break;
    case 128:
        code = is_macro ? std::span<const u32>{TILING_MACRO_128_COMP}
                        : std::span<const u32>{TILING_MICRO_128_COMP};
        break;
    default:
        UNREACHABLE_MSG("Unsupported tiling pixel width {}", key.num_bits);
    }

    const auto module = Vulkan::CompileSPV(code, device);
    static constexpr auto spec_entries = [] {
        std::array<vk::SpecializationMapEntry, 12> entries{};
        for (u32 i = 0; i < entries.size(); ++i) {
            entries[i] = vk::SpecializationMapEntry{i, i * u32(sizeof(u32)), sizeof(u32)};
        }
        return entries;
    }();

    const u32 spec_count = is_macro ? 12 : 4;
    const vk::SpecializationInfo specialization{
        .mapEntryCount = spec_count,
        .pMapEntries = spec_entries.data(),
        .dataSize = spec_count * sizeof(u32),
        .pData = spec_data.data(),
    };

    const auto module_name = fmt::format("{}_{} {}", magic_enum::enum_name(key.tile_mode),
                                         key.num_bits, key.is_tiler ? "tiler" : "detiler");
    Vulkan::SetObjectName(device, module, module_name);

    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = &specialization,
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = *pl_layout,
    };

    auto [result, pipeline] =
        device.createComputePipelineUnique(VK_NULL_HANDLE, compute_pipeline_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Detiler pipeline creation failed {}",
               vk::to_string(result));
    device.destroyShaderModule(module);
    return std::move(pipeline);
}

std::pair<const Buffer*, u64> TileManager::DetileImage(const VideoCore::Buffer* in_buffer,
                                                       u64 in_offset, const ImageInfo& info) {
    if (!info.props.is_tiled) {
        return {in_buffer, in_offset};
    }

    const Vulkan::Scheduler::ScopedGpuSection gpu_section{scheduler,
                                                          Vulkan::Scheduler::GpuSection::Detiling};
    scheduler.AddGpuSectionBytes(info.guest_size);

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.micro_tiled_mips = info.micro_tiled_mips;
    params.num_mips = info.resources.levels;
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max<u16>((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max<u16>((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto staging = runtime.GetStagingPool().Request(info.guest_size, MemoryType::DeviceLocal,
                                                          256, false, true);

    scheduler.EndRendering();
    runtime.FlushBarriers();

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, false));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = in_buffer->Handle(),
        .offset = in_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = staging.buffer->Handle(),
        .offset = staging.offset,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const auto dim_x = (info.guest_size / (info.num_bits / 8)) / 64;
    cmdbuf.dispatch(dim_x, 1, 1);

    runtime.AccessBuffer(staging.buffer, staging.offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);

    return {staging.buffer, staging.offset};
}

void TileManager::TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                            const VideoCore::Buffer* out_buffer, u64 out_offset) {
    const auto& info = in_image.info;
    if (!info.props.is_tiled) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        runtime.DownloadImage(&in_image, out_buffer, buffer_copies);
        return;
    }

    const Vulkan::Scheduler::ScopedGpuSection gpu_section{scheduler,
                                                          Vulkan::Scheduler::GpuSection::Detiling};

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.micro_tiled_mips = info.micro_tiled_mips;
    params.num_mips = static_cast<u32>(buffer_copies.size());
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max<u16>((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max<u16>((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto staging = runtime.GetStagingPool().Request(info.guest_size, MemoryType::DeviceLocal,
                                                          256, false, true);
    for (auto& copy : buffer_copies) {
        copy.bufferOffset += staging.offset;
    }

    runtime.DownloadImage(&in_image, staging.buffer, buffer_copies);
    runtime.FlushBarriers();

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, true));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = out_buffer->Handle(),
        .offset = out_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = staging.buffer->Handle(),
        .offset = staging.offset,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const auto dim_x = (info.guest_size / (info.num_bits / 8)) / 64;
    cmdbuf.dispatch(dim_x, 1, 1);

    runtime.AccessBuffer(staging.buffer, staging.offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead);
    runtime.AccessBuffer(out_buffer, out_offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);
}

} // namespace VideoCore
