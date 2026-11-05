// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <atomic>
#include <boost/container/small_vector.hpp>

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Vulkan {

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

class Instance;
class Scheduler;
class DescriptorHeap;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        const VkPipeline upgraded = upgraded_handle.load(std::memory_order_acquire);
        return upgraded != VK_NULL_HANDLE ? vk::Pipeline{upgraded} : *pipeline;
    }

    /// False while a worker still creates it (draws that use it are skipped until then), or if
    /// that was dropped at shutdown.
    [[nodiscard]] bool IsReady() const noexcept {
        return state.load(std::memory_order_acquire) == State::Ready;
    }

    /// Time the driver took to create it, once it's ready.
    [[nodiscard]] u64 DriverNs() const noexcept {
        return driver_ns;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::SwStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::SwStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    void BindResources(DescriptorWrites& set_writes, const Shader::PushData& push_data) const;

protected:
    enum class State : u8 {
        Ready,
        Pending,
        Dropped,
    };

    [[nodiscard]] std::string GetDebugString() const;

    /// Publishes the result of a background creation.
    void Finish(State result) noexcept {
        state.store(result, std::memory_order_release);
        state.notify_all();
    }

    /// Waits for background work on the pipeline, so it never outlives it.
    void WaitIdle() const noexcept {
        state.wait(State::Pending, std::memory_order_acquire);
        upgrading.wait(true, std::memory_order_acquire);
    }

    /// Publishes an optimized pipeline that replaces the first one. Both stay alive, as
    /// recorded commands may still use the first.
    void Upgrade(vk::UniquePipeline optimized) noexcept {
        upgraded = std::move(optimized);
        upgraded_handle.store(*upgraded, std::memory_order_release);
    }

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    bool uses_push_descriptors{};
    bool is_compute;
    std::atomic<State> state{State::Ready};
    /// An optimized version is being created in the background.
    std::atomic<bool> upgrading{false};
    vk::UniquePipeline upgraded;
    std::atomic<VkPipeline> upgraded_handle{VK_NULL_HANDLE};
    u64 driver_ns{};
};

} // namespace Vulkan
