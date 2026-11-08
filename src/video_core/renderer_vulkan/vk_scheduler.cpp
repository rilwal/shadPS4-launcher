// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <functional>
#include <iterator>
#include <fmt/format.h>
#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    if (EmulatorSettings.IsGpuTiming()) {
        auto [result, pool] = instance.GetDevice().createQueryPoolUnique({
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = NumTimedSubmits * QueriesPerSubmit,
        });
        if (result == vk::Result::eSuccess) {
            timestamp_pool = std::move(pool);
        } else {
            LOG_WARNING(Render_Vulkan, "Can't create a timestamp query pool: {}",
                        vk::to_string(result));
        }
    }
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    CommandBuffer().beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().endRendering();
}

vk::CommandBuffer Scheduler::UploadCommandBuffer() {
    auto& upload_cmdbuf = sessions.back().upload;
    if (upload_cmdbuf) {
        return upload_cmdbuf;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    upload_cmdbuf = command_pool.Commit();
    Check(upload_cmdbuf.begin(begin_info));
    return upload_cmdbuf;
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish(std::source_location where) {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick, where);
}

void Scheduler::Wait(u64 tick, std::source_location where) {
    const VideoCore::PerfStats::ScopedCost cost{VideoCore::PerfStats::Cost::GpuWaits};
    const bool timed = VideoCore::PerfStats::is_gpu_thread;
    const u64 start = timed ? VideoCore::PerfStats::NowNs() : 0;
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    work_semaphore.Wait(tick);
    if (!timed) {
        return;
    }
    const u64 ns = VideoCore::PerfStats::NowNs() - start;
    const auto it = std::ranges::find_if(wait_sites, [&](const WaitSite& site) {
        return site.line == where.line() && site.file == where.file_name();
    });
    if (it != wait_sites.end()) {
        ++it->count;
        it->ns += ns;
    } else {
        wait_sites.push_back({where.file_name(), where.line(), 1, ns});
    }
}

std::vector<std::string> Scheduler::TakeWaitSites() {
    std::ranges::sort(wait_sites, std::greater{}, &WaitSite::ns);
    std::vector<std::string> out;
    for (const auto& site : wait_sites) {
        std::string_view file{site.file};
        if (const auto slash = file.find("src/"); slash != std::string_view::npos) {
            file.remove_prefix(slash + 4);
        }
        out.push_back(
            fmt::format("{}:{} {} waits {:.1f} ms", file, site.line, site.count, site.ns / 1e6));
    }
    wait_sites.clear();
    return out;
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::BeginSession() {
    EndSession();

    auto& session = sessions.emplace_back();

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    session.primary = command_pool.Commit();
    Check(session.primary.begin(begin_info));

    // The first session of a submission starts its timing, if a slot is free.
    gpu_section = GpuSection::Other;
    if (timestamp_pool && sessions.size() == 1 && timed_submits.size() < NumTimedSubmits) {
        const u32 slot = next_timed;
        next_timed = (next_timed + 1) % NumTimedSubmits;
        session.primary.resetQueryPool(*timestamp_pool, slot * QueriesPerSubmit, QueriesPerSubmit);
        session.primary.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestamp_pool,
                                       slot * QueriesPerSubmit);
        current_timed = slot;
        timed_sections.clear();
        timed_bytes = {};
    }

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::EndSession() {
    if (sessions.empty()) {
        return;
    }

    if (on_session) {
        on_session();
    }

    const auto& session = sessions.back();
    if (session.upload) {
        Check(session.upload.end());
    }

    EndRendering();
    if (ending_submit && current_timed) {
        timed_sections.push_back(gpu_section);
        session.primary.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestamp_pool,
                                       *current_timed * QueriesPerSubmit +
                                           static_cast<u32>(timed_sections.size()));
    }
    Check(session.primary.end());
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    const VideoCore::PerfStats::ScopedCost cost{VideoCore::PerfStats::Cost::Submits};
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    ending_submit = true;
    EndSession();
    ending_submit = false;

    std::vector<vk::CommandBuffer> cmd_buffers;
    cmd_buffers.reserve(sessions.size() * 2);

    for (const auto& session : sessions) {
        if (session.upload) {
            cmd_buffers.push_back(session.upload);
        }
        cmd_buffers.push_back(session.primary);
    }
    sessions.clear();

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    std::array<vk::PipelineStageFlags, std::tuple_size_v<decltype(info.wait_semas)>>
        wait_stage_masks;
    wait_stage_masks.fill(vk::PipelineStageFlagBits::eAllCommands);

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = static_cast<u32>(cmd_buffers.size()),
        .pCommandBuffers = cmd_buffers.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceFault();
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    work_semaphore.Refresh();
    if (current_timed) {
        timed_submits.push({*current_timed, signal_value, std::move(timed_sections), timed_bytes});
        timed_sections = {};
        current_timed.reset();
    }
    ReadGpuTimes();
    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::MarkGpuSection(GpuSection section) {
    // One query is kept for the submission's end; past that, the rest goes to the current kind.
    if (current_timed && timed_sections.size() + 2 < QueriesPerSubmit) {
        timed_sections.push_back(gpu_section);
        sessions.back().primary.writeTimestamp(
            vk::PipelineStageFlagBits::eBottomOfPipe, *timestamp_pool,
            *current_timed * QueriesPerSubmit + static_cast<u32>(timed_sections.size()));
        gpu_section = section;
    } else if (!current_timed) {
        gpu_section = section;
    }
}

void Scheduler::ReadGpuTimes() {
    const double period = instance.TimestampPeriod();
    std::vector<u64> stamps;
    while (!timed_submits.empty() && work_semaphore.IsFree(timed_submits.front().tick)) {
        const TimedSubmit timed = std::move(timed_submits.front());
        timed_submits.pop();
        const u32 count = static_cast<u32>(timed.sections.size()) + 1;
        stamps.assign(count, 0);
        const auto result = instance.GetDevice().getQueryPoolResults(
            *timestamp_pool, timed.slot * QueriesPerSubmit, count, count * sizeof(u64),
            stamps.data(), sizeof(u64), vk::QueryResultFlagBits::e64);
        if (result != vk::Result::eSuccess || stamps.back() < stamps.front()) {
            continue;
        }
        const auto to_ns = [period](u64 ticks) { return static_cast<u64>(double(ticks) * period); };
        const u64 ns = to_ns(stamps.back() - stamps.front());
        std::array<u64, size_t(GpuSection::Count)> section_ns{};
        std::array<u32, size_t(GpuSection::Count)> section_count{};
        // A later timestamp can land before an earlier one when work overlaps, so each stretch
        // runs from the latest stamp so far.
        u64 last = stamps.front();
        for (u32 i = 0; i + 1 < count; ++i) {
            const auto kind = static_cast<size_t>(timed.sections[i]);
            const u64 end = std::max(stamps[i + 1], last);
            section_ns[kind] += to_ns(end - last);
            last = end;
            ++section_count[kind];
        }
        const bool overflowed = count >= QueriesPerSubmit;
        if (ns >= SlowSubmitNs) {
            std::string parts;
            for (size_t k = 0; k < section_ns.size(); ++k) {
                if (section_count[k] == 0) {
                    continue;
                }
                fmt::format_to(std::back_inserter(parts), ", {} {:.2f} ms ({}", GpuSectionNames[k],
                               section_ns[k] / 1e6, section_count[k]);
                if (const u64 bytes = timed.bytes[k]; bytes != 0) {
                    fmt::format_to(std::back_inserter(parts), ", {:.1f} MB, {:.1f} GB/s",
                                   bytes / 1e6,
                                   section_ns[k] ? double(bytes) / section_ns[k] : 0.0);
                }
                parts += ')';
            }
            LOG_INFO(Render_Vulkan, "Slow submit {:.2f} ms on the GPU{}{}", ns / 1e6, parts,
                     overflowed ? " (ran out of timestamps)" : "");
        }
        std::scoped_lock lock{gpu_time_mutex};
        gpu_time.busy_ns += ns;
        gpu_time.worst_ns = std::max(gpu_time.worst_ns, ns);
        ++gpu_time.submits;
        gpu_time.overflowed += overflowed ? 1 : 0;
        for (size_t k = 0; k < section_ns.size(); ++k) {
            gpu_time.section_ns[k] += section_ns[k];
        }
    }
}

Scheduler::GpuTime Scheduler::TakeGpuTime() {
    std::scoped_lock lock{gpu_time_mutex};
    return std::exchange(gpu_time, GpuTime{});
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
}

} // namespace Vulkan
