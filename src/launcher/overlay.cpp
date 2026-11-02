// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <launcher_overlay.h>

#include "common/logging/log.h"
#include "launcher/overlay.h"

#ifdef LAUNCHER_CLIENT_VULKAN
#include <launcher_overlay_vk.h>
#endif

namespace Launcher::Overlay {

namespace {

lc_overlay* Get() {
    const auto session = lc_get();
    return session != nullptr ? lc_overlay_get(session) : nullptr;
}

#ifdef LAUNCHER_CLIENT_VULKAN
lc_vk* g_vk = nullptr;
#endif

} // namespace

std::span<const char* const> DeviceExtensions() {
#ifdef LAUNCHER_CLIENT_VULKAN
    if (Get() != nullptr) {
        u32 count = 0;
        const char* const* names = lc_vk_device_extensions(&count);
        return {names, count};
    }
#endif
    return {};
}

void Create(const DeviceInfo& info) {
#ifdef LAUNCHER_CLIENT_VULKAN
    lc_overlay* overlay = Get();
    if (overlay == nullptr || g_vk != nullptr) {
        return;
    }
    const lc_vk_info vk_info{
        .instance = info.instance,
        .physical_device = info.physical_device,
        .device = info.device,
        .queue_family = info.queue_family,
        .get_instance_proc_addr = info.get_instance_proc_addr,
        .get_device_proc_addr = nullptr,
    };
    std::array<char, 512> error{};
    g_vk = lc_vk_create(overlay, &vk_info, error.data(), error.size());
    if (g_vk == nullptr) {
        LOG_ERROR(Render_Vulkan,
                  "Launcher: the quick menu can't be drawn in the game's window ({}); the "
                  "launcher's own window shows it instead",
                  error.data());
    }
#else
    (void)info;
#endif
}

void Destroy() {
#ifdef LAUNCHER_CLIENT_VULKAN
    lc_vk_destroy(g_vk);
    g_vk = nullptr;
#endif
}

bool Begin(u64 completed, VkExtent2D window, const std::function<u64(u64)>& wait) {
#ifdef LAUNCHER_CLIENT_VULKAN
    if (g_vk == nullptr) {
        return false;
    }
    lc_vk_frame frame{};
    lc_vk_begin(g_vk, completed, window, &frame);
    // A newer frame of the menu whose buffer an earlier present still reads: once that's done
    // (it's usually a frame old), this present takes it
    if (frame.wait != 0) {
        lc_vk_begin(g_vk, wait(frame.wait), window, &frame);
    }
    return frame.capture;
#else
    (void)completed;
    (void)window;
    (void)wait;
    return false;
#endif
}

void Capture(VkCommandBuffer command, VkImage image, VkFormat format, VkExtent2D extent) {
#ifdef LAUNCHER_CLIENT_VULKAN
    lc_vk_capture(g_vk, command, image, format, extent);
#else
    (void)command;
    (void)image;
    (void)format;
    (void)extent;
#endif
}

void Draw(VkCommandBuffer command, VkImageView target, VkFormat format, VkExtent2D extent) {
#ifdef LAUNCHER_CLIENT_VULKAN
    if (g_vk == nullptr) {
        return;
    }
    const VkImageView menu = lc_vk_acquire(g_vk, command);
    if (menu == VK_NULL_HANDLE) {
        return;
    }
    const auto cmd = vk::CommandBuffer{command};

    // The pass before wrote the image; the menu blends over it
    const vk::MemoryBarrier2 written{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask =
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &written,
    });

    const vk::RenderingAttachmentInfo attachment{
        .imageView = target,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmd.beginRendering(vk::RenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = {extent.width, extent.height}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    lc_vk_draw(g_vk, command, VK_NULL_HANDLE, 0, format);
    cmd.endRendering();
#else
    (void)command;
    (void)target;
    (void)format;
    (void)extent;
#endif
}

void End(u64 serial) {
#ifdef LAUNCHER_CLIENT_VULKAN
    lc_vk_end(g_vk, serial);
#else
    (void)serial;
#endif
}

bool WantsPresents() {
    lc_overlay* overlay = Get();
    return overlay != nullptr && lc_overlay_wants_presents(overlay);
}

} // namespace Launcher::Overlay
