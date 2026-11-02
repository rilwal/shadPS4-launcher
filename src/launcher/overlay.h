// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

// The game launcher's quick menu drawn into the game's own frames (launcher_overlay_vk.h): the
// shell draws the menu into buffers it shares with shadPS4, and the presenter draws the newest one
// over each frame it presents. When the menu opens, the frame the game paused on is copied for the
// shell, as the menu's background. Linux and macOS for now; elsewhere, and outside launcher mode,
// every call here does nothing and the shell's own window comes to the front for the menu.
//
// All but DeviceExtensions and WantsPresents are for the present thread. Serials are the present
// scheduler's ticks.
namespace Launcher::Overlay {

// The device extensions the overlay needs, to enable when they're there
[[nodiscard]] std::span<const char* const> DeviceExtensions();

struct DeviceInfo {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    u32 queue_family;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
};

// Once the device is made; logs why if the overlay can't be drawn on it
void Create(const DeviceInfo& info);
// Once the device is idle
void Destroy();

// Starts a present into an image `window` big. `wait` waits for a serial to finish on the GPU and
// returns the newest finished one. True if this present's frame is to be copied with Capture.
bool Begin(u64 completed, VkExtent2D window, const std::function<u64(u64)>& wait);

// Copies `image` (TRANSFER_SRC_OPTIMAL) for the shell
void Capture(VkCommandBuffer command, VkImage image, VkFormat format, VkExtent2D extent);

// Draws the menu over `target` (COLOR_ATTACHMENT_OPTIMAL, written by an earlier pass), outside any
// render pass
void Draw(VkCommandBuffer command, VkImageView target, VkFormat format, VkExtent2D extent);

// Ends a present submitted with `serial`
void End(u64 serial);

// Whether the shell needs presents though the game sends no frames (loading, or paused): to show
// the menu, or finish a capture. Any thread.
[[nodiscard]] bool WantsPresents();

} // namespace Launcher::Overlay
