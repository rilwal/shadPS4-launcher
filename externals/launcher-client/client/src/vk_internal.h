// What the Vulkan helper's parts share (overlay_vk.cpp, list_vk.cpp): the functions they load from
// the emulator's loader, and a few helpers.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "launcher_overlay_vk.h"

#include <cstdint>

namespace lc::vk {

// The functions the helper uses, from the emulator's loader
struct Fns {
#define LC_VK_INSTANCE_FNS(X)                    \
	X(GetPhysicalDeviceProperties)               \
	X(GetPhysicalDeviceMemoryProperties)         \
	X(GetPhysicalDeviceFormatProperties)         \
	X(GetPhysicalDeviceQueueFamilyProperties)    \
	X(GetPhysicalDeviceImageFormatProperties2)
#define LC_VK_DEVICE_FNS(X)                     \
	X(CreateImage)                              \
	X(DestroyImage)                             \
	X(GetImageMemoryRequirements)               \
	X(GetImageSubresourceLayout)                \
	X(CreateBuffer)                             \
	X(DestroyBuffer)                            \
	X(GetBufferMemoryRequirements)              \
	X(BindBufferMemory)                         \
	X(AllocateMemory)                           \
	X(FreeMemory)                               \
	X(MapMemory)                                \
	X(BindImageMemory)                          \
	X(CreateImageView)                          \
	X(DestroyImageView)                         \
	X(CreateSampler)                            \
	X(DestroySampler)                           \
	X(CreateDescriptorSetLayout)                \
	X(DestroyDescriptorSetLayout)               \
	X(CreateDescriptorPool)                     \
	X(DestroyDescriptorPool)                    \
	X(AllocateDescriptorSets)                   \
	X(UpdateDescriptorSets)                     \
	X(CreatePipelineLayout)                     \
	X(DestroyPipelineLayout)                    \
	X(CreateShaderModule)                       \
	X(DestroyShaderModule)                      \
	X(CreateGraphicsPipelines)                  \
	X(DestroyPipeline)                          \
	X(CreateRenderPass)                         \
	X(DestroyRenderPass)                        \
	X(CreateFramebuffer)                        \
	X(DestroyFramebuffer)                       \
	X(CmdPipelineBarrier)                       \
	X(CmdBlitImage)                             \
	X(CmdCopyImage)                             \
	X(CmdCopyBufferToImage)                     \
	X(CmdCopyImageToBuffer)                     \
	X(CmdUpdateBuffer)                          \
	X(CmdClearColorImage)                       \
	X(CmdBeginRenderPass)                       \
	X(CmdEndRenderPass)                         \
	X(CmdBindPipeline)                          \
	X(CmdBindDescriptorSets)                    \
	X(CmdBindVertexBuffers)                     \
	X(CmdPushConstants)                         \
	X(CmdSetViewport)                           \
	X(CmdSetScissor)                            \
	X(CmdDraw)                                  \
	X(CreateQueryPool)                          \
	X(DestroyQueryPool)                         \
	X(GetQueryPoolResults)                      \
	X(CmdResetQueryPool)                        \
	X(CmdWriteTimestamp)
#define LC_VK_LINUX_FNS(X) \
	X(GetMemoryFdKHR)      \
	X(GetMemoryFdPropertiesKHR)
#define LC_VK_HOST_INSTANCE_FNS(X)     \
	X(GetPhysicalDeviceProperties2)    \
	X(GetPhysicalDeviceExternalBufferProperties)
#define LC_VK_HOST_FNS(X) X(GetMemoryHostPointerPropertiesEXT)

#define LC_VK_DECLARE(name) PFN_vk##name name = nullptr;
	LC_VK_INSTANCE_FNS(LC_VK_DECLARE)
	LC_VK_DEVICE_FNS(LC_VK_DECLARE)
	LC_VK_LINUX_FNS(LC_VK_DECLARE)
	LC_VK_HOST_INSTANCE_FNS(LC_VK_DECLARE)
	LC_VK_HOST_FNS(LC_VK_DECLARE)
#undef LC_VK_DECLARE

	// The name of the first one missing, or nullptr. `shares`: the ones sharing the shell's
	// buffers needs too (not for draw lists)
	const char* load(const lc_vk_info& info, bool shares);
};

void barrier(const Fns& vk, VkCommandBuffer command, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
             VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
             uint32_t src_queue = VK_QUEUE_FAMILY_IGNORED, uint32_t dst_queue = VK_QUEUE_FAMILY_IGNORED, uint32_t levels = 1);

// Makes writes to memory visible, one way or the other (also: other processes')
void memory_barrier(const Fns& vk, VkCommandBuffer command, VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                    VkPipelineStageFlags dst_stage);

inline VkDeviceSize round_up(VkDeviceSize value, VkDeviceSize alignment) {
	return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

// Whether an image of `format` holds linear values, which the shader reads and writes: an sRGB
// format, which the hardware encodes, or a float one (an HDR swapchain's, scRGB). The shell's
// buffers, its lists' images and the capture hold sRGB-encoded values. A 10-bit UNORM swapchain
// holds encoded ones, like an 8-bit UNORM one. Values over 1 (HDR highlights) clip in a capture.
bool holds_linear(VkFormat format);

// A memory type in `bits` with all of `want`; failing that, any in `bits`
uint32_t memory_type(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits, VkMemoryPropertyFlags want);

} // namespace lc::vk
