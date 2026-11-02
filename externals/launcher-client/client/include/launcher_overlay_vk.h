/*
 * launcher_overlay_vk.h: an optional Vulkan helper for the composited overlay
 * (launcher_overlay.h). It imports the shell's surfaces as images, copies the paused frame into
 * a surface the shell can open, and draws the menu as one premultiplied quad.
 *
 * An emulator with a renderer of another API, or its own way of drawing, uses launcher_overlay.h
 * directly instead.
 *
 * The helper loads every Vulkan function through the pointers in lc_vk_info, so it works with any
 * loader (volk, a static one) and links none. All calls are for the present thread.
 *
 * Per present, around the emulator's own submission with serial `serial`:
 *
 *   lc_vk_frame f;
 *   lc_vk_begin(vk, completed, window, new_frame, &f);
 *   if (f.wait) { wait_for_serial(f.wait); lc_vk_begin(vk, f.wait, window, new_frame, &f); }
 *   ...record the frame; with f.capture, the finished frame in TRANSFER_SRC_OPTIMAL:
 *   lc_vk_capture(vk, cmd, image, format, extent);
 *   VkImageView menu = lc_vk_acquire(vk, cmd);      outside any render pass
 *   ...begin the pass that draws into the window's image, then, if menu:
 *   lc_vk_draw(vk, cmd, render_pass, subpass, format);
 *   ...end the pass, submit with `serial`
 *   lc_vk_end(vk, serial);
 *
 * Linux shares dma-bufs as images. macOS shares IOSurfaces, whose memory is imported as host memory
 * (VK_EXT_external_memory_host, which MoltenVK and KosmicKrisp both have) and copied into an image
 * of the helper's own while the menu is drawn. Elsewhere lc_vk_create returns NULL.
 *
 * With LC_CAP_DRAW_LISTS the helper draws the shell's draw lists itself (launcher_list.h), with
 * the shell's own shaders, into a canvas it then draws as that quad: the same calls, with nothing
 * shared with the shell, so no extension is needed and any platform will do. f.capture then asks
 * for the frame under the menu, which the menu's glass shows blurred; the helper keeps that copy
 * itself. The menu's art can be BC7: enable textureCompressionBC on the device, or such a list
 * isn't drawn and the shell comes to the front instead.
 */
#ifndef LAUNCHER_OVERLAY_VK_H
#define LAUNCHER_OVERLAY_VK_H

#include "launcher_overlay.h"

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lc_vk lc_vk;

typedef struct lc_vk_info {
	VkInstance       instance;
	VkPhysicalDevice physical_device;
	VkDevice         device;
	/* The queue family the emulator presents (and submits these commands) on. */
	uint32_t queue_family;
	PFN_vkGetInstanceProcAddr get_instance_proc_addr;
	/* NULL to get it through get_instance_proc_addr. */
	PFN_vkGetDeviceProcAddr get_device_proc_addr;
} lc_vk_info;

/* The device extensions the helper needs to share the shell's buffers, to enable when creating
   the device (draw lists need none). They assume a Vulkan 1.2 device; on an older one, also enable
   what VK_EXT_image_drm_format_modifier requires (VK_KHR_image_format_list and the rest). On macOS:
   VK_KHR_external_memory and VK_EXT_external_memory_host. */
const char *const *lc_vk_device_extensions(uint32_t *count);

/* NULL if this device can't share images with the launcher, with the reason in `error`. Then the
   emulator shouldn't call lc_overlay_view, and the shell comes to the front instead. */
lc_vk *lc_vk_create(lc_overlay *overlay, const lc_vk_info *info, char *error, size_t error_len);

/* Once the device is idle. */
void lc_vk_destroy(lc_vk *vk);

typedef struct lc_vk_frame {
	/* Copy this present's frame with lc_vk_capture: for the shell, or, drawing lists, for the
	   menu's glass. */
	bool capture;
	/* Nonzero: wait for the submission with this serial to finish, then call lc_vk_begin again
	   for this present (lc_frame.wait). */
	uint64_t wait;
} lc_vk_frame;

/* Starts a present: `completed` is the newest serial the GPU has finished, `window` the size of
   the image the overlay will be drawn into (sent to the launcher as the view). `new_frame` says
   the game's frame isn't the one it presented last: false while it's paused and presents the
   same frame again, so the menu's glass copies and blurs it once instead of every present.
   Imports new surfaces, and frees old ones the GPU is done with. Called again, after waiting for
   `wait`, it starts the same present over. */
void lc_vk_begin(lc_vk *vk, uint64_t completed, VkExtent2D window, bool new_frame, lc_vk_frame *frame);

/* Copies `source` (TRANSFER_SRC_OPTIMAL, of `format`, `extent` big) for the shell, converting it
   to BGRA8, or, drawing lists, for the menu's glass. The source stays in its layout. Does nothing
   unless the frame asked for a capture. VK_NULL_HANDLE says the frame can't be copied: the shell
   comes to the front for the menu instead. */
void lc_vk_capture(lc_vk *vk, VkCommandBuffer command, VkImage source, VkFormat format, VkExtent2D extent);

/* The buffer to draw over this present (made readable by the fragment shader), or VK_NULL_HANDLE
   for none. Outside any render pass. Drawing lists, this is where the list is drawn, into the
   helper's canvas, which it returns; that records render passes into `command` and leaves its
   bound graphics pipeline, descriptor sets, vertex buffer 0, viewport and scissor changed, so
   bind your own state after calling it. */
VkImageView lc_vk_acquire(lc_vk *vk, VkCommandBuffer command);

/* Draws the acquired buffer over the whole target as one premultiplied quad. Inside a render
   pass (`render_pass`, `subpass`) or, with VK_NULL_HANDLE, dynamic rendering (which the device
   must have), drawing into one colour attachment of `format`. Sets its own viewport and scissor
   to the window's size. Does nothing if lc_vk_acquire returned VK_NULL_HANDLE. */
void lc_vk_draw(lc_vk *vk, VkCommandBuffer command, VkRenderPass render_pass, uint32_t subpass, VkFormat format);

/* Ends a present: its commands were submitted with `serial`. */
void lc_vk_end(lc_vk *vk, uint64_t serial);

#ifdef __cplusplus
}
#endif

#endif /* LAUNCHER_OVERLAY_VK_H */
