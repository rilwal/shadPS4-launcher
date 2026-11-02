// The Vulkan helper (launcher_overlay_vk.h).
//
// Linux: the shell's buffers are linear BGRA8 dma-bufs, imported with VK_EXT_external_memory_dma_buf
// and VK_EXT_image_drm_format_modifier, and the capture is a linear image of the game's own,
// exported the same way.
//
// macOS: the shell's buffers are IOSurfaces. Neither MoltenVK nor KosmicKrisp imports one as an
// image the same way (MoltenVK has VK_EXT_metal_objects, KosmicKrisp only Metal heaps), but both
// take host memory (VK_EXT_external_memory_host), and an IOSurface's memory is mapped into every
// process that uses it, the GPU's as well as the CPU's on Apple silicon. So each buffer's memory
// is imported as a VkBuffer and copied into an image of the helper's own when the menu is drawn
// (a few hundred microseconds at 4K); the capture is an IOSurface of the helper's own, copied into
// the same way.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "launcher_overlay_vk.h"

#include "internal.h"
#include "overlay_vk_shaders.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <unistd.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurfaceRef.h>
#define LC_VK_HOST_SURFACES 1
#endif

namespace {

constexpr VkFormat FORMAT                = VK_FORMAT_B8G8R8A8_UNORM;
[[maybe_unused]] constexpr uint64_t DRM_FORMAT_MOD_LINEAR = 0;

const char* const DEVICE_EXTENSIONS[] = {
#if defined(__linux__)
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
    VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
#elif defined(LC_VK_HOST_SURFACES)
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
#else
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#endif
};

// Whether an image of `format` holds linear values, which the shader reads and writes: an sRGB
// format, which the hardware encodes, or a float one (an HDR swapchain's, scRGB). The shell's
// buffers and the capture hold sRGB-encoded values. A 10-bit UNORM swapchain holds encoded ones,
// like an 8-bit UNORM one. Values over 1 (HDR highlights) clip in the capture.
bool holds_linear(VkFormat format) {
	switch (format) {
		case VK_FORMAT_R8G8B8A8_SRGB:
		case VK_FORMAT_B8G8R8A8_SRGB:
		case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
		case VK_FORMAT_R8G8B8_SRGB:
		case VK_FORMAT_B8G8R8_SRGB:
		case VK_FORMAT_R16G16B16A16_SFLOAT:
		case VK_FORMAT_R32G32B32A32_SFLOAT:
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return true;
		default: return false;
	}
}

// The functions the helper uses, from the emulator's loader
struct Fns {
#define LC_VK_INSTANCE_FNS(X)                    \
	X(GetPhysicalDeviceMemoryProperties)         \
	X(GetPhysicalDeviceFormatProperties)         \
	X(GetPhysicalDeviceImageFormatProperties2)
#define LC_VK_DEVICE_FNS(X)                     \
	X(CreateImage)                              \
	X(DestroyImage)                             \
	X(GetImageMemoryRequirements)               \
	X(GetImageSubresourceLayout)                \
	X(AllocateMemory)                           \
	X(FreeMemory)                               \
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
	X(CmdPipelineBarrier)                       \
	X(CmdBlitImage)                             \
	X(CmdCopyImage)                             \
	X(CmdBindPipeline)                          \
	X(CmdBindDescriptorSets)                    \
	X(CmdPushConstants)                         \
	X(CmdSetViewport)                           \
	X(CmdSetScissor)                            \
	X(CmdDraw)
#define LC_VK_LINUX_FNS(X) \
	X(GetMemoryFdKHR)      \
	X(GetMemoryFdPropertiesKHR)
#define LC_VK_HOST_INSTANCE_FNS(X)     \
	X(GetPhysicalDeviceProperties2)    \
	X(GetPhysicalDeviceExternalBufferProperties)
#define LC_VK_HOST_FNS(X)                  \
	X(CreateBuffer)                        \
	X(DestroyBuffer)                       \
	X(GetBufferMemoryRequirements)         \
	X(BindBufferMemory)                    \
	X(GetMemoryHostPointerPropertiesEXT)   \
	X(CmdCopyBufferToImage)                \
	X(CmdCopyImageToBuffer)

#define LC_VK_DECLARE(name) PFN_vk##name name = nullptr;
	LC_VK_INSTANCE_FNS(LC_VK_DECLARE)
	LC_VK_DEVICE_FNS(LC_VK_DECLARE)
	LC_VK_LINUX_FNS(LC_VK_DECLARE)
	LC_VK_HOST_INSTANCE_FNS(LC_VK_DECLARE)
	LC_VK_HOST_FNS(LC_VK_DECLARE)
#undef LC_VK_DECLARE

	// The name of the first one missing, or nullptr
	const char* load(const lc_vk_info& info) {
		auto get_device = info.get_device_proc_addr;
		if (get_device == nullptr) {
			get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(info.get_instance_proc_addr(info.instance, "vkGetDeviceProcAddr"));
		}
		if (get_device == nullptr) {
			return "vkGetDeviceProcAddr";
		}
#define LC_VK_LOAD_INSTANCE(name)                                                                              \
	if ((name = reinterpret_cast<PFN_vk##name>(info.get_instance_proc_addr(info.instance, "vk" #name))) == nullptr) { \
		return "vk" #name;                                                                                     \
	}
#define LC_VK_LOAD_DEVICE(name)                                                                       \
	if ((name = reinterpret_cast<PFN_vk##name>(get_device(info.device, "vk" #name))) == nullptr) { \
		return "vk" #name;                                                                            \
	}
		LC_VK_INSTANCE_FNS(LC_VK_LOAD_INSTANCE)
		LC_VK_DEVICE_FNS(LC_VK_LOAD_DEVICE)
#if defined(__linux__)
		LC_VK_LINUX_FNS(LC_VK_LOAD_DEVICE)
#endif
#if defined(LC_VK_HOST_SURFACES)
		LC_VK_HOST_INSTANCE_FNS(LC_VK_LOAD_INSTANCE)
		LC_VK_HOST_FNS(LC_VK_LOAD_DEVICE)
#endif
#undef LC_VK_LOAD_INSTANCE
#undef LC_VK_LOAD_DEVICE
		return nullptr;
	}
};

void barrier(const Fns& vk, VkCommandBuffer command, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
             VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
             uint32_t src_queue = VK_QUEUE_FAMILY_IGNORED, uint32_t dst_queue = VK_QUEUE_FAMILY_IGNORED) {
	VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	b.srcAccessMask       = src_access;
	b.dstAccessMask       = dst_access;
	b.oldLayout           = from;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = src_queue;
	b.dstQueueFamilyIndex = dst_queue;
	b.image               = image;
	b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	vk.CmdPipelineBarrier(command, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Makes writes to memory that other processes share visible, one way or the other
void memory_barrier(const Fns& vk, VkCommandBuffer command, VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                    VkPipelineStageFlags dst_stage) {
	VkMemoryBarrier b {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	b.srcAccessMask = src_access;
	b.dstAccessMask = dst_access;
	vk.CmdPipelineBarrier(command, src_stage, dst_stage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

VkDeviceSize round_up(VkDeviceSize value, VkDeviceSize alignment) {
	return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

// A BGRA8 image: one of the shell's buffers (imported), the capture (exported), or the capture's
// staging image. On macOS a shell's buffer is an image of the helper's own plus the IOSurface's
// memory as a buffer (`host`) to copy from, and the capture is only the IOSurface's memory.
struct Image {
	VkImage         image  = VK_NULL_HANDLE;
	VkDeviceMemory  memory = VK_NULL_HANDLE;
	VkImageView     view   = VK_NULL_HANDLE;
	VkDescriptorSet set    = VK_NULL_HANDLE; // A buffer's, for drawing it
	VkImageLayout   layout = VK_IMAGE_LAYOUT_UNDEFINED;
	uint32_t        width  = 0;
	uint32_t        height = 0;
	uint32_t        stride = 0; // The exported capture's one plane, or the IOSurface's rows
	uint64_t        offset = 0;
	VkBuffer        host        = VK_NULL_HANDLE; // The IOSurface's memory
	VkDeviceMemory  host_memory = VK_NULL_HANDLE;
	void*           surface     = nullptr; // The IOSurfaceRef, retained while its memory is in use
	bool            loaded      = false;   // A shell's buffer copied into `image` since it was written

	bool exists() const { return image != VK_NULL_HANDLE || host != VK_NULL_HANDLE; }

	void destroy(const Fns& vk, VkDevice device) {
		if (host != VK_NULL_HANDLE) {
			vk.DestroyBuffer(device, host, nullptr);
		}
		if (host_memory != VK_NULL_HANDLE) {
			vk.FreeMemory(device, host_memory, nullptr);
		}
#if defined(LC_VK_HOST_SURFACES)
		if (surface != nullptr) {
			CFRelease(static_cast<CFTypeRef>(surface));
		}
#endif
		if (view != VK_NULL_HANDLE) {
			vk.DestroyImageView(device, view, nullptr);
		}
		if (image != VK_NULL_HANDLE) {
			vk.DestroyImage(device, image, nullptr);
		}
		if (memory != VK_NULL_HANDLE) {
			vk.FreeMemory(device, memory, nullptr);
		}
		*this = {};
	}
};

// One generation of the shell's buffers, until the GPU is done with them
struct BufferSet {
	std::vector<Image> buffers;
	VkDescriptorPool   pool      = VK_NULL_HANDLE;
	uint64_t           last_used = 0; // The serial that last read any of them, 0 if none
};

// A capture's images, until the GPU is done with the copy into them
struct RetiredCapture {
	Image    capture;
	Image    staging;
	uint64_t last_used = 0;
};

struct Pipeline {
	VkRenderPass render_pass;
	uint32_t     subpass;
	VkFormat     format;
	VkPipeline   pipeline;
};

} // namespace

struct lc_vk {
	lc_overlay* overlay = nullptr;
	lc_vk_info  info {};
	Fns         vk;

	VkPhysicalDeviceMemoryProperties memory_props {};
	VkDeviceSize                     host_alignment = 0; // minImportedHostPointerAlignment, on macOS
	VkSampler                        sampler    = VK_NULL_HANDLE;
	VkDescriptorSetLayout            set_layout = VK_NULL_HANDLE;
	VkPipelineLayout                 layout     = VK_NULL_HANDLE;
	VkShaderModule                   vert       = VK_NULL_HANDLE;
	VkShaderModule                   frag       = VK_NULL_HANDLE;
	std::vector<Pipeline>            pipelines;

	uint32_t               generation = 0;
	BufferSet              current;
	std::vector<BufferSet> retired;

	VkExtent2D window {};
	int32_t    draw     = -1; // This present's buffer, from lc_overlay_begin
	bool       fresh    = false;
	bool       acquired = false; // lc_vk_acquire made it readable this present

	bool  capture_wanted   = false;
	bool  captured         = false; // lc_vk_capture recorded a copy this present
	Image capture;                  // Linear, exportable
	Image staging;                  // Optimal, for the blit that converts the format
	VkFormat staging_format = VK_FORMAT_UNDEFINED;
	uint64_t capture_used   = 0; // The serial that last copied into them, 0 if none
	std::vector<RetiredCapture> retired_captures;

	void log(lc_log_level level, const char* format, ...) LC_PRINTF(3, 4) {
		auto*   session = overlay != nullptr ? lc::overlay_session(overlay) : nullptr;
		va_list args;
		va_start(args, format);
		if (session != nullptr) {
			lc::vlog(session->callbacks, level, format, args);
		} else {
			const lc_callbacks none {};
			lc::vlog(none, level, format, args);
		}
		va_end(args);
	}

	// For host memory: one the CPU sees too, so no driver has to override a private type
	uint32_t pick_host_memory_type(uint32_t bits) const {
		const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		for (uint32_t i = 0; i < memory_props.memoryTypeCount; i++) {
			if ((bits & (1u << i)) != 0 && (memory_props.memoryTypes[i].propertyFlags & want) == want) {
				return i;
			}
		}
		return pick_memory_type(bits);
	}

	// Whether the GPU and CPU share memory (Apple silicon), which unlocked IOSurface memory needs:
	// a discrete GPU can keep an IOSurface in VRAM until it's locked
	bool unified_memory() const {
		const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		for (uint32_t i = 0; i < memory_props.memoryTypeCount; i++) {
			if ((memory_props.memoryTypes[i].propertyFlags & want) == want) {
				return true;
			}
		}
		return false;
	}

	uint32_t pick_memory_type(uint32_t bits) const {
		for (uint32_t i = 0; i < memory_props.memoryTypeCount; i++) {
			if ((bits & (1u << i)) != 0 && (memory_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
				return i;
			}
		}
		for (uint32_t i = 0; i < 32; i++) {
			if ((bits & (1u << i)) != 0) {
				return i;
			}
		}
		return 0;
	}

	// Whether linear BGRA8 dma-bufs can be made with `usage` and `feature`
	bool can_share(VkImageUsageFlags usage, VkExternalMemoryFeatureFlags feature) const {
#if defined(__linux__)
		VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifier {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT};
		modifier.drmFormatModifier = DRM_FORMAT_MOD_LINEAR;
		modifier.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
		VkPhysicalDeviceExternalImageFormatInfo external {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
		external.pNext      = &modifier;
		external.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
		VkPhysicalDeviceImageFormatInfo2 format_info {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
		format_info.pNext  = &external;
		format_info.format = FORMAT;
		format_info.type   = VK_IMAGE_TYPE_2D;
		format_info.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
		format_info.usage  = usage;
		VkExternalImageFormatProperties external_props {VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
		VkImageFormatProperties2        props {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
		props.pNext = &external_props;
		return vk.GetPhysicalDeviceImageFormatProperties2(info.physical_device, &format_info, &props) == VK_SUCCESS &&
		       (external_props.externalMemoryProperties.externalMemoryFeatures & feature) != 0;
#elif defined(LC_VK_HOST_SURFACES)
		// Host memory is only ever imported, so that's the one feature that matters
		(void)usage;
		(void)feature;
		VkPhysicalDeviceExternalBufferInfo buffer_info {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
		buffer_info.usage      = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		buffer_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
		VkExternalBufferProperties props {VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
		vk.GetPhysicalDeviceExternalBufferProperties(info.physical_device, &buffer_info, &props);
		return (props.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
#else
		(void)usage;
		(void)feature;
		return false;
#endif
	}

	// Imports `size` bytes of host memory at `pointer` (an IOSurface's, `mapped` bytes of it) as a
	// buffer. Both have to be multiples of minImportedHostPointerAlignment; an IOSurface's memory
	// is whole pages.
	bool import_host(Image& img, void* pointer, VkDeviceSize size, VkDeviceSize mapped, VkBufferUsageFlags usage, uint32_t id) {
		const VkDevice device = info.device;
		if (pointer == nullptr || host_alignment == 0 || reinterpret_cast<uintptr_t>(pointer) % host_alignment != 0) {
			log(LC_LOG_ERROR, "overlay: IOSurface %u's memory isn't aligned for importing (%p, alignment %llu)", id, pointer,
			    static_cast<unsigned long long>(host_alignment));
			return false;
		}
		size = round_up(size, host_alignment);
		if (size > round_up(mapped, host_alignment)) {
			log(LC_LOG_ERROR, "overlay: IOSurface %u is smaller (%llu bytes) than its rows need (%llu)", id,
			    static_cast<unsigned long long>(mapped), static_cast<unsigned long long>(size));
			return false;
		}

		VkExternalMemoryBufferCreateInfo external {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
		external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
		VkBufferCreateInfo ci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		ci.pNext       = &external;
		ci.size        = size;
		ci.usage       = usage;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vk.CreateBuffer(device, &ci, nullptr, &img.host) != VK_SUCCESS) {
			img.host = VK_NULL_HANDLE;
			log(LC_LOG_ERROR, "overlay: can't make a buffer of IOSurface %u's memory", id);
			return false;
		}
		VkMemoryRequirements req {};
		vk.GetBufferMemoryRequirements(device, img.host, &req);
		if (req.size > size) {
			log(LC_LOG_ERROR, "overlay: a buffer of IOSurface %u's memory needs more (%llu bytes) than it has", id,
			    static_cast<unsigned long long>(req.size));
			return false;
		}
		VkMemoryHostPointerPropertiesEXT pointer_props {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
		if (vk.GetMemoryHostPointerPropertiesEXT(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, pointer, &pointer_props) !=
		    VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: can't tell what memory IOSurface %u's is", id);
			return false;
		}
		const uint32_t types = req.memoryTypeBits & pointer_props.memoryTypeBits;
		if (types == 0) {
			log(LC_LOG_ERROR, "overlay: IOSurface %u's memory can't back a buffer", id);
			return false;
		}
		VkImportMemoryHostPointerInfoEXT import_info {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
		import_info.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
		import_info.pHostPointer = pointer;
		VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		alloc.pNext           = &import_info;
		alloc.allocationSize  = size;
		alloc.memoryTypeIndex = pick_host_memory_type(types);
		if (const VkResult r = vk.AllocateMemory(device, &alloc, nullptr, &img.host_memory); r != VK_SUCCESS) {
			img.host_memory = VK_NULL_HANDLE;
			log(LC_LOG_ERROR, "overlay: importing IOSurface %u's memory failed: VkResult %d", id, r);
			return false;
		}
		return vk.BindBufferMemory(device, img.host, img.host_memory, 0) == VK_SUCCESS;
	}

	// An optimal image of the helper's own, for a shell's buffer to be copied into and drawn
	bool create_local(Image& img, uint32_t width, uint32_t height) {
		const VkDevice device = info.device;
		const auto     ci = image_info(nullptr, width, height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL);
		if (vk.CreateImage(device, &ci, nullptr, &img.image) != VK_SUCCESS) {
			img.image = VK_NULL_HANDLE;
			return false;
		}
		VkMemoryRequirements req {};
		vk.GetImageMemoryRequirements(device, img.image, &req);
		VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		alloc.allocationSize  = req.size;
		alloc.memoryTypeIndex = pick_memory_type(req.memoryTypeBits);
		if (vk.AllocateMemory(device, &alloc, nullptr, &img.memory) != VK_SUCCESS) {
			img.memory = VK_NULL_HANDLE;
			return false;
		}
		return vk.BindImageMemory(device, img.image, img.memory, 0) == VK_SUCCESS && create_view(img);
	}

	VkImageCreateInfo image_info(const void* next, uint32_t width, uint32_t height, VkImageUsageFlags usage, VkImageTiling tiling,
	                             VkFormat format = FORMAT) const {
		VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		ci.pNext         = next;
		ci.imageType     = VK_IMAGE_TYPE_2D;
		ci.format        = format;
		ci.extent        = {width, height, 1};
		ci.mipLevels     = 1;
		ci.arrayLayers   = 1;
		ci.samples       = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling        = tiling;
		ci.usage         = usage;
		ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		return ci;
	}

	bool create_view(Image& img) const {
		VkImageViewCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		ci.image            = img.image;
		ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
		ci.format           = FORMAT;
		ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		return vk.CreateImageView(info.device, &ci, nullptr, &img.view) == VK_SUCCESS;
	}

	// Imports one of the shell's buffers
	bool import(Image& img, const lc_surface* surface, uint32_t width, uint32_t height) {
#if defined(__linux__)
		const VkDevice device = info.device;
		img.width             = width;
		img.height            = height;

		VkSubresourceLayout plane {};
		plane.offset   = lc_surface_offset(surface);
		plane.rowPitch = lc_surface_stride(surface);

		VkImageDrmFormatModifierExplicitCreateInfoEXT modifier {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
		modifier.drmFormatModifier           = lc_surface_modifier(surface);
		modifier.drmFormatModifierPlaneCount = 1;
		modifier.pPlaneLayouts               = &plane;

		VkExternalMemoryImageCreateInfo external {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
		external.pNext       = &modifier;
		external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

		const auto ci = image_info(&external, width, height, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT);
		if (const VkResult r = vk.CreateImage(device, &ci, nullptr, &img.image); r != VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: can't use dma-buf %u (stride %u, modifier 0x%llx): VkResult %d", lc_surface_id(surface),
			    lc_surface_stride(surface), static_cast<unsigned long long>(lc_surface_modifier(surface)), r);
			img.image = VK_NULL_HANDLE;
			return false;
		}

		VkMemoryRequirements req {};
		vk.GetImageMemoryRequirements(device, img.image, &req);

		// Vulkan takes the descriptor when the import works; the library keeps its own
		const int fd = ::dup(lc_surface_fd(surface));
		if (fd < 0) {
			return false;
		}
		VkMemoryFdPropertiesKHR fd_props {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
		if (vk.GetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fd_props) != VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: can't tell what memory dma-buf %u is", lc_surface_id(surface));
			::close(fd);
			return false;
		}
		const uint32_t types = req.memoryTypeBits & fd_props.memoryTypeBits;
		if (types == 0) {
			log(LC_LOG_ERROR, "overlay: dma-buf %u's memory can't back an image", lc_surface_id(surface));
			::close(fd);
			return false;
		}

		VkImportMemoryFdInfoKHR import_info {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
		import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
		import_info.fd         = fd;
		VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
		dedicated.pNext = &import_info;
		dedicated.image = img.image;
		VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		alloc.pNext           = &dedicated;
		alloc.allocationSize  = req.size;
		alloc.memoryTypeIndex = pick_memory_type(types);
		if (const VkResult r = vk.AllocateMemory(device, &alloc, nullptr, &img.memory); r != VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: importing dma-buf %u failed: VkResult %d", lc_surface_id(surface), r);
			img.memory = VK_NULL_HANDLE;
			::close(fd);
			return false;
		}
		return vk.BindImageMemory(device, img.image, img.memory, 0) == VK_SUCCESS && create_view(img);
#elif defined(LC_VK_HOST_SURFACES)
		auto* ref = static_cast<IOSurfaceRef>(lc_surface_iosurface(surface));
		if (ref == nullptr) {
			return false;
		}
		CFRetain(ref);
		img.surface = ref;
		img.width   = width;
		img.height  = height;
		img.stride  = static_cast<uint32_t>(IOSurfaceGetBytesPerRow(ref));
		if (IOSurfaceGetWidth(ref) < width || IOSurfaceGetHeight(ref) < height || img.stride % 4 != 0 || img.stride < width * 4) {
			log(LC_LOG_ERROR, "overlay: IOSurface %u isn't %ux%u BGRA8 (%zux%zu, %u bytes a row)", lc_surface_id(surface), width, height,
			    IOSurfaceGetWidth(ref), IOSurfaceGetHeight(ref), img.stride);
			return false;
		}
		const VkDeviceSize size = static_cast<VkDeviceSize>(img.stride) * height;
		return import_host(img, IOSurfaceGetBaseAddress(ref), size, IOSurfaceGetAllocSize(ref), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                   lc_surface_id(surface)) &&
		       create_local(img, width, height);
#else
		(void)img;
		(void)surface;
		(void)width;
		(void)height;
		return false;
#endif
	}

	// Imports the frame's generation, and gives each buffer a descriptor set to draw it with
	bool import_set(const lc_frame& frame) {
		BufferSet set;
		if (frame.buffer_count == 0) {
			current = std::move(set);
			return true;
		}
		VkDescriptorPoolSize       size {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frame.buffer_count};
		VkDescriptorPoolCreateInfo pool_ci {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		pool_ci.maxSets       = frame.buffer_count;
		pool_ci.poolSizeCount = 1;
		pool_ci.pPoolSizes    = &size;
		bool ok               = vk.CreateDescriptorPool(info.device, &pool_ci, nullptr, &set.pool) == VK_SUCCESS;
		for (uint32_t i = 0; ok && i < frame.buffer_count; i++) {
			Image& img = set.buffers.emplace_back();
			ok         = import(img, frame.buffers[i], frame.width, frame.height);
			if (ok) {
				VkDescriptorSetAllocateInfo alloc {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
				alloc.descriptorPool     = set.pool;
				alloc.descriptorSetCount = 1;
				alloc.pSetLayouts        = &set_layout;
				ok                       = vk.AllocateDescriptorSets(info.device, &alloc, &img.set) == VK_SUCCESS;
			}
			if (ok) {
				VkDescriptorImageInfo image {sampler, img.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
				VkWriteDescriptorSet  write {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
				write.dstSet          = img.set;
				write.descriptorCount = 1;
				write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				write.pImageInfo      = &image;
				vk.UpdateDescriptorSets(info.device, 1, &write, 0, nullptr);
			}
		}
		if (!ok) {
			free_set(set);
			current = {};
			return false;
		}
		log(LC_LOG_INFO, "overlay: imported %u buffer(s) of %ux%u", frame.buffer_count, frame.width, frame.height);
		current = std::move(set);
		return true;
	}

	void free_set(BufferSet& set) {
		for (auto& img: set.buffers) {
			img.destroy(vk, info.device);
		}
		if (set.pool != VK_NULL_HANDLE) {
			vk.DestroyDescriptorPool(info.device, set.pool, nullptr);
		}
		set = {};
	}

	void free_retired(uint64_t completed, bool all) {
		const auto done = [&](BufferSet& set) {
			if (!all && set.last_used != 0 && completed < set.last_used) {
				return false;
			}
			free_set(set);
			return true;
		};
		retired.erase(std::remove_if(retired.begin(), retired.end(), done), retired.end());
		const auto capture_done = [&](RetiredCapture& r) {
			if (!all && r.last_used != 0 && completed < r.last_used) {
				return false;
			}
			r.capture.destroy(vk, info.device);
			r.staging.destroy(vk, info.device);
			return true;
		};
		retired_captures.erase(std::remove_if(retired_captures.begin(), retired_captures.end(), capture_done), retired_captures.end());
	}

	// Whether a blit can read `source` (an optimal image, as a swapchain's are) and write `staging`
	bool can_blit(VkFormat source, VkFormat staging_format) const {
		VkFormatProperties src {};
		VkFormatProperties dst {};
		vk.GetPhysicalDeviceFormatProperties(info.physical_device, source, &src);
		vk.GetPhysicalDeviceFormatProperties(info.physical_device, staging_format, &dst);
		return (src.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0 &&
		       (dst.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
	}

	// A linear image whose memory can go to the shell as a dma-buf, and an optimal one of
	// `format`'s colour space to blit the frame into first: a blit into a linear image isn't
	// supported everywhere, a copy is
	//
	// The images are reused from one capture to the next. The GPU orders the copies, but the shell
	// may still be showing the last capture as a menu's background while the next one is written;
	// each menu comes with a capture of its own, so that's only ever for a moment.
	bool prepare_capture(uint32_t width, uint32_t height, VkFormat source_format) {
		const VkFormat want_staging = holds_linear(source_format) ? VK_FORMAT_B8G8R8A8_SRGB : FORMAT;
		if (!can_blit(source_format, want_staging)) {
			log(LC_LOG_ERROR, "overlay: the device can't blit from the frame's format (%d)", static_cast<int>(source_format));
			return false;
		}
		if (capture.exists() && capture.width == width && capture.height == height && staging_format == want_staging) {
			return true;
		}
		if (capture.exists() || staging.image != VK_NULL_HANDLE) {
			// An earlier copy into them may still be running
			retired_captures.push_back({capture, staging, capture_used});
			capture      = {};
			staging      = {};
			capture_used = 0;
		}
#if defined(__linux__)
		const VkDevice device = info.device;

		const uint64_t                             modifiers[] = {DRM_FORMAT_MOD_LINEAR};
		VkImageDrmFormatModifierListCreateInfoEXT modifier_list {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
		modifier_list.drmFormatModifierCount = 1;
		modifier_list.pDrmFormatModifiers    = modifiers;
		VkExternalMemoryImageCreateInfo external {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
		external.pNext       = &modifier_list;
		external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
		const auto ci = image_info(&external, width, height, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT);
		if (vk.CreateImage(device, &ci, nullptr, &capture.image) != VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: creating a linear %ux%u dma-buf failed", width, height);
			capture.image = VK_NULL_HANDLE;
			return false;
		}
		VkMemoryRequirements req {};
		vk.GetImageMemoryRequirements(device, capture.image, &req);
		VkExportMemoryAllocateInfo export_info {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
		export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
		VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
		dedicated.pNext = &export_info;
		dedicated.image = capture.image;
		VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		alloc.pNext           = &dedicated;
		alloc.allocationSize  = req.size;
		alloc.memoryTypeIndex = pick_memory_type(req.memoryTypeBits);
		if (vk.AllocateMemory(device, &alloc, nullptr, &capture.memory) != VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: allocating the capture's dma-buf failed");
			capture.memory = VK_NULL_HANDLE;
			capture.destroy(vk, device);
			return false;
		}
		if (vk.BindImageMemory(device, capture.image, capture.memory, 0) != VK_SUCCESS) {
			capture.destroy(vk, device);
			return false;
		}
		const VkImageSubresource plane {VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0};
		VkSubresourceLayout      layout_info {};
		vk.GetImageSubresourceLayout(device, capture.image, &plane, &layout_info);
		capture.width  = width;
		capture.height = height;
		capture.stride = static_cast<uint32_t>(layout_info.rowPitch);
		capture.offset = layout_info.offset;

		return create_staging(width, height, want_staging);
#elif defined(LC_VK_HOST_SURFACES)
		// An IOSurface of the helper's own, its rows and size whole multiples of what an import
		// takes, its memory a buffer to copy the frame into
		const size_t row   = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, static_cast<size_t>(width) * 4);
		const auto   alloc = static_cast<int64_t>(round_up(static_cast<VkDeviceSize>(row) * height, host_alignment));
		CFMutableDictionaryRef props = CFDictionaryCreateMutable(nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
		const auto put = [&](CFStringRef key, int64_t value) {
			CFNumberRef number = CFNumberCreate(nullptr, kCFNumberSInt64Type, &value);
			CFDictionarySetValue(props, key, number);
			CFRelease(number);
		};
		put(kIOSurfaceWidth, width);
		put(kIOSurfaceHeight, height);
		put(kIOSurfaceBytesPerElement, 4);
		put(kIOSurfaceBytesPerRow, static_cast<int64_t>(row));
		put(kIOSurfaceAllocSize, alloc);
		put(kIOSurfacePixelFormat, 0x42475241 /* 'BGRA' */);
		IOSurfaceRef ref = IOSurfaceCreate(props);
		CFRelease(props);
		if (ref == nullptr) {
			log(LC_LOG_ERROR, "overlay: creating a %ux%u IOSurface for the capture failed", width, height);
			return false;
		}
		capture.surface = ref;
		capture.width   = width;
		capture.height  = height;
		capture.stride  = static_cast<uint32_t>(IOSurfaceGetBytesPerRow(ref));
		if (!import_host(capture, IOSurfaceGetBaseAddress(ref), static_cast<VkDeviceSize>(capture.stride) * height,
		                 IOSurfaceGetAllocSize(ref), VK_BUFFER_USAGE_TRANSFER_DST_BIT, IOSurfaceGetID(ref))) {
			capture.destroy(vk, info.device);
			return false;
		}
		return create_staging(width, height, want_staging);
#else
		(void)width;
		(void)height;
		return false;
#endif
	}

	// The optimal image the frame is blitted into first
	bool create_staging(uint32_t width, uint32_t height, VkFormat want_staging) {
		const VkDevice device = info.device;
		VkMemoryRequirements req {};
		const auto staging_ci =
		    image_info(nullptr, width, height, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL, want_staging);
		if (vk.CreateImage(device, &staging_ci, nullptr, &staging.image) != VK_SUCCESS) {
			staging.image = VK_NULL_HANDLE;
			capture.destroy(vk, device);
			return false;
		}
		vk.GetImageMemoryRequirements(device, staging.image, &req);
		VkMemoryAllocateInfo staging_alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		staging_alloc.allocationSize  = req.size;
		staging_alloc.memoryTypeIndex = pick_memory_type(req.memoryTypeBits);
		if (vk.AllocateMemory(device, &staging_alloc, nullptr, &staging.memory) != VK_SUCCESS ||
		    vk.BindImageMemory(device, staging.image, staging.memory, 0) != VK_SUCCESS) {
			staging.destroy(vk, device);
			capture.destroy(vk, device);
			return false;
		}
		staging.width  = width;
		staging.height = height;
		staging_format = want_staging;
		return true;
	}

	// A new descriptor for the capture's memory, or -1
	int export_capture() const {
#if defined(__linux__)
		VkMemoryGetFdInfoKHR get {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
		get.memory     = capture.memory;
		get.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
		int fd         = -1;
		return capture.memory != VK_NULL_HANDLE && vk.GetMemoryFdKHR(info.device, &get, &fd) == VK_SUCCESS ? fd : -1;
#else
		return -1;
#endif
	}

	bool create_shared() {
		VkSamplerCreateInfo sampler_ci {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		sampler_ci.magFilter    = VK_FILTER_LINEAR;
		sampler_ci.minFilter    = VK_FILTER_LINEAR;
		sampler_ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler_ci.maxLod       = 0.0f;
		if (vk.CreateSampler(info.device, &sampler_ci, nullptr, &sampler) != VK_SUCCESS) {
			return false;
		}

		VkDescriptorSetLayoutBinding binding {};
		binding.binding         = 0;
		binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		binding.descriptorCount = 1;
		binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
		VkDescriptorSetLayoutCreateInfo set_ci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		set_ci.bindingCount = 1;
		set_ci.pBindings    = &binding;
		if (vk.CreateDescriptorSetLayout(info.device, &set_ci, nullptr, &set_layout) != VK_SUCCESS) {
			return false;
		}

		VkPushConstantRange        push {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(uint32_t)};
		VkPipelineLayoutCreateInfo layout_ci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		layout_ci.setLayoutCount         = 1;
		layout_ci.pSetLayouts            = &set_layout;
		layout_ci.pushConstantRangeCount = 1;
		layout_ci.pPushConstantRanges    = &push;
		if (vk.CreatePipelineLayout(info.device, &layout_ci, nullptr, &layout) != VK_SUCCESS) {
			return false;
		}

		const auto module = [&](const uint32_t* code, size_t size, VkShaderModule& out) {
			VkShaderModuleCreateInfo ci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
			ci.codeSize = size;
			ci.pCode    = code;
			return vk.CreateShaderModule(info.device, &ci, nullptr, &out) == VK_SUCCESS;
		};
		return module(OVERLAY_VERT, sizeof(OVERLAY_VERT), vert) && module(OVERLAY_FRAG, sizeof(OVERLAY_FRAG), frag);
	}

	VkPipeline pipeline_for(VkRenderPass render_pass, uint32_t subpass, VkFormat format) {
		for (const auto& p: pipelines) {
			if (p.render_pass == render_pass && p.subpass == subpass && p.format == format) {
				return p.pipeline;
			}
		}

		VkPipelineShaderStageCreateInfo stages[2] {};
		stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert;
		stages[0].pName  = "main";
		stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag;
		stages[1].pName  = "main";

		VkPipelineVertexInputStateCreateInfo   vertex_input {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		VkPipelineInputAssemblyStateCreateInfo assembly {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		VkPipelineViewportStateCreateInfo viewport {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		VkPipelineRasterizationStateCreateInfo raster {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode    = VK_CULL_MODE_NONE;
		raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth   = 1.0f;
		VkPipelineMultisampleStateCreateInfo multisample {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		VkPipelineDepthStencilStateCreateInfo depth {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
		// Premultiplied: out = menu + (1 - menu.a) * game
		VkPipelineColorBlendAttachmentState blend {};
		blend.blendEnable         = VK_TRUE;
		blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blend.colorBlendOp        = VK_BLEND_OP_ADD;
		blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blend.alphaBlendOp        = VK_BLEND_OP_ADD;
		blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blend_state {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		blend_state.attachmentCount = 1;
		blend_state.pAttachments    = &blend;
		const VkDynamicState             dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamic {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynamic.dynamicStateCount = 2;
		dynamic.pDynamicStates    = dynamic_states;

		VkPipelineRenderingCreateInfo rendering {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &format;

		VkGraphicsPipelineCreateInfo ci {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		ci.pNext               = render_pass == VK_NULL_HANDLE ? &rendering : nullptr;
		ci.stageCount          = 2;
		ci.pStages             = stages;
		ci.pVertexInputState   = &vertex_input;
		ci.pInputAssemblyState = &assembly;
		ci.pViewportState      = &viewport;
		ci.pRasterizationState = &raster;
		ci.pMultisampleState   = &multisample;
		ci.pDepthStencilState  = &depth;
		ci.pColorBlendState    = &blend_state;
		ci.pDynamicState       = &dynamic;
		ci.layout              = layout;
		ci.renderPass          = render_pass;
		ci.subpass             = subpass;

		VkPipeline pipeline = VK_NULL_HANDLE;
		if (const VkResult r = vk.CreateGraphicsPipelines(info.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline); r != VK_SUCCESS) {
			log(LC_LOG_ERROR, "overlay: creating the pipeline failed: VkResult %d", r);
			return VK_NULL_HANDLE;
		}
		pipelines.push_back({render_pass, subpass, format, pipeline});
		return pipeline;
	}

	void destroy() {
		free_set(current);
		free_retired(0, true);
		capture.destroy(vk, info.device);
		staging.destroy(vk, info.device);
		for (const auto& p: pipelines) {
			vk.DestroyPipeline(info.device, p.pipeline, nullptr);
		}
		if (vert != VK_NULL_HANDLE) {
			vk.DestroyShaderModule(info.device, vert, nullptr);
		}
		if (frag != VK_NULL_HANDLE) {
			vk.DestroyShaderModule(info.device, frag, nullptr);
		}
		if (layout != VK_NULL_HANDLE) {
			vk.DestroyPipelineLayout(info.device, layout, nullptr);
		}
		if (set_layout != VK_NULL_HANDLE) {
			vk.DestroyDescriptorSetLayout(info.device, set_layout, nullptr);
		}
		if (sampler != VK_NULL_HANDLE) {
			vk.DestroySampler(info.device, sampler, nullptr);
		}
	}
};

extern "C" {

const char* const* lc_vk_device_extensions(uint32_t* count) {
	if (count != nullptr) {
		*count = static_cast<uint32_t>(sizeof(DEVICE_EXTENSIONS) / sizeof(DEVICE_EXTENSIONS[0]));
	}
	return DEVICE_EXTENSIONS;
}

lc_vk* lc_vk_create(lc_overlay* overlay, const lc_vk_info* info, char* error, size_t error_len) {
	const auto fail = [&](lc_vk* vk, const std::string& why) -> lc_vk* {
		if (error != nullptr && error_len != 0) {
			std::snprintf(error, error_len, "%s", why.c_str());
		}
		if (vk != nullptr) {
			vk->log(LC_LOG_ERROR, "overlay: %s", why.c_str());
			vk->destroy();
			delete vk;
		}
		return nullptr;
	};
#if !defined(__linux__) && !defined(LC_VK_HOST_SURFACES)
	(void)overlay;
	(void)info;
	return fail(nullptr, "the Vulkan helper has no backend on this platform yet");
#else
	if (overlay == nullptr || info == nullptr || info->device == VK_NULL_HANDLE || info->get_instance_proc_addr == nullptr) {
		return fail(nullptr, "no overlay or no device");
	}
	auto* vk    = new lc_vk();
	vk->overlay = overlay;
	vk->info    = *info;
	if (const char* missing = vk->vk.load(*info)) {
		return fail(vk, std::string("this device has no ") + missing + " (enable lc_vk_device_extensions)");
	}
	vk->vk.GetPhysicalDeviceMemoryProperties(info->physical_device, &vk->memory_props);
#if defined(LC_VK_HOST_SURFACES)
	VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
	VkPhysicalDeviceProperties2                     props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	props.pNext = &host_props;
	vk->vk.GetPhysicalDeviceProperties2(info->physical_device, &props);
	// Metal takes host memory in whole pages (newBufferWithBytesNoCopy), whatever the driver says.
	// Apple silicon's kernel pages are 16 KiB even where getpagesize() says 4 KiB (x86_64 under
	// Rosetta), and IOSurfaces are allocated in them, so imports are 16 KiB-aligned and long. Some
	// drivers (KosmicKrisp) don't check what Metal makes of the import: a misaligned one becomes a
	// buffer at GPU address 0 with no error.
	constexpr VkDeviceSize apple_page = 16384;
	vk->host_alignment = std::max({host_props.minImportedHostPointerAlignment, static_cast<VkDeviceSize>(::getpagesize()), apple_page});
	if (host_props.minImportedHostPointerAlignment == 0) {
		vk->host_alignment = 0;
	}
	// Intel and AMD GPUs (Intel Macs) can't read the shell's IOSurfaces without locking them. Apple
	// silicon's GPU is the only other one on a Mac; drivers differ in the vendor ID they give it, so
	// that's left to the memory check.
	const uint32_t vendor = props.properties.vendorID;
	if (vendor == 0x8086 || vendor == 0x1002 || !vk->unified_memory()) {
		return fail(vk, "the launcher's IOSurfaces can only be shared without locking them on Apple silicon (unified memory)");
	}
	const bool can_import = vk->host_alignment != 0 && vk->can_share(0, VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
	vk->log(LC_LOG_INFO, "overlay: IOSurface memory import %s (alignment %llu)", can_import ? "yes" : "no",
	        static_cast<unsigned long long>(vk->host_alignment));
	if (!can_import) {
		return fail(vk, "this device can't import host memory (VK_EXT_external_memory_host), which the launcher's IOSurfaces need");
	}
#else
	const bool can_import = vk->can_share(VK_IMAGE_USAGE_SAMPLED_BIT, VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
	const bool can_export = vk->can_share(VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
	vk->log(LC_LOG_INFO, "overlay: linear BGRA8 dma-buf import %s, export %s", can_import ? "yes" : "no", can_export ? "yes" : "no");
	if (!can_import || !can_export) {
		return fail(vk, "this GPU can't share linear BGRA8 dma-bufs with the launcher");
	}
#endif
	if (!vk->create_shared()) {
		return fail(vk, "creating the overlay's sampler, layouts or shaders failed");
	}
	return vk;
#endif
}

void lc_vk_destroy(lc_vk* vk) {
	if (vk != nullptr) {
		vk->destroy();
		delete vk;
	}
}

void lc_vk_begin(lc_vk* vk, uint64_t completed, VkExtent2D window, lc_vk_frame* out) {
	if (out != nullptr) {
		*out = {};
	}
	if (vk == nullptr) {
		return;
	}
	lc_frame frame {};
	lc_overlay_begin(vk->overlay, completed, &frame);

	if (frame.generation != vk->generation) {
		vk->generation = frame.generation;
		if (!vk->current.buffers.empty() || vk->current.pool != VK_NULL_HANDLE) {
			vk->retired.push_back(std::move(vk->current));
			vk->current = {};
		}
		if (frame.buffer_count != 0) {
			lc_overlay_surfaces_ready(vk->overlay, frame.generation, vk->import_set(frame));
		}
	}
	vk->free_retired(completed, false);

	// Only a renderer that can draw the overlay says how big it should be
	if (window.width != 0 && window.height != 0) {
		vk->window = window;
		lc_overlay_view(vk->overlay, window.width, window.height);
	}

	const bool usable = frame.buffer_count != 0 && frame.buffer_count == vk->current.buffers.size();
	vk->draw          = usable ? frame.draw : -1;
	vk->fresh         = frame.fresh;
	vk->acquired      = false;
	vk->captured      = false;
	vk->capture_wanted = frame.capture;
	if (out != nullptr) {
		out->capture = frame.capture;
		out->wait    = frame.wait;
	}
}

void lc_vk_capture(lc_vk* vk, VkCommandBuffer command, VkImage source, VkFormat format, VkExtent2D extent) {
	if (vk == nullptr || !vk->capture_wanted) {
		return;
	}
	vk->capture_wanted = false;
	if (source == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0 || !vk->prepare_capture(extent.width, extent.height, format)) {
		lc_overlay_capture(vk->overlay, nullptr, 0);
		return;
	}
	const auto& f = vk->vk;

	// The frame, converted to BGRA8 by a blit, then copied to the linear image
	barrier(f, command, vk->staging.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkImageBlit blit {};
	blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blit.srcOffsets[1]  = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
	blit.dstOffsets[1]  = blit.srcOffsets[1];
	f.CmdBlitImage(command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk->staging.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
	               VK_FILTER_NEAREST);
	barrier(f, command, vk->staging.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	if (vk->capture.host != VK_NULL_HANDLE) {
		// Into the IOSurface's memory, for the shell to read once the submission is done
		VkBufferImageCopy copy {};
		copy.bufferRowLength  = vk->capture.stride / 4;
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent      = {extent.width, extent.height, 1};
		f.CmdCopyImageToBuffer(command, vk->staging.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk->capture.host, 1, &copy);
		memory_barrier(f, command, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		               VK_PIPELINE_STAGE_HOST_BIT);
		vk->captured = true;
		vk->log(LC_LOG_INFO, "overlay: capturing the game's frame (%ux%u)", extent.width, extent.height);
		return;
	}

	// Linear, so taking it from UNDEFINED loses nothing it needs
	barrier(f, command, vk->capture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkImageCopy copy {};
	copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy.extent         = {extent.width, extent.height, 1};
	f.CmdCopyImage(command, vk->staging.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk->capture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
	               &copy);
	// Released to the shell's process. EXTERNAL is for another process on this device; a shell
	// drawing on another GPU (a hybrid laptop's iGPU) would want VK_QUEUE_FAMILY_FOREIGN_EXT
	barrier(f, command, vk->capture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
	        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vk->info.queue_family, VK_QUEUE_FAMILY_EXTERNAL);
	vk->captured     = true;
	vk->log(LC_LOG_INFO, "overlay: capturing the game's frame (%ux%u)", extent.width, extent.height);
}

VkImageView lc_vk_acquire(lc_vk* vk, VkCommandBuffer command) {
	if (vk == nullptr || vk->draw < 0) {
		return VK_NULL_HANDLE;
	}
	Image& buffer = vk->current.buffers[static_cast<size_t>(vk->draw)];
	if (buffer.host != VK_NULL_HANDLE) {
		// The shell wrote the IOSurface from its own process: its memory, copied into the image the
		// menu is drawn from, whenever the shell has written it since
		if (buffer.loaded && !vk->fresh) {
			vk->acquired = true;
			return buffer.view;
		}
		const auto& f = vk->vk;
		memory_barrier(f, command, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT,
		               VK_PIPELINE_STAGE_TRANSFER_BIT);
		barrier(f, command, buffer.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
		        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy copy {};
		copy.bufferRowLength  = buffer.stride / 4;
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent      = {buffer.width, buffer.height, 1};
		f.CmdCopyBufferToImage(command, buffer.host, buffer.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		barrier(f, command, buffer.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
		buffer.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		buffer.loaded = true;
		vk->acquired  = true;
		return buffer.view;
	}
	// The shell wrote it from its own process: acquire it from there, which makes its writes
	// visible. It's linear, so leaving UNDEFINED loses nothing.
	if (vk->fresh || buffer.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier(vk->vk, command, buffer.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT,
		        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_QUEUE_FAMILY_EXTERNAL, vk->info.queue_family);
		buffer.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}
	vk->acquired = true;
	return buffer.view;
}

void lc_vk_draw(lc_vk* vk, VkCommandBuffer command, VkRenderPass render_pass, uint32_t subpass, VkFormat format) {
	if (vk == nullptr || !vk->acquired || vk->window.width == 0) {
		return;
	}
	const VkPipeline pipeline = vk->pipeline_for(render_pass, subpass, format);
	if (pipeline == VK_NULL_HANDLE) {
		return;
	}
	const auto&      f   = vk->vk;
	const Image&     buf = vk->current.buffers[static_cast<size_t>(vk->draw)];
	const VkViewport viewport {0, 0, static_cast<float>(vk->window.width), static_cast<float>(vk->window.height), 0, 1};
	const VkRect2D   scissor {{0, 0}, vk->window};
	const uint32_t   decode = holds_linear(format) ? 1 : 0;
	f.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	f.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, vk->layout, 0, 1, &buf.set, 0, nullptr);
	f.CmdPushConstants(command, vk->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(decode), &decode);
	f.CmdSetViewport(command, 0, 1, &viewport);
	f.CmdSetScissor(command, 0, 1, &scissor);
	f.CmdDraw(command, 3, 1, 0, 0);
}

void lc_vk_end(lc_vk* vk, uint64_t serial) {
	if (vk == nullptr) {
		return;
	}
	if (vk->acquired) {
		vk->current.last_used = serial;
	}
	lc_overlay_end(vk->overlay, serial, vk->acquired);
	if (vk->captured) {
		vk->capture_used = serial;
		lc_capture capture {};
#if defined(LC_VK_HOST_SURFACES)
		capture.fd        = -1;
		capture.iosurface = vk->capture.surface;
		capture.stride    = vk->capture.stride;
		capture.width     = vk->capture.width;
		capture.height    = vk->capture.height;
		if (capture.iosurface == nullptr) {
			vk->log(LC_LOG_ERROR, "overlay: no IOSurface for the capture");
			lc_overlay_capture(vk->overlay, nullptr, 0);
		} else {
			lc_overlay_capture(vk->overlay, &capture, serial);
		}
#else
		capture.fd       = vk->export_capture();
		capture.stride   = vk->capture.stride;
		capture.offset   = vk->capture.offset;
		capture.modifier = DRM_FORMAT_MOD_LINEAR;
		capture.width    = vk->capture.width;
		capture.height   = vk->capture.height;
		if (capture.fd < 0) {
			vk->log(LC_LOG_ERROR, "overlay: couldn't export the capture's dma-buf");
			lc_overlay_capture(vk->overlay, nullptr, 0);
		} else {
			lc_overlay_capture(vk->overlay, &capture, serial);
		}
#endif
	}
	if (vk->capture_wanted) {
		// Asked for, and lc_vk_capture wasn't called: the shell comes to the front instead
		lc_overlay_capture(vk->overlay, nullptr, 0);
	}
	vk->acquired       = false;
	vk->captured       = false;
	vk->capture_wanted = false;
}

} // extern "C"
