// A Vulkan device for the helper's tests (vk_test.cpp, vk_list_test.cpp), loaded at run time: the
// Vulkan loader, or the library LC_TEST_VULKAN names.
#pragma once

#include "launcher_overlay_vk.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

#define LC_TEST_FNS(X)                     \
	X(EnumeratePhysicalDevices)            \
	X(EnumerateDeviceExtensionProperties)  \
	X(GetPhysicalDeviceQueueFamilyProperties) \
	X(GetPhysicalDeviceMemoryProperties)   \
	X(GetPhysicalDeviceProperties)         \
	X(GetPhysicalDeviceFeatures)           \
	X(CreateDevice)                        \
	X(DestroyInstance)                     \
	X(DestroyDevice)                       \
	X(GetDeviceQueue)                      \
	X(DeviceWaitIdle)                      \
	X(QueueSubmit)                         \
	X(QueueWaitIdle)                       \
	X(CreateCommandPool)                   \
	X(DestroyCommandPool)                  \
	X(AllocateCommandBuffers)              \
	X(BeginCommandBuffer)                  \
	X(EndCommandBuffer)                    \
	X(ResetCommandBuffer)                  \
	X(CreateImage)                         \
	X(DestroyImage)                        \
	X(CreateImageView)                     \
	X(DestroyImageView)                    \
	X(GetImageMemoryRequirements)          \
	X(GetImageSubresourceLayout)           \
	X(CreateBuffer)                        \
	X(DestroyBuffer)                       \
	X(GetBufferMemoryRequirements)         \
	X(BindBufferMemory)                    \
	X(AllocateMemory)                      \
	X(FreeMemory)                          \
	X(BindImageMemory)                     \
	X(MapMemory)                           \
	X(GetMemoryFdKHR)                      \
	X(CreateRenderPass)                    \
	X(DestroyRenderPass)                   \
	X(CreateFramebuffer)                   \
	X(DestroyFramebuffer)                  \
	X(CmdBeginRenderPass)                  \
	X(CmdEndRenderPass)                    \
	X(CmdClearColorImage)                  \
	X(CmdPipelineBarrier)                  \
	X(CmdCopyImageToBuffer)                \
	X(CmdCopyBufferToImage)

struct Gpu {
	void*                     lib  = nullptr;
	PFN_vkGetInstanceProcAddr gipa = nullptr;
	VkInstance                instance = VK_NULL_HANDLE;
	VkPhysicalDevice          physical = VK_NULL_HANDLE;
	VkDevice                  device   = VK_NULL_HANDLE;
	uint32_t                  family   = 0;
	VkQueue                   queue    = VK_NULL_HANDLE;
	VkCommandPool             pool     = VK_NULL_HANDLE;
	VkCommandBuffer           command  = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties memory {};
	bool                      bc       = false; // textureCompressionBC, enabled
	bool                      timestamps = false; // The queue writes timestamps
#define LC_TEST_DECLARE(name) PFN_vk##name name = nullptr;
	LC_TEST_FNS(LC_TEST_DECLARE)
#undef LC_TEST_DECLARE

	// The reason to skip, or "" once it has a device: with `shares`, one the helper can share
	// buffers with (lc_vk_device_extensions); without, any, with BC textures if it has them
	std::string open(bool shares) {
#if defined(__APPLE__)
		std::vector<std::string> names = {"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"};
#else
		std::vector<std::string> names = {"libvulkan.so.1"};
#endif
		if (const char* named = std::getenv("LC_TEST_VULKAN")) {
			names = {named};
		}
		for (const auto& name: names) {
			lib = lib != nullptr ? lib : ::dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
		}
		if (lib == nullptr) {
			return "no Vulkan loader";
		}
		gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(::dlsym(lib, "vkGetInstanceProcAddr"));
		if (gipa == nullptr) {
			// A driver named straight (a Mesa ICD such as KosmicKrisp) has only the ICD entry point
			gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(::dlsym(lib, "vk_icdGetInstanceProcAddr"));
			// What the loader would tell it first
			using Negotiate = VkResult (*)(uint32_t*);
			if (auto negotiate = reinterpret_cast<Negotiate>(::dlsym(lib, "vk_icdNegotiateLoaderICDInterfaceVersion"))) {
				uint32_t version = 5;
				negotiate(&version);
			}
		}
		if (gipa == nullptr) {
			return "no vkGetInstanceProcAddr";
		}
		auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
		auto instance_extensions =
		    reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
		VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
		app.apiVersion = VK_API_VERSION_1_3;
		VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		ici.pApplicationInfo = &app;
		// The Vulkan loader lists MoltenVK and KosmicKrisp only to an instance that asks for them
		const char* portability = "VK_KHR_portability_enumeration";
		if (instance_extensions != nullptr) {
			uint32_t n = 0;
			instance_extensions(nullptr, &n, nullptr);
			std::vector<VkExtensionProperties> have(n);
			instance_extensions(nullptr, &n, have.data());
			for (const auto& e: have) {
				if (std::string(e.extensionName) == portability) {
					ici.flags                   = 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
					ici.enabledExtensionCount   = 1;
					ici.ppEnabledExtensionNames = &portability;
				}
			}
		}
		if (create_instance == nullptr || create_instance(&ici, nullptr, &instance) != VK_SUCCESS) {
			return "no Vulkan instance";
		}
#define LC_TEST_LOAD(name) name = reinterpret_cast<PFN_vk##name>(gipa(instance, "vk" #name));
		LC_TEST_FNS(LC_TEST_LOAD)
#undef LC_TEST_LOAD

		uint32_t                 extension_count = 0;
		const char* const*       wanted          = lc_vk_device_extensions(&extension_count);
		if (!shares) {
			extension_count = 0;
		}
		std::vector<const char*> enable(wanted, wanted + extension_count);
		uint32_t                 count = 0;
		EnumeratePhysicalDevices(instance, &count, nullptr);
		std::vector<VkPhysicalDevice> devices(count);
		EnumeratePhysicalDevices(instance, &count, devices.data());
		for (auto candidate: devices) {
			uint32_t n = 0;
			EnumerateDeviceExtensionProperties(candidate, nullptr, &n, nullptr);
			std::vector<VkExtensionProperties> have(n);
			EnumerateDeviceExtensionProperties(candidate, nullptr, &n, have.data());
			bool all = true;
			for (uint32_t i = 0; i < extension_count; i++) {
				bool found = false;
				for (const auto& e: have) {
					found = found || std::string(e.extensionName) == wanted[i];
				}
				all = all && found;
			}
			if (all) {
				physical = candidate;
				for (const auto& e: have) {
					// A portability driver's device has to be made with this when it has it
					if (std::string(e.extensionName) == "VK_KHR_portability_subset") {
						enable.push_back("VK_KHR_portability_subset");
					}
				}
				break;
			}
		}
		if (physical == VK_NULL_HANDLE) {
			return "no device with the extensions the helper needs";
		}
		VkPhysicalDeviceProperties props {};
		GetPhysicalDeviceProperties(physical, &props);
		std::printf("vk: using %s (vendor 0x%04x, driver 0x%x)\n", props.deviceName, props.vendorID, props.driverVersion);
		GetPhysicalDeviceMemoryProperties(physical, &memory);

		uint32_t families = 0;
		GetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
		std::vector<VkQueueFamilyProperties> family_props(families);
		GetPhysicalDeviceQueueFamilyProperties(physical, &families, family_props.data());
		for (uint32_t i = 0; i < families; i++) {
			if ((family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
				family     = i;
				timestamps = family_props[i].timestampValidBits != 0;
				break;
			}
		}
		const float             priority = 1.0f;
		VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		qci.queueFamilyIndex = family;
		qci.queueCount       = 1;
		qci.pQueuePriorities = &priority;
		VkPhysicalDeviceFeatures have_features {};
		GetPhysicalDeviceFeatures(physical, &have_features);
		VkPhysicalDeviceFeatures features {};
		features.textureCompressionBC = have_features.textureCompressionBC;
		bc                            = have_features.textureCompressionBC == VK_TRUE;
		VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		dci.pEnabledFeatures = &features;
		dci.queueCreateInfoCount    = 1;
		dci.pQueueCreateInfos       = &qci;
		dci.enabledExtensionCount   = static_cast<uint32_t>(enable.size());
		dci.ppEnabledExtensionNames = enable.data();
		if (CreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) {
			return "creating the device failed";
		}
		// Device functions through the instance's dispatch work for one device
		GetDeviceQueue(device, family, 0, &queue);
		VkCommandPoolCreateInfo pci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pci.queueFamilyIndex = family;
		CreateCommandPool(device, &pci, nullptr, &pool);
		VkCommandBufferAllocateInfo cai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cai.commandPool        = pool;
		cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cai.commandBufferCount = 1;
		AllocateCommandBuffers(device, &cai, &command);
		return "";
	}

	uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) const {
		for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
			if ((bits & (1u << i)) != 0 && (memory.memoryTypes[i].propertyFlags & flags) == flags) {
				return i;
			}
		}
		return 0;
	}

	void begin() {
		ResetCommandBuffer(command, 0);
		VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		BeginCommandBuffer(command, &bi);
	}

	void submit() {
		EndCommandBuffer(command);
		VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		si.commandBufferCount = 1;
		si.pCommandBuffers    = &command;
		QueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
		QueueWaitIdle(queue);
	}

	void barrier(VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst, uint32_t src_queue = VK_QUEUE_FAMILY_IGNORED,
	             uint32_t dst_queue = VK_QUEUE_FAMILY_IGNORED) {
		VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		b.srcAccessMask       = src;
		b.dstAccessMask       = dst;
		b.oldLayout           = from;
		b.newLayout           = to;
		b.srcQueueFamilyIndex = src_queue;
		b.dstQueueFamilyIndex = dst_queue;
		b.image               = image;
		b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		CmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
	}

	void close() {
		if (device != VK_NULL_HANDLE) {
			DeviceWaitIdle(device);
			DestroyCommandPool(device, pool, nullptr);
			DestroyDevice(device, nullptr);
		}
		if (instance != VK_NULL_HANDLE) {
			DestroyInstance(instance, nullptr);
		}
		if (lib != nullptr) {
			::dlclose(lib);
		}
	}
};

} // namespace
