// The Vulkan helper on a real GPU: the "shell" fills the first of its three buffers with a
// half-transparent red, the helper imports them, captures a blue game frame, and draws the menu
// over it. The result has to be the premultiplied blend of the two.
// Linux: the buffers are linear dma-bufs exported from the same device; skipped where no device
// can share them (lavapipe has no DRM format modifiers).
// macOS: the buffers are IOSurfaces, as the shell makes them, and the capture is read back from
// the IOSurface the shell gets. LC_TEST_VULKAN names the library to load (the Vulkan loader, or a
// driver such as libMoltenVK.dylib); the loader's VK_DRIVER_FILES picks KosmicKrisp.
#include "check.h"
#include "fake_launcher.h"
#if defined(__APPLE__)
#include "fake_shell_macos.h"
#else
#include "fake_shell.h"
#endif
#include "vk_gpu.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t SIZE = 64;

struct Owned {
	VkImage        image  = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
};

#if !defined(__APPLE__)
// A linear BGRA8 dma-buf as the shell makes one, cleared to `color`
Owned shell_buffer(Gpu& gpu, VkClearColorValue color, int& fd, uint32_t& stride) {
	Owned                                     out;
	const uint64_t                            linear = 0;
	VkImageDrmFormatModifierListCreateInfoEXT list {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT};
	list.drmFormatModifierCount = 1;
	list.pDrmFormatModifiers    = &linear;
	VkExternalMemoryImageCreateInfo external {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
	external.pNext       = &list;
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.pNext       = &external;
	ci.imageType   = VK_IMAGE_TYPE_2D;
	ci.format      = VK_FORMAT_B8G8R8A8_UNORM;
	ci.extent      = {SIZE, SIZE, 1};
	ci.mipLevels   = 1;
	ci.arrayLayers = 1;
	ci.samples     = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling      = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	ci.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	gpu.CreateImage(gpu.device, &ci, nullptr, &out.image);
	VkMemoryRequirements req {};
	gpu.GetImageMemoryRequirements(gpu.device, out.image, &req);
	VkExportMemoryAllocateInfo export_info {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
	export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkMemoryDedicatedAllocateInfo dedicated {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
	dedicated.pNext = &export_info;
	dedicated.image = out.image;
	VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc.pNext           = &dedicated;
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = gpu.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	gpu.AllocateMemory(gpu.device, &alloc, nullptr, &out.memory);
	gpu.BindImageMemory(gpu.device, out.image, out.memory, 0);
	const VkImageSubresource plane {VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0};
	VkSubresourceLayout      layout {};
	gpu.GetImageSubresourceLayout(gpu.device, out.image, &plane, &layout);
	stride = static_cast<uint32_t>(layout.rowPitch);

	gpu.begin();
	gpu.barrier(out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
	const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	gpu.CmdClearColorImage(gpu.command, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
	gpu.barrier(out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0, gpu.family,
	            VK_QUEUE_FAMILY_EXTERNAL);
	gpu.submit();

	VkMemoryGetFdInfoKHR get {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
	get.memory     = out.memory;
	get.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	gpu.GetMemoryFdKHR(gpu.device, &get, &fd);
	return out;
}
#else
// Fills an IOSurface with one BGRA colour, as the shell's renderer would
void fill(IOSurfaceRef surface, const uint8_t (&bgra)[4]) {
	IOSurfaceLock(surface, 0, nullptr);
	auto*        base = static_cast<uint8_t*>(IOSurfaceGetBaseAddress(surface));
	const size_t row  = IOSurfaceGetBytesPerRow(surface);
	for (size_t y = 0; y < IOSurfaceGetHeight(surface); y++) {
		for (size_t x = 0; x < IOSurfaceGetWidth(surface); x++) {
			std::memcpy(base + y * row + x * 4, bgra, 4);
		}
	}
	IOSurfaceUnlock(surface, 0, nullptr);
}
#endif

void run(Gpu& gpu) {
	// The shell's buffers: 0 is premultiplied red at half alpha, 1 green at half alpha, 2 clear
	std::vector<Owned> owned;
#if defined(__APPLE__)
	FakeShell shell;
	fill(shell.buffer(0), {0, 0, 128, 128});
	fill(shell.buffer(1), {0, 128, 0, 128});
	fill(shell.buffer(2), {0, 0, 0, 0});
	const std::string offer = shell.offer(SIZE);
#else
	std::vector<int> fds(3, -1);
	uint32_t         stride = 0;
	for (int i = 0; i < 3; i++) {
		const VkClearColorValue color = i == 0   ? VkClearColorValue {{0.5f, 0.0f, 0.0f, 0.5f}}
		                                : i == 1 ? VkClearColorValue {{0.0f, 0.5f, 0.0f, 0.5f}}
		                                         : VkClearColorValue {{0, 0, 0, 0}};
		owned.push_back(shell_buffer(gpu, color, fds[static_cast<size_t>(i)], stride));
	}
	FakeShell         shell(fds, stride);
	const std::string offer = R"({"type":"overlay_surface","buffers":[11,12,13],"header":10,"service":")" + shell.name() +
	                          R"(","width":64,"height":64})";
#endif
	FakeLauncher launcher;

	lc_callbacks cb {};
	cb.log = [](void*, lc_log_level, const char* line) {
		if (std::getenv("LC_TEST_LOG") != nullptr) {
			std::fprintf(stderr, "log: %s\n", line);
		}
	};
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	lc_overlay* o     = lc_overlay_get(s);
	CHECK(o != nullptr);

	lc_vk_info vi {};
	vi.instance               = gpu.instance;
	vi.physical_device        = gpu.physical;
	vi.device                 = gpu.device;
	vi.queue_family           = gpu.family;
	vi.get_instance_proc_addr = gpu.gipa;
	char   error[256] {};
	lc_vk* vk = lc_vk_create(o, &vi, error, sizeof(error));
	if (vk == nullptr) {
		FAIL("lc_vk_create: %s", error);
		lc_close(s);
		return;
	}
	lc_ready(s, 0, 0);

	// The game's frame and its render pass, which loads it and draws the menu over it
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.imageType   = VK_IMAGE_TYPE_2D;
	ci.format      = VK_FORMAT_B8G8R8A8_UNORM;
	ci.extent      = {SIZE, SIZE, 1};
	ci.mipLevels   = 1;
	ci.arrayLayers = 1;
	ci.samples     = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
	ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	VkImage target = VK_NULL_HANDLE;
	gpu.CreateImage(gpu.device, &ci, nullptr, &target);
	VkMemoryRequirements req {};
	gpu.GetImageMemoryRequirements(gpu.device, target, &req);
	VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = gpu.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VkDeviceMemory target_memory = VK_NULL_HANDLE;
	gpu.AllocateMemory(gpu.device, &alloc, nullptr, &target_memory);
	gpu.BindImageMemory(gpu.device, target, target_memory, 0);
	VkImageViewCreateInfo vci {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vci.image            = target;
	vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
	vci.format           = VK_FORMAT_B8G8R8A8_UNORM;
	vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	VkImageView target_view = VK_NULL_HANDLE;
	gpu.CreateImageView(gpu.device, &vci, nullptr, &target_view);

	VkAttachmentDescription attachment {};
	attachment.format        = VK_FORMAT_B8G8R8A8_UNORM;
	attachment.samples       = VK_SAMPLE_COUNT_1_BIT;
	attachment.loadOp        = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachment.storeOp       = VK_ATTACHMENT_STORE_OP_STORE;
	attachment.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	attachment.finalLayout   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	VkAttachmentReference ref {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription  subpass {};
	subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &ref;
	VkRenderPassCreateInfo rci {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
	rci.attachmentCount = 1;
	rci.pAttachments    = &attachment;
	rci.subpassCount    = 1;
	rci.pSubpasses      = &subpass;
	VkRenderPass pass   = VK_NULL_HANDLE;
	gpu.CreateRenderPass(gpu.device, &rci, nullptr, &pass);
	VkFramebufferCreateInfo fci {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
	fci.renderPass      = pass;
	fci.attachmentCount = 1;
	fci.pAttachments    = &target_view;
	fci.width           = SIZE;
	fci.height          = SIZE;
	fci.layers          = 1;
	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	gpu.CreateFramebuffer(gpu.device, &fci, nullptr, &framebuffer);

	// Where the result is read back
	VkBufferCreateInfo bci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size  = SIZE * SIZE * 4;
	bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBuffer readback = VK_NULL_HANDLE;
	gpu.CreateBuffer(gpu.device, &bci, nullptr, &readback);
	gpu.GetBufferMemoryRequirements(gpu.device, readback, &req);
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = gpu.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VkDeviceMemory readback_memory = VK_NULL_HANDLE;
	gpu.AllocateMemory(gpu.device, &alloc, nullptr, &readback_memory);
	gpu.BindBufferMemory(gpu.device, readback, readback_memory, 0);

	// The menu opens over the game
	launcher.send(R"({"type":"overlay","pause":true,"input":true})");
	launcher.send(offer);
	shell.publish(0);
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	void* mapped = nullptr;
	gpu.MapMemory(gpu.device, readback_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	const auto* px   = static_cast<const uint8_t*>(mapped) + (SIZE / 2 * SIZE + SIZE / 2) * 4;
	const auto  near = [](int got, int want) { return got >= want - 3 && got <= want + 3; };
	const auto  is   = [&](int b, int g, int r, int a) { return near(px[0], b) && near(px[1], g) && near(px[2], r) && near(px[3], a); };

	bool     drew      = false;
	uint64_t completed = 0;
	uint64_t serial    = 0;
	// One present of the game's frame with the menu over it, read back into `px`
	const auto present = [&]() {
		serial++;
		lc_vk_frame frame {};
		lc_vk_begin(vk, completed, {SIZE, SIZE}, true, &frame);
		gpu.begin();
		// The game's frame: blue
		gpu.barrier(target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
		const VkClearColorValue       blue {{0, 0, 1, 1}};
		const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		gpu.CmdClearColorImage(gpu.command, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &blue, 1, &range);
		gpu.barrier(target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
		if (frame.capture) {
			lc_vk_capture(vk, gpu.command, target, VK_FORMAT_B8G8R8A8_UNORM, {SIZE, SIZE});
		}
		const VkImageView menu = lc_vk_acquire(vk, gpu.command);
		VkRenderPassBeginInfo rbi {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rbi.renderPass  = pass;
		rbi.framebuffer = framebuffer;
		rbi.renderArea  = {{0, 0}, {SIZE, SIZE}};
		gpu.CmdBeginRenderPass(gpu.command, &rbi, VK_SUBPASS_CONTENTS_INLINE);
		if (menu != VK_NULL_HANDLE) {
			lc_vk_draw(vk, gpu.command, pass, 0, VK_FORMAT_B8G8R8A8_UNORM);
			drew = true;
		}
		gpu.CmdEndRenderPass(gpu.command);
		VkBufferImageCopy copy {};
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent      = {SIZE, SIZE, 1};
		gpu.CmdCopyImageToBuffer(gpu.command, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
		gpu.submit();
		lc_vk_end(vk, serial);
		completed = serial;
	};
	for (int i = 0; i < 20; i++) {
		present();
		if (drew && launcher.count("capture_surface") != 0) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	CHECK(drew);
	CHECK(launcher.wait_for("view").has_value());
#if defined(__APPLE__)
	CHECK(shell.wait_capture().has_value());
	// The capture is the game's frame as it was before the menu went over it: blue
	if (IOSurfaceRef capture = shell.capture_surface()) {
		IOSurfaceLock(capture, kIOSurfaceLockReadOnly, nullptr);
		const auto* base = static_cast<const uint8_t*>(IOSurfaceGetBaseAddress(capture));
		const auto* cp   = base + IOSurfaceGetBytesPerRow(capture) * (SIZE / 2) + (SIZE / 2) * 4;
		const int bgra[4] = {cp[0], cp[1], cp[2], cp[3]};
		IOSurfaceUnlock(capture, kIOSurfaceLockReadOnly, nullptr);
		if (!(bgra[0] == 255 && bgra[1] == 0 && bgra[2] == 0 && bgra[3] == 255)) {
			FAIL("captured pixel is BGRA %d %d %d %d, not 255 0 0 255", bgra[0], bgra[1], bgra[2], bgra[3]);
		}
	}
#else
	CHECK(shell.wait_capture().has_value() && shell.capture_had_fd());
#endif
	CHECK(launcher.wait_for("capture_surface").has_value());

	// menu + (1 - menu.a) * game = (0.5, 0, 0, 0.5) + 0.5 * (0, 0, 1, 1), as BGRA bytes
	if (!is(128, 0, 128, 255)) {
		FAIL("drawn pixel is BGRA %d %d %d %d, not about 128 0 128 255", px[0], px[1], px[2], px[3]);
	}

	// The menu changes: the shell publishes buffer 1, and the next presents draw it, not the
	// copy of buffer 0 (on macOS, a buffer is copied again whenever the shell publishes it)
	shell.publish(1);
	for (int i = 0; i < 20 && !is(128, 128, 0, 255); i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		present();
	}
	// (0, 0.5, 0, 0.5) + 0.5 * (0, 0, 1, 1)
	if (!is(128, 128, 0, 255)) {
		FAIL("after the menu changed, drawn pixel is BGRA %d %d %d %d, not about 128 128 0 255", px[0], px[1], px[2], px[3]);
	}
#if defined(__APPLE__)
	// The shell draws into buffer 0 again, which the helper copied before and has handed back, and
	// publishes it: the helper has to copy it again, not draw its old red copy (128 0 128 255)
	fill(shell.buffer(0), {128, 0, 0, 128});
	shell.publish(0);
	for (int i = 0; i < 20 && !is(255, 0, 0, 255); i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		present();
	}
	// (0.5, 0, 0, 0.5) as BGRA + 0.5 * (0, 0, 1, 1) = blue
	if (!is(255, 0, 0, 255)) {
		FAIL("after buffer 0 was redrawn, drawn pixel is BGRA %d %d %d %d, not about 255 0 0 255", px[0], px[1], px[2], px[3]);
	}
#endif

	gpu.DeviceWaitIdle(gpu.device);
	lc_vk_destroy(vk);
	lc_close(s);
	gpu.DestroyFramebuffer(gpu.device, framebuffer, nullptr);
	gpu.DestroyRenderPass(gpu.device, pass, nullptr);
	gpu.DestroyImageView(gpu.device, target_view, nullptr);
	gpu.DestroyImage(gpu.device, target, nullptr);
	gpu.FreeMemory(gpu.device, target_memory, nullptr);
	gpu.DestroyBuffer(gpu.device, readback, nullptr);
	gpu.FreeMemory(gpu.device, readback_memory, nullptr);
	for (auto& b: owned) {
		gpu.DestroyImage(gpu.device, b.image, nullptr);
		gpu.FreeMemory(gpu.device, b.memory, nullptr);
	}
}

} // namespace

int main() {
	Gpu               gpu;
	const std::string skip = gpu.open(true);
	if (!skip.empty()) {
		std::printf("vk: skipped (%s)\n", skip.c_str());
		gpu.close();
		return SKIPPED;
	}
	run(gpu);
	gpu.close();
	return finish("vk");
}
