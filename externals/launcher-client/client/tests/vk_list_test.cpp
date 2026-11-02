// The Vulkan helper drawing draw lists on a real GPU (lavapipe will do: nothing is shared with
// the shell but memory). The "shell" publishes the menu the renderer's test recorded
// (data/scene.ldl, from crates/render/tests/list.rs), the game's frame is red on the left and blue
// on the right, and what the helper draws over it has to match what the shell's own renderer drew
// from the same list (data/scene.rgba) within the bounds that test sets.
//
// The helper draws the list into its canvas, then over the game's frame: the game's frame is
// cleared to nothing after it's copied for the glass, so what's read back is the canvas.
#include "check.h"
#include "fake_launcher.h"
#include "fake_shell.h"
#include "vk_gpu.h"

#include <algorithm>
#include <fstream>
#include <iterator>

namespace {

constexpr uint32_t W = 320;
constexpr uint32_t H = 200;

std::vector<uint8_t> read_file(const char* path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// The shell's three slots
struct Slots {
	static constexpr size_t SIZE = 1 << 20;
	int                     fds[3] {};
	uint8_t*                maps[3] {};

	Slots() {
		for (int i = 0; i < 3; i++) {
			fds[i]  = make_memfd(SIZE, true);
			maps[i] = static_cast<uint8_t*>(::mmap(nullptr, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fds[i], 0));
		}
	}
	~Slots() {
		for (auto* m: maps) {
			::munmap(m, SIZE);
		}
	}
	std::vector<int> give() const { return {fds[0], fds[1], fds[2]}; }

	void write(int i, std::vector<uint8_t> bytes, uint64_t serial) {
		if (bytes.size() >= 48 && bytes.size() <= SIZE) {
			std::memcpy(bytes.data() + 40, &serial, 8);
			std::memcpy(maps[i], bytes.data(), bytes.size());
		}
	}
};

struct Owned {
	VkImage        image  = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkBuffer       buffer = VK_NULL_HANDLE;
	void*          mapped = nullptr;
};

Owned host_buffer(Gpu& gpu, VkDeviceSize size, VkBufferUsageFlags usage) {
	Owned              out;
	VkBufferCreateInfo bci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size  = size;
	bci.usage = usage;
	gpu.CreateBuffer(gpu.device, &bci, nullptr, &out.buffer);
	VkMemoryRequirements req {};
	gpu.GetBufferMemoryRequirements(gpu.device, out.buffer, &req);
	VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = gpu.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	gpu.AllocateMemory(gpu.device, &alloc, nullptr, &out.memory);
	gpu.BindBufferMemory(gpu.device, out.buffer, out.memory, 0);
	gpu.MapMemory(gpu.device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped);
	return out;
}

// How far apart two canvases are, as crates/render/tests/list.rs measures it: the worst channel,
// and the pixels with any channel more than 4 apart
std::pair<int, size_t> difference(const uint8_t* a, const uint8_t* b, size_t pixels) {
	int    worst = 0;
	size_t off   = 0;
	for (size_t i = 0; i < pixels; i++) {
		bool far = false;
		for (size_t c = 0; c < 4; c++) {
			const int d = std::abs(int(a[i * 4 + c]) - int(b[i * 4 + c]));
			worst       = std::max(worst, d);
			far         = far || d > 4;
		}
		off += far ? 1 : 0;
	}
	return {worst, off};
}

void run(Gpu& gpu) {
	const auto scene = read_file(LC_TEST_DATA_DIR "/scene.ldl");
	const auto want  = read_file(LC_TEST_DATA_DIR "/scene.rgba");
	CHECK(!scene.empty() && want.size() == W * H * 4);

	Slots        slots;
	FakeShell    shell(slots.give());
	FakeLauncher launcher;
	lc_callbacks cb {};
	cb.log = [](void*, lc_log_level, const char* line) {
		if (std::getenv("LC_TEST_LOG") != nullptr) {
			std::fprintf(stderr, "log: %s\n", line);
		}
	};
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY | LC_CAP_DRAW_LISTS;
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
		// Lists need nothing shared with the shell, so any device draws them
		FAIL("lc_vk_create: %s", error);
		lc_close(s);
		return;
	}
	lc_ready(s, 0, 0);

	// The game's frame, which the menu is drawn over in the game's own render pass
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.imageType   = VK_IMAGE_TYPE_2D;
	ci.format      = VK_FORMAT_R8G8B8A8_UNORM;
	ci.extent      = {W, H, 1};
	ci.mipLevels   = 1;
	ci.arrayLayers = 1;
	ci.samples     = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
	ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	Owned target;
	gpu.CreateImage(gpu.device, &ci, nullptr, &target.image);
	VkMemoryRequirements req {};
	gpu.GetImageMemoryRequirements(gpu.device, target.image, &req);
	VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = gpu.memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	gpu.AllocateMemory(gpu.device, &alloc, nullptr, &target.memory);
	gpu.BindImageMemory(gpu.device, target.image, target.memory, 0);
	VkImageViewCreateInfo vci {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vci.image            = target.image;
	vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
	vci.format           = ci.format;
	vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	VkImageView target_view = VK_NULL_HANDLE;
	gpu.CreateImageView(gpu.device, &vci, nullptr, &target_view);

	VkAttachmentDescription attachment {};
	attachment.format        = ci.format;
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
	// The frame is read back after it
	VkSubpassDependency after {};
	after.srcSubpass    = 0;
	after.dstSubpass    = VK_SUBPASS_EXTERNAL;
	after.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	after.dstStageMask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
	after.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	after.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	VkRenderPassCreateInfo rci {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
	rci.attachmentCount = 1;
	rci.pAttachments    = &attachment;
	rci.subpassCount    = 1;
	rci.pSubpasses      = &subpass;
	rci.dependencyCount = 1;
	rci.pDependencies   = &after;
	VkRenderPass pass   = VK_NULL_HANDLE;
	gpu.CreateRenderPass(gpu.device, &rci, nullptr, &pass);
	VkFramebufferCreateInfo fci {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
	fci.renderPass      = pass;
	fci.attachmentCount = 1;
	fci.pAttachments    = &target_view;
	fci.width           = W;
	fci.height          = H;
	fci.layers          = 1;
	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	gpu.CreateFramebuffer(gpu.device, &fci, nullptr, &framebuffer);

	// The game's picture: red on the left half, blue on the right (as the renderer's test's live
	// frame), or `green` all over
	Owned      frame_pixels = host_buffer(gpu, W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	auto*      px           = static_cast<uint8_t*>(frame_pixels.mapped);
	const auto paint        = [&](bool green) {
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                const bool    left    = x < W / 2 && !green;
                const uint8_t rgba[4] = {static_cast<uint8_t>(left ? 255 : 0), static_cast<uint8_t>(green ? 255 : 0),
                                         static_cast<uint8_t>(left || green ? 0 : 255), 255};
                std::memcpy(px + (y * W + x) * 4, rgba, 4);
            }
        }
	};
	paint(false);
	Owned       readback = host_buffer(gpu, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	const auto* got      = static_cast<const uint8_t*>(readback.mapped);

	launcher.send(R"({"type":"overlay_opened"})");
	launcher.send(R"({"type":"overlay_surface","buffers":[11,12,13],"header":10,"service":")" + shell.name() +
	              R"(","width":320,"height":200,"lists":true})");
	launcher.send(R"({"type":"overlay_visible","visible":true})");

	uint64_t   completed = 0;
	uint64_t   serial    = 0;
	bool       drew      = false;
	bool       captured  = false;
	bool       no_frame  = false; // Captures pass VK_NULL_HANDLE, as for a frame that can't be copied
	const auto present   = [&](bool new_frame = true, bool copies = true) {
        serial++;
        lc_vk_frame frame {};
        lc_vk_begin(vk, completed, {W, H}, new_frame, &frame);
        gpu.begin();
        const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        gpu.barrier(target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy copy {};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent      = {W, H, 1};
        gpu.CmdCopyBufferToImage(gpu.command, frame_pixels.buffer, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        gpu.barrier(target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
        if (frame.capture && copies) {
            lc_vk_capture(vk, gpu.command, no_frame ? VK_NULL_HANDLE : target.image, VK_FORMAT_R8G8B8A8_UNORM, {W, H});
            captured = true;
        }
        // Nothing under the menu, so what's read back is what the helper drew
        gpu.barrier(target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue none {};
        gpu.CmdClearColorImage(gpu.command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &none, 1, &range);
        gpu.barrier(target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        const VkImageView menu = lc_vk_acquire(vk, gpu.command);
        VkRenderPassBeginInfo rbi {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass  = pass;
        rbi.framebuffer = framebuffer;
        rbi.renderArea  = {{0, 0}, {W, H}};
        gpu.CmdBeginRenderPass(gpu.command, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        if (menu != VK_NULL_HANDLE) {
            lc_vk_draw(vk, gpu.command, pass, 0, VK_FORMAT_R8G8B8A8_UNORM);
            drew = true;
        }
        gpu.CmdEndRenderPass(gpu.command);
        gpu.CmdCopyImageToBuffer(gpu.command, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &copy);
        gpu.submit();
        lc_vk_end(vk, serial);
        completed = serial;
	};

	// Nothing to draw until the shell publishes a list
	present();
	CHECK(!drew);
	slots.write(0, scene, 1);
	shell.publish(0);
	for (int i = 0; i < 50 && !drew; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		present();
	}
	CHECK(drew);
	CHECK(captured);
	CHECK(launcher.wait_for("view").has_value());

	const auto check_drawn = [&](const char* what) {
		const auto [worst, off] = difference(got, want.data(), W * H);
		std::printf("vk_list: %s: worst channel %d apart, %zu pixels more than 4 apart\n", what, worst, off);
		if (worst > 48 || off * 100 > W * H) {
			FAIL("%s isn't drawn as the shell draws it (worst %d, %zu pixels off)", what, worst, off);
			if (const char* dump = std::getenv("LC_TEST_DUMP")) {
				std::ofstream(dump, std::ios::binary).write(reinterpret_cast<const char*>(got), W * H * 4);
			}
		}
	};
	check_drawn("the menu");

	// Presents with nothing new draw the same, from the canvas kept, its glass from each new copy
	for (int i = 0; i < 3; i++) {
		present();
	}
	check_drawn("the menu, presents later");

	// The GPU time it took is in the header for the shell's frame timings: copying and blurring the
	// frame, drawing the canvas, and laying it over the frame, each measured by a present that did it
	{
		float          ms[4] {};
		const uint32_t count = shell.gpu_times(ms);
		std::printf("vk_list: GPU times written %u times; last: render %.3f, glass %.3f, draw %.3f, composite %.3f ms\n", count,
		            static_cast<double>(ms[0]), static_cast<double>(ms[1]), static_cast<double>(ms[2]), static_cast<double>(ms[3]));
		if (gpu.timestamps) {
			CHECK(count > 0);
			for (float t: ms) {
				CHECK(t > 0 && t < 1000);
			}
			CHECK(ms[0] >= ms[1] && ms[0] >= ms[2]);
		}
	}

	// The same list again, as a shell carrying changes the game may have missed sends them: a
	// repeat changes nothing
	slots.write(1, scene, 2);
	CHECK(shell.publish(1) != 1);
	present();
	present();
	check_drawn("the menu, sent again");

	// A paused game presents the frame it paused on again: it's copied and blurred once, not
	// every present. So a frame that changes without the game saying so isn't copied...
	captured = false;
	paint(true);
	present(false);
	present(false);
	CHECK(!captured);
	check_drawn("the menu, paused");
	// ...and a new one is, its glass now green
	present(true);
	CHECK(captured);
	{
		const auto [worst, off] = difference(got, want.data(), W * H);
		std::printf("vk_list: a new frame: worst channel %d apart, %zu pixels more than 4 apart\n", worst, off);
		CHECK(worst > 48 && off * 10 > W * H);
	}
	// Reopened, the menu copies the frame it opens over even if the game says it's the same one
	captured = false;
	launcher.send(R"({"type":"overlay_visible","visible":false})");
	for (int i = 0; i < 50 && drew; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		drew = false;
		present(false);
	}
	CHECK(!drew);
	paint(false);
	launcher.send(R"({"type":"overlay_visible","visible":true})");
	for (int i = 0; i < 50 && !drew; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		present(false);
	}
	CHECK(drew && captured);
	check_drawn("the menu, reopened");

	// A copy asked for and not made (the emulator skipped it) is asked for again, until one is
	launcher.send(R"({"type":"overlay_visible","visible":false})");
	for (int i = 0; i < 50 && drew; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		drew = false;
		present(false);
	}
	paint(true);
	launcher.send(R"({"type":"overlay_visible","visible":true})");
	for (int i = 0; i < 50 && !drew; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		present(false, false);
	}
	CHECK(drew);
	captured = false;
	present(false);
	CHECK(captured);
	{
		const auto [worst, off] = difference(got, want.data(), W * H);
		CHECK(worst > 48 && off * 10 > W * H); // Its glass is the green frame, not the one before
	}

	// A frame the emulator can't copy (VK_NULL_HANDLE): the menu isn't drawn in the game, and the
	// shell comes to the front for it
	no_frame = true;
	captured = false;
	drew     = false;
	present(true);
	CHECK(captured && !drew);
	drew = false;
	present(true);
	present(true);
	CHECK(!drew);

	gpu.DeviceWaitIdle(gpu.device);
	lc_vk_destroy(vk);
	lc_close(s);
	gpu.DestroyFramebuffer(gpu.device, framebuffer, nullptr);
	gpu.DestroyRenderPass(gpu.device, pass, nullptr);
	gpu.DestroyImageView(gpu.device, target_view, nullptr);
	gpu.DestroyImage(gpu.device, target.image, nullptr);
	gpu.FreeMemory(gpu.device, target.memory, nullptr);
	for (auto* b: {&frame_pixels, &readback}) {
		gpu.DestroyBuffer(gpu.device, b->buffer, nullptr);
		gpu.FreeMemory(gpu.device, b->memory, nullptr);
	}
}

} // namespace

int main() {
	Gpu               gpu;
	std::string skip = gpu.open(false);
	if (skip.empty() && !gpu.bc) {
		skip = "no BC textures, which the scene's art is in";
	}
	if (!skip.empty()) {
		std::printf("vk_list: skipped (%s)\n", skip.c_str());
		gpu.close();
		return SKIPPED;
	}
	run(gpu);
	gpu.close();
	return finish("vk_list");
}
