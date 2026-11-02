// The draw-list player (list_vk.h): crates/render's Renderer::play, in Vulkan, with the same
// shaders (list_vk_shaders.h, SPIR-V made from the shell's WGSL).
//
// What a list draws with travels with it or before it, and is kept here: the glyph atlas, built
// from each list's updates; the images, added and dropped by key; and their blurs, made once. The
// game's own frame stands in for the picture the shell would have drawn the menu over (BgSource
// Live): lc_vk_capture copies it, and its blur is made again whenever it's copied.
//
// A list is drawn into a canvas of its size, kept from present to present, which lc_vk_draw lays
// over the game's frame as it does a shell's buffer: the canvas holds sRGB-encoded premultiplied
// values like the shell's own, whatever the game's swapchain format. It's drawn again only when
// something it shows changed: a new list, new pixels, or a new copy of the game's frame.
//
// GPU lifetimes are tracked with the presents' serials: anything replaced is freed once the
// present that last could have used it has finished.
#include "list_vk.h"

#include "internal.h"
#include "list_vk_shaders.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace lc::vk {
namespace {

constexpr VkFormat     CANVAS_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat     BLUR_FORMAT   = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkDeviceSize GLOBALS_BYTES = 16;
constexpr VkDeviceSize BG_BYTES      = LC_LIST_BG_FLOATS * 4;
// Each blur step's parameters (blur.wgsl's Params) at its own offset, which any device can bind
constexpr VkDeviceSize PARAMS_STRIDE = 256;
// Host buffers kept for reuse once the GPU is done with them
constexpr size_t SPARE_BUFFERS = 4;
// The pooled host buffers' usage: staging for uploads, and instances
constexpr VkBufferUsageFlags POOLED = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

struct Buffer {
	VkBuffer       buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	uint8_t*       mapped = nullptr; // Host buffers'
	VkDeviceSize   size   = 0;
};

// An image of the player's own, sampled, and drawn into if it has a framebuffer
struct Image {
	VkImage        image       = VK_NULL_HANDLE;
	VkDeviceMemory memory      = VK_NULL_HANDLE;
	VkImageView    view        = VK_NULL_HANDLE;
	VkFramebuffer  framebuffer = VK_NULL_HANDLE;
	VkFormat       format      = VK_FORMAT_UNDEFINED;
	uint32_t       width       = 0;
	uint32_t       height      = 0;
	uint32_t       levels      = 1;
	VkImageLayout  layout      = VK_IMAGE_LAYOUT_UNDEFINED;
};

// A picture blurred for glass (blur.wgsl) at each of two strengths: half-size levels down from
// it, then back up, per strength, to a quarter of its size (crates/render's Blur::blur_image)
struct Chain {
	struct Step {
		bool            down;
		uint32_t        target;
		VkDescriptorSet set;
	};
	static constexpr uint32_t NONE = ~0u;
	uint32_t           levels[2] {};
	VkImageView        source = VK_NULL_HANDLE;
	std::vector<Image> targets; // The levels down, then each strength's own result
	Buffer             params;
	VkDescriptorPool   pool = VK_NULL_HANDLE;
	std::vector<Step>  steps;
	uint32_t           results[2] {NONE, NONE};

	// The picture blurred at strength `n`, if that strength was made
	VkImageView result(int n) const { return results[n] == NONE ? VK_NULL_HANDLE : targets[results[n]].view; }
};

// A picture lists draw by key, the glyph atlas, or one of the player's stand-ins
struct Texture {
	Image                  image;
	VkDescriptorPool       pool = VK_NULL_HANDLE; // For `set`
	VkDescriptorSet        set  = VK_NULL_HANDLE; // Bound as a draw's image (set 1)
	uint64_t               generation = 0;        // The atlas's
	std::unique_ptr<Chain> blur;
};

// Writes into a texture, recorded at the next prepare
struct Upload {
	Texture*                       target = nullptr;
	bool                           clear  = false; // Cleared to `color` first
	VkClearColorValue              color {};
	VkBuffer                       source = VK_NULL_HANDLE;
	std::vector<VkBufferImageCopy> copies;
};

struct Garbage {
	uint64_t              serial; // 0: until the present that's being recorded is submitted
	std::function<void()> free;
};

struct PassRecord {
	VkRect2D scissor;
	uint32_t first_draw;
	uint32_t draw_count;
};

// GPU time for the menu, from timestamps around each part of a present (GpuTimes), read back once
// the GPU has finished it without waiting. Presents whose timestamps aren't read yet hold a slot
// each; a present with none free goes untimed.
class Timer {
public:
	bool create(const Fns& vk, const lc_vk_info& info) {
		uint32_t count = 0;
		vk.GetPhysicalDeviceQueueFamilyProperties(info.physical_device, &count, nullptr);
		std::vector<VkQueueFamilyProperties> families(count);
		vk.GetPhysicalDeviceQueueFamilyProperties(info.physical_device, &count, families.data());
		VkPhysicalDeviceProperties props {};
		vk.GetPhysicalDeviceProperties(info.physical_device, &props);
		const uint32_t bits = info.queue_family < count ? families[info.queue_family].timestampValidBits : 0;
		if (bits == 0 || props.limits.timestampPeriod <= 0) {
			return false; // Untimed, which is fine
		}
		m_mask   = bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
		m_period = props.limits.timestampPeriod;
		VkQueryPoolCreateInfo qci {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
		qci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
		qci.queryCount = SLOTS * MARKS;
		return vk.CreateQueryPool(info.device, &qci, nullptr, &m_pool) == VK_SUCCESS;
	}

	void destroy(const Fns& vk, VkDevice device) {
		if (m_pool != VK_NULL_HANDLE) {
			vk.DestroyQueryPool(device, m_pool, nullptr);
			m_pool = VK_NULL_HANDLE;
		}
	}

	// Reads what finished presents measured, and starts this one
	void begin(const Fns& vk, VkDevice device, uint64_t completed) {
		if (m_pool == VK_NULL_HANDLE) {
			return;
		}
		for (uint32_t n = 0; n < SLOTS; n++) {
			Slot& slot = m_slots[n];
			// One still unreadable a while after its present finished was never submitted (a
			// present the emulator dropped): given up on, so it doesn't hold its slot for good
			if (slot.serial != 0 && slot.serial <= completed && (read(vk, device, n) || ++slot.tries >= GIVE_UP)) {
				slot = {};
			}
		}
		m_open    = m_slots[m_current].serial == 0;
		m_reset   = false;
		m_written = 0;
	}

	// Outside any render pass unless something was stamped before it this present
	void stamp(const Fns& vk, VkCommandBuffer command, Mark mark, bool in_pass) {
		if (!m_open) {
			return;
		}
		if (!m_reset) {
			if (in_pass) {
				return;
			}
			vk.CmdResetQueryPool(command, m_pool, m_current * MARKS, MARKS);
			m_reset = true;
		}
		// Starts too: at the top of the pipe a start would be written while the game's own work
		// before it still runs, and its span would include that work's tail
		vk.CmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool, m_current * MARKS + static_cast<uint32_t>(mark));
		m_written |= 1u << static_cast<uint32_t>(mark);
	}

	void end(uint64_t serial) {
		if (m_open && m_written != 0) {
			m_slots[m_current] = {serial, m_written, 0};
			m_current          = (m_current + 1) % SLOTS;
		}
		m_open = false;
	}

	// The latest times, if they changed since the last call
	bool times(GpuTimes& out) {
		if (!m_fresh) {
			return false;
		}
		m_fresh = false;
		out     = m_times;
		return true;
	}

private:
	static constexpr uint32_t SLOTS   = 4;
	static constexpr uint32_t MARKS   = 6;
	static constexpr uint32_t GIVE_UP = 8; // Presents

	struct Slot {
		uint64_t serial  = 0;
		uint32_t written = 0;
		uint32_t tries   = 0;
	};

	// False if a timestamp isn't there yet
	bool read(const Fns& vk, VkDevice device, uint32_t n) {
		const Slot& slot = m_slots[n];
		uint64_t    at[MARKS] {};
		for (uint32_t m = 0; m < MARKS; m++) {
			if ((slot.written & (1u << m)) != 0 &&
			    vk.GetQueryPoolResults(device, m_pool, n * MARKS + m, 1, sizeof(uint64_t), &at[m], sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) !=
			        VK_SUCCESS) {
				return false;
			}
		}
		const auto has  = [&](Mark from) { return (slot.written & (3u << static_cast<uint32_t>(from))) == (3u << static_cast<uint32_t>(from)); };
		const auto span = [&](uint32_t from, uint32_t to) {
			return static_cast<float>(static_cast<double>((at[to] - at[from]) & m_mask) * m_period / 1e6);
		};
		const bool glass = has(Mark::GlassStart), draw = has(Mark::DrawStart), composite = has(Mark::CompositeStart);
		if (glass) {
			m_times.glass = span(0, 1);
		}
		if (draw) {
			m_times.draw = span(2, 3);
		}
		if (composite) {
			m_times.composite = span(4, 5);
		}
		if (glass || draw) {
			// From the first part's start to the last one's end, as crates/render's `render`
			m_times.render = span(glass ? 0 : 2, draw ? 3 : 1);
		}
		m_fresh = m_fresh || glass || draw || composite;
		return true;
	}

	VkQueryPool m_pool   = VK_NULL_HANDLE;
	uint64_t    m_mask   = 0;
	float       m_period = 0; // Nanoseconds a tick
	Slot        m_slots[SLOTS];
	uint32_t    m_current = 0;
	bool        m_open    = false; // This present can be timed
	bool        m_reset   = false;
	uint32_t    m_written = 0;
	GpuTimes    m_times {};
	bool        m_fresh = false;
};

} // namespace

struct ListPlayer::State {
	Fns                              vk;
	lc_vk_info                       info {};
	Log                              log;
	VkPhysicalDeviceMemoryProperties memory {};
	Timer                            timer;
	uint32_t                         max_side = 0;
	bool                             bc7      = false;

	VkDescriptorSetLayout composite_layout  = VK_NULL_HANDLE;
	VkSampler             composite_sampler = VK_NULL_HANDLE;

	// As crates/render's: `linear` for pictures and glyphs, `noise_sampler` tiling glass noise
	VkSampler             linear        = VK_NULL_HANDLE;
	VkSampler             noise_sampler = VK_NULL_HANDLE;
	VkSampler             blur_sampler  = VK_NULL_HANDLE;
	VkDescriptorSetLayout globals_layout = VK_NULL_HANDLE; // set 0
	VkDescriptorSetLayout image_layout   = VK_NULL_HANDLE; // set 1
	VkDescriptorSetLayout bg_layout      = VK_NULL_HANDLE; // set 2
	VkDescriptorSetLayout blur_layout    = VK_NULL_HANDLE;
	VkPipelineLayout      sdf_layout     = VK_NULL_HANDLE;
	VkPipelineLayout      blur_layout_p  = VK_NULL_HANDLE;
	VkShaderModule        sdf_module     = VK_NULL_HANDLE;
	VkShaderModule        blur_module    = VK_NULL_HANDLE;
	VkRenderPass          canvas_pass    = VK_NULL_HANDLE;
	VkRenderPass          blur_pass      = VK_NULL_HANDLE;
	VkPipeline            blend          = VK_NULL_HANDLE;
	VkPipeline            replace        = VK_NULL_HANDLE;
	VkPipeline            down           = VK_NULL_HANDLE;
	VkPipeline            up             = VK_NULL_HANDLE;
	Buffer                globals_ubo;
	Buffer                bg_ubo;

	Texture blank;       // White, for draws without an image
	Texture noise;       // Glass noise
	Texture no_glyphs;   // Until a list brings an atlas
	Texture placeholder; // For a missing picture or blur

	std::unique_ptr<Texture>                               atlas;
	std::unordered_map<uint64_t, std::unique_ptr<Texture>> images;
	std::vector<Upload>                                    uploads;
	std::vector<Buffer>                                    staged; // Uploads' sources, until recorded
	std::vector<Garbage>                                   garbage;
	std::vector<Buffer>                                    spare;

	// The list taken last, as much of it as drawing it again needs
	uint32_t                  generation = 0;
	bool                      taken      = false;
	bool                      showing    = false; // This present has a list
	uint64_t                  serial     = 0;
	uint32_t                  width      = 0;
	uint32_t                  height     = 0;
	uint32_t                  flags      = 0;
	float                     globals[4] {};
	float                     bg[LC_LIST_BG_FLOATS] {};
	lc_list_source            sources[2] {};
	std::vector<lc_list_blur> blurs;
	std::vector<PassRecord>   passes;
	std::vector<lc_list_draw> draws;
	Buffer                    instances;

	// The game's frame, for glass
	Image                  live;
	std::unique_ptr<Chain> live_blur;
	bool                   live_fresh = false; // Copied this present
	bool                   copy       = false; // This present's frame is to be copied
	bool                   stale      = true;  // The copy isn't of the frame the game shows

	Image            canvas;
	VkDescriptorPool canvas_pool = VK_NULL_HANDLE;
	VkDescriptorSet  canvas_set  = VK_NULL_HANDLE;
	bool             dirty       = false;

	// Sets 0 and 2, kept while what they point at stays
	VkDescriptorPool globals_pool = VK_NULL_HANDLE;
	VkDescriptorSet  globals_set  = VK_NULL_HANDLE;
	VkImageView      globals_atlas = VK_NULL_HANDLE;
	VkDescriptorPool bg_pool      = VK_NULL_HANDLE;
	VkDescriptorSet  bg_set       = VK_NULL_HANDLE;
	VkImageView      bg_views[6] {};

	void say(lc_log_level level, const std::string& line) const {
		if (log) {
			log(level, "overlay: " + line);
		}
	}

	// ---- Memory and lifetimes ----

	void collect(uint64_t completed, bool all) {
		std::vector<Garbage> keep;
		std::vector<Garbage> done;
		for (auto& g: garbage) {
			if (all || (g.serial != 0 && g.serial <= completed)) {
				done.push_back(std::move(g));
			} else {
				keep.push_back(std::move(g));
			}
		}
		garbage = std::move(keep);
		for (auto& g: done) {
			g.free();
		}
	}

	void later(std::function<void()> free) { garbage.push_back({0, std::move(free)}); }

	bool make_buffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
		VkBufferCreateInfo ci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		ci.size        = size;
		ci.usage       = usage;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vk.CreateBuffer(info.device, &ci, nullptr, &b.buffer) != VK_SUCCESS) {
			b.buffer = VK_NULL_HANDLE;
			return false;
		}
		VkMemoryRequirements req {};
		vk.GetBufferMemoryRequirements(info.device, b.buffer, &req);
		VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		alloc.allocationSize  = req.size;
		alloc.memoryTypeIndex = memory_type(memory, req.memoryTypeBits,
		                                    host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		                                         : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (host && (memory.memoryTypes[alloc.memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
			free_buffer(b);
			return false;
		}
		if (vk.AllocateMemory(info.device, &alloc, nullptr, &b.memory) != VK_SUCCESS) {
			b.memory = VK_NULL_HANDLE;
			free_buffer(b);
			return false;
		}
		void* mapped = nullptr;
		if (vk.BindBufferMemory(info.device, b.buffer, b.memory, 0) != VK_SUCCESS ||
		    (host && vk.MapMemory(info.device, b.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)) {
			free_buffer(b);
			return false;
		}
		b.mapped = static_cast<uint8_t*>(mapped);
		b.size   = size;
		return true;
	}

	void free_buffer(Buffer& b) {
		if (b.buffer != VK_NULL_HANDLE) {
			vk.DestroyBuffer(info.device, b.buffer, nullptr);
		}
		if (b.memory != VK_NULL_HANDLE) {
			vk.FreeMemory(info.device, b.memory, nullptr); // Unmaps it
		}
		b = {};
	}

	// A host buffer of at least `size` bytes: a spare one, or a new one
	bool take(Buffer& out, VkDeviceSize size) {
		auto best = spare.end();
		for (auto it = spare.begin(); it != spare.end(); ++it) {
			if (it->size >= size && (best == spare.end() || it->size < best->size)) {
				best = it;
			}
		}
		if (best != spare.end()) {
			out = *best;
			spare.erase(best);
			return true;
		}
		VkDeviceSize want = 64 * 1024;
		while (want < size) {
			want *= 2;
		}
		return make_buffer(out, want, POOLED, true);
	}

	// Back to the spares once the GPU is done with it
	void recycle(Buffer& b) {
		if (b.buffer == VK_NULL_HANDLE) {
			return;
		}
		later([this, b]() mutable {
			spare.push_back(b);
			if (spare.size() > SPARE_BUFFERS) {
				auto smallest = std::min_element(spare.begin(), spare.end(), [](const Buffer& x, const Buffer& y) { return x.size < y.size; });
				free_buffer(*smallest);
				spare.erase(smallest);
			}
		});
		b = {};
	}

	// `view_format`: another format to view it as (an sRGB image read as UNORM); `pass`: one it's
	// drawn into with
	bool make_image(Image& img, VkFormat format, uint32_t width, uint32_t height, uint32_t levels, VkImageUsageFlags usage,
	                VkFormat view_format = VK_FORMAT_UNDEFINED, VkRenderPass pass = VK_NULL_HANDLE) {
		VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		ci.flags         = view_format != VK_FORMAT_UNDEFINED ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
		ci.imageType     = VK_IMAGE_TYPE_2D;
		ci.format        = format;
		ci.extent        = {width, height, 1};
		ci.mipLevels     = levels;
		ci.arrayLayers   = 1;
		ci.samples       = VK_SAMPLE_COUNT_1_BIT;
		ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
		ci.usage         = usage;
		ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (vk.CreateImage(info.device, &ci, nullptr, &img.image) != VK_SUCCESS) {
			img.image = VK_NULL_HANDLE;
			return false;
		}
		img.format = format;
		img.width  = width;
		img.height = height;
		img.levels = levels;
		img.layout = VK_IMAGE_LAYOUT_UNDEFINED;
		VkMemoryRequirements req {};
		vk.GetImageMemoryRequirements(info.device, img.image, &req);
		VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		alloc.allocationSize  = req.size;
		alloc.memoryTypeIndex = memory_type(memory, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vk.AllocateMemory(info.device, &alloc, nullptr, &img.memory) != VK_SUCCESS) {
			img.memory = VK_NULL_HANDLE;
			free_image(img);
			return false;
		}
		VkImageViewCreateInfo vci {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		vci.image            = img.image;
		vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
		vci.format           = view_format != VK_FORMAT_UNDEFINED ? view_format : format;
		vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1};
		if (vk.BindImageMemory(info.device, img.image, img.memory, 0) != VK_SUCCESS ||
		    vk.CreateImageView(info.device, &vci, nullptr, &img.view) != VK_SUCCESS) {
			img.view = VK_NULL_HANDLE;
			free_image(img);
			return false;
		}
		if (pass != VK_NULL_HANDLE) {
			VkFramebufferCreateInfo fci {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
			fci.renderPass      = pass;
			fci.attachmentCount = 1;
			fci.pAttachments    = &img.view;
			fci.width           = width;
			fci.height          = height;
			fci.layers          = 1;
			if (vk.CreateFramebuffer(info.device, &fci, nullptr, &img.framebuffer) != VK_SUCCESS) {
				img.framebuffer = VK_NULL_HANDLE;
				free_image(img);
				return false;
			}
		}
		return true;
	}

	void free_image(Image& img) {
		if (img.framebuffer != VK_NULL_HANDLE) {
			vk.DestroyFramebuffer(info.device, img.framebuffer, nullptr);
		}
		if (img.view != VK_NULL_HANDLE) {
			vk.DestroyImageView(info.device, img.view, nullptr);
		}
		if (img.image != VK_NULL_HANDLE) {
			vk.DestroyImage(info.device, img.image, nullptr);
		}
		if (img.memory != VK_NULL_HANDLE) {
			vk.FreeMemory(info.device, img.memory, nullptr);
		}
		img = {};
	}

	void free_chain(Chain& c) {
		for (auto& t: c.targets) {
			free_image(t);
		}
		free_buffer(c.params);
		if (c.pool != VK_NULL_HANDLE) {
			vk.DestroyDescriptorPool(info.device, c.pool, nullptr);
		}
		c = {};
	}

	void free_texture(Texture& t) {
		free_image(t.image);
		if (t.pool != VK_NULL_HANDLE) {
			vk.DestroyDescriptorPool(info.device, t.pool, nullptr);
		}
		if (t.blur) {
			free_chain(*t.blur);
		}
		t = {};
	}

	// The sets that may point at something just retired are made again when next drawn
	void forget_sets() {
		for (VkDescriptorPool* pool: {&globals_pool, &bg_pool}) {
			if (*pool != VK_NULL_HANDLE) {
				later([this, p = *pool]() { vk.DestroyDescriptorPool(info.device, p, nullptr); });
				*pool = VK_NULL_HANDLE;
			}
		}
		globals_set   = VK_NULL_HANDLE;
		globals_atlas = VK_NULL_HANDLE;
		bg_set        = VK_NULL_HANDLE;
		std::fill(std::begin(bg_views), std::end(bg_views), VK_NULL_HANDLE);
		dirty = true;
	}

	void retire(std::unique_ptr<Texture>& t) {
		if (!t) {
			return;
		}
		Texture* gone = t.get();
		uploads.erase(std::remove_if(uploads.begin(), uploads.end(), [&](const Upload& u) { return u.target == gone; }), uploads.end());
		std::shared_ptr<Texture> owned(t.release());
		later([this, owned]() { free_texture(*owned); });
		forget_sets();
	}

	void retire(std::unique_ptr<Chain>& c) {
		if (!c) {
			return;
		}
		std::shared_ptr<Chain> owned(c.release());
		later([this, owned]() { free_chain(*owned); });
		forget_sets();
	}

	void retire(Image& img) {
		if (img.image == VK_NULL_HANDLE) {
			return;
		}
		later([this, img]() mutable { free_image(img); });
		img = {};
		forget_sets();
	}

	// ---- Descriptor sets ----

	// A set of `layout` from a pool of its own
	VkDescriptorSet allocate(VkDescriptorSetLayout layout, std::initializer_list<VkDescriptorPoolSize> sizes, VkDescriptorPool& pool,
	                         uint32_t count = 1, VkDescriptorSet* many = nullptr) {
		std::vector<VkDescriptorPoolSize> scaled(sizes);
		for (auto& s: scaled) {
			s.descriptorCount *= count;
		}
		VkDescriptorPoolCreateInfo ci {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		ci.maxSets       = count;
		ci.poolSizeCount = static_cast<uint32_t>(scaled.size());
		ci.pPoolSizes    = scaled.data();
		if (vk.CreateDescriptorPool(info.device, &ci, nullptr, &pool) != VK_SUCCESS) {
			pool = VK_NULL_HANDLE;
			return VK_NULL_HANDLE;
		}
		std::vector<VkDescriptorSetLayout> layouts(count, layout);
		std::vector<VkDescriptorSet>       sets(count);
		VkDescriptorSetAllocateInfo        alloc {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		alloc.descriptorPool     = pool;
		alloc.descriptorSetCount = count;
		alloc.pSetLayouts        = layouts.data();
		if (vk.AllocateDescriptorSets(info.device, &alloc, sets.data()) != VK_SUCCESS) {
			return VK_NULL_HANDLE;
		}
		if (many != nullptr) {
			std::copy(sets.begin(), sets.end(), many);
		}
		return sets[0];
	}

	static VkWriteDescriptorSet image_write(VkDescriptorSet set, uint32_t binding, const VkDescriptorImageInfo* image, VkDescriptorType type) {
		VkWriteDescriptorSet w {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		w.dstSet          = set;
		w.dstBinding      = binding;
		w.descriptorCount = 1;
		w.descriptorType  = type;
		w.pImageInfo      = image;
		return w;
	}

	static VkWriteDescriptorSet buffer_write(VkDescriptorSet set, uint32_t binding, const VkDescriptorBufferInfo* buffer) {
		VkWriteDescriptorSet w {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		w.dstSet          = set;
		w.dstBinding      = binding;
		w.descriptorCount = 1;
		w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		w.pBufferInfo     = buffer;
		return w;
	}

	static VkDescriptorImageInfo sampled(VkImageView view) { return {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; }

	// A texture's set 1, for draws that bind it
	bool give_set(Texture& t) {
		t.set = allocate(image_layout, {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}}, t.pool);
		if (t.set == VK_NULL_HANDLE) {
			return false;
		}
		const auto image = sampled(t.image.view);
		const auto write = image_write(t.set, 0, &image, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
		vk.UpdateDescriptorSets(info.device, 1, &write, 0, nullptr);
		return true;
	}

	// Set 0: the globals, the glyph atlas and glass noise
	VkDescriptorSet globals_for(VkImageView atlas_view) {
		if (globals_set != VK_NULL_HANDLE && globals_atlas == atlas_view) {
			return globals_set;
		}
		if (globals_pool != VK_NULL_HANDLE) {
			later([this, p = globals_pool]() { vk.DestroyDescriptorPool(info.device, p, nullptr); });
			globals_pool = VK_NULL_HANDLE;
		}
		globals_set = allocate(globals_layout,
		                       {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2}},
		                       globals_pool);
		if (globals_set == VK_NULL_HANDLE) {
			return VK_NULL_HANDLE;
		}
		globals_atlas = atlas_view;
		const VkDescriptorBufferInfo ubo {globals_ubo.buffer, 0, GLOBALS_BYTES};
		const auto                   glyphs = sampled(atlas_view);
		const VkDescriptorImageInfo  samp {linear, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
		const auto                   noise_view = sampled(noise.image.view);
		const VkDescriptorImageInfo  noise_samp {noise_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
		const VkWriteDescriptorSet   writes[] = {
            buffer_write(globals_set, 0, &ubo),
            image_write(globals_set, 1, &glyphs, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            image_write(globals_set, 2, &samp, VK_DESCRIPTOR_TYPE_SAMPLER),
            image_write(globals_set, 3, &noise_view, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            image_write(globals_set, 4, &noise_samp, VK_DESCRIPTOR_TYPE_SAMPLER),
        };
		vk.UpdateDescriptorSets(info.device, 5, writes, 0, nullptr);
		return globals_set;
	}

	// Set 2: the background's two pictures, their blurs, and its uniform
	VkDescriptorSet bg_for(const VkImageView (&views)[6]) {
		if (bg_set != VK_NULL_HANDLE && std::equal(std::begin(views), std::end(views), std::begin(bg_views))) {
			return bg_set;
		}
		if (bg_pool != VK_NULL_HANDLE) {
			later([this, p = bg_pool]() { vk.DestroyDescriptorPool(info.device, p, nullptr); });
			bg_pool = VK_NULL_HANDLE;
		}
		bg_set = allocate(bg_layout, {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}}, bg_pool);
		if (bg_set == VK_NULL_HANDLE) {
			return VK_NULL_HANDLE;
		}
		std::copy(std::begin(views), std::end(views), std::begin(bg_views));
		VkDescriptorImageInfo images[6];
		VkWriteDescriptorSet  writes[7];
		for (uint32_t i = 0; i < 6; i++) {
			images[i] = sampled(views[i]);
			writes[i] = image_write(bg_set, i, &images[i], VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
		}
		const VkDescriptorBufferInfo ubo {bg_ubo.buffer, 0, BG_BYTES};
		writes[6] = buffer_write(bg_set, 6, &ubo);
		vk.UpdateDescriptorSets(info.device, 7, writes, 0, nullptr);
		return bg_set;
	}

	// ---- Set up ----

	bool make_layouts() {
		const auto layout = [&](std::initializer_list<std::pair<VkDescriptorType, VkShaderStageFlags>> bindings, VkDescriptorSetLayout& out) {
			std::vector<VkDescriptorSetLayoutBinding> b;
			for (const auto& [type, stages]: bindings) {
				VkDescriptorSetLayoutBinding one {};
				one.binding         = static_cast<uint32_t>(b.size());
				one.descriptorType  = type;
				one.descriptorCount = 1;
				one.stageFlags      = stages;
				b.push_back(one);
			}
			VkDescriptorSetLayoutCreateInfo ci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
			ci.bindingCount = static_cast<uint32_t>(b.size());
			ci.pBindings    = b.data();
			return vk.CreateDescriptorSetLayout(info.device, &ci, nullptr, &out) == VK_SUCCESS;
		};
		constexpr auto FS   = VK_SHADER_STAGE_FRAGMENT_BIT;
		constexpr auto BOTH = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		constexpr auto UBO  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		constexpr auto TEX  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		constexpr auto SAMP = VK_DESCRIPTOR_TYPE_SAMPLER;
		if (!layout({{UBO, BOTH}, {TEX, FS}, {SAMP, FS}, {TEX, FS}, {SAMP, FS}}, globals_layout) || !layout({{TEX, FS}}, image_layout) ||
		    !layout({{TEX, FS}, {TEX, FS}, {TEX, FS}, {TEX, FS}, {TEX, FS}, {TEX, FS}, {UBO, FS}}, bg_layout) || !layout({{TEX, FS}, {SAMP, FS}, {UBO, FS}}, blur_layout)) {
			return false;
		}
		const VkDescriptorSetLayout sets[] = {globals_layout, image_layout, bg_layout};
		VkPipelineLayoutCreateInfo  ci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		ci.setLayoutCount = 3;
		ci.pSetLayouts    = sets;
		if (vk.CreatePipelineLayout(info.device, &ci, nullptr, &sdf_layout) != VK_SUCCESS) {
			return false;
		}
		ci.setLayoutCount = 1;
		ci.pSetLayouts    = &blur_layout;
		return vk.CreatePipelineLayout(info.device, &ci, nullptr, &blur_layout_p) == VK_SUCCESS;
	}

	bool make_samplers() {
		// wgpu's defaults: clamped, LOD 0 to 32
		const auto sampler = [&](VkSamplerAddressMode address, VkSamplerMipmapMode mip, VkSampler& out) {
			VkSamplerCreateInfo ci {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
			ci.magFilter    = VK_FILTER_LINEAR;
			ci.minFilter    = VK_FILTER_LINEAR;
			ci.mipmapMode   = mip;
			ci.addressModeU = address;
			ci.addressModeV = address;
			ci.addressModeW = address;
			ci.maxLod       = 32.0f;
			return vk.CreateSampler(info.device, &ci, nullptr, &out) == VK_SUCCESS;
		};
		return sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR, linear) &&
		       sampler(VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST, noise_sampler) &&
		       sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST, blur_sampler);
	}

	// Into an image already in COLOR_ATTACHMENT_OPTIMAL; `clear` or load what's there
	VkRenderPass make_pass(VkFormat format, bool clear) {
		VkAttachmentDescription attachment {};
		attachment.format         = format;
		attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp         = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
		attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		VkAttachmentReference ref {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
		VkSubpassDescription  subpass {};
		subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments    = &ref;
		VkRenderPassCreateInfo ci {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
		ci.attachmentCount = 1;
		ci.pAttachments    = &attachment;
		ci.subpassCount    = 1;
		ci.pSubpasses      = &subpass;
		VkRenderPass pass  = VK_NULL_HANDLE;
		return vk.CreateRenderPass(info.device, &ci, nullptr, &pass) == VK_SUCCESS ? pass : VK_NULL_HANDLE;
	}

	VkPipeline make_pipeline(VkShaderModule module, const char* vertex, const char* fragment, VkPipelineLayout layout, VkRenderPass pass,
	                         bool instances, bool blend_on) {
		VkPipelineShaderStageCreateInfo stages[2] {};
		stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = module;
		stages[0].pName  = vertex;
		stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = module;
		stages[1].pName  = fragment;

		// The instance layout (crates/render's Instance, wgpu's vertex_attr_array)
		const VkVertexInputBindingDescription binding {0, LC_LIST_INSTANCE_BYTES, VK_VERTEX_INPUT_RATE_INSTANCE};
		const VkVertexInputAttributeDescription attributes[] = {
		    {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0},   {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16}, {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 32},
		    {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 48},  {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 64}, {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 80},
		    {9, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 96},  {6, 0, VK_FORMAT_R32_SFLOAT, 112},         {7, 0, VK_FORMAT_R32_SFLOAT, 116},
		    {8, 0, VK_FORMAT_R32_UINT, 120},           {10, 0, VK_FORMAT_R32_UINT, 124},
		};
		VkPipelineVertexInputStateCreateInfo vertex_input {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		if (instances) {
			vertex_input.vertexBindingDescriptionCount   = 1;
			vertex_input.pVertexBindingDescriptions      = &binding;
			vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(std::size(attributes));
			vertex_input.pVertexAttributeDescriptions    = attributes;
		}
		VkPipelineInputAssemblyStateCreateInfo assembly {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		assembly.topology = instances ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
		// Premultiplied, as wgpu's PREMULTIPLIED_ALPHA_BLENDING
		VkPipelineColorBlendAttachmentState blend_state {};
		blend_state.blendEnable         = blend_on ? VK_TRUE : VK_FALSE;
		blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blend_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blend_state.colorBlendOp        = VK_BLEND_OP_ADD;
		blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blend_state.alphaBlendOp        = VK_BLEND_OP_ADD;
		blend_state.colorWriteMask =
		    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blending {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		blending.attachmentCount = 1;
		blending.pAttachments    = &blend_state;
		const VkDynamicState             dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamic {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynamic.dynamicStateCount = 2;
		dynamic.pDynamicStates    = dynamic_states;
		VkPipelineDepthStencilStateCreateInfo depth {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

		VkGraphicsPipelineCreateInfo ci {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		ci.stageCount          = 2;
		ci.pStages             = stages;
		ci.pVertexInputState   = &vertex_input;
		ci.pInputAssemblyState = &assembly;
		ci.pViewportState      = &viewport;
		ci.pRasterizationState = &raster;
		ci.pMultisampleState   = &multisample;
		ci.pDepthStencilState  = &depth;
		ci.pColorBlendState    = &blending;
		ci.pDynamicState       = &dynamic;
		ci.layout              = layout;
		ci.renderPass          = pass;
		VkPipeline pipeline    = VK_NULL_HANDLE;
		if (const VkResult r = vk.CreateGraphicsPipelines(info.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline); r != VK_SUCCESS) {
			say(LC_LOG_ERROR, "creating a draw-list pipeline failed: VkResult " + std::to_string(r));
			return VK_NULL_HANDLE;
		}
		return pipeline;
	}

	bool make_pipelines() {
		const auto module = [&](const uint32_t* code, size_t size, VkShaderModule& out) {
			VkShaderModuleCreateInfo ci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
			ci.codeSize = size;
			ci.pCode    = code;
			return vk.CreateShaderModule(info.device, &ci, nullptr, &out) == VK_SUCCESS;
		};
		if (!module(shaders::SDF, sizeof(shaders::SDF), sdf_module) || !module(shaders::BLUR, sizeof(shaders::BLUR), blur_module)) {
			return false;
		}
		canvas_pass = make_pass(CANVAS_FORMAT, false);
		blur_pass   = make_pass(BLUR_FORMAT, true);
		if (canvas_pass == VK_NULL_HANDLE || blur_pass == VK_NULL_HANDLE) {
			return false;
		}
		blend   = make_pipeline(sdf_module, "vs", "fs", sdf_layout, canvas_pass, true, true);
		replace = make_pipeline(sdf_module, "vs", "fs", sdf_layout, canvas_pass, true, false);
		down    = make_pipeline(blur_module, "vs", "down", blur_layout_p, blur_pass, false, false);
		up      = make_pipeline(blur_module, "vs", "up", blur_layout_p, blur_pass, false, false);
		return blend != VK_NULL_HANDLE && replace != VK_NULL_HANDLE && down != VK_NULL_HANDLE && up != VK_NULL_HANDLE;
	}

	// A 1x1 texture cleared to `color`, with its set
	bool make_solid(Texture& t, VkFormat format, VkClearColorValue color) {
		if (!make_image(t.image, format, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !give_set(t)) {
			return false;
		}
		uploads.push_back({&t, true, color, VK_NULL_HANDLE, {}});
		return true;
	}

	bool make_stand_ins() {
		const VkClearColorValue white {{1, 1, 1, 1}};
		const VkClearColorValue zero {};
		if (!make_solid(blank, VK_FORMAT_R8G8B8A8_UNORM, white) || !make_solid(no_glyphs, VK_FORMAT_R8_UNORM, zero) ||
		    !make_solid(placeholder, BLUR_FORMAT, zero)) {
			return false;
		}
		const uint32_t side = shaders::NOISE_SIDE;
		Buffer         staging;
		if (!make_image(noise.image, VK_FORMAT_R8G8B8A8_UNORM, side, side, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
		    !take(staging, sizeof(shaders::NOISE))) {
			return false;
		}
		std::memcpy(staging.mapped, shaders::NOISE, sizeof(shaders::NOISE));
		VkBufferImageCopy copy {};
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent      = {side, side, 1};
		uploads.push_back({&noise, false, {}, staging.buffer, {copy}});
		staged.push_back(staging);
		return make_buffer(globals_ubo, GLOBALS_BYTES, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false) &&
		       make_buffer(bg_ubo, BG_BYTES, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
	}

	// ---- Taking a list ----

	void drop_all() {
		retire(atlas);
		for (auto& [key, t]: images) {
			retire(t);
		}
		images.clear();
		recycle(instances);
		taken = false;
	}

	bool fail(const std::string& why) {
		say(LC_LOG_ERROR, why + "; the menu isn't drawn in the game");
		return false;
	}

	// Its glyph updates and image ops, queued as uploads, and what drawing it takes, kept
	bool take(const lc_list& l) {
		if (l.width > max_side || l.height > max_side) {
			return fail("the launcher's screen (" + std::to_string(l.width) + "x" + std::to_string(l.height) +
			            ") is bigger than this GPU's images");
		}
		if (l.atlas_side > max_side) {
			return fail("the launcher's glyph atlas (" + std::to_string(l.atlas_side) + ") is bigger than this GPU's images");
		}
		if (l.atlas_side != 0 && (!atlas || atlas->generation != l.atlas_generation || atlas->image.width != l.atlas_side)) {
			retire(atlas);
			atlas = std::make_unique<Texture>();
			atlas->generation = l.atlas_generation;
			if (!make_image(atlas->image, VK_FORMAT_R8_UNORM, l.atlas_side, l.atlas_side, 1,
			                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
				atlas.reset();
				return fail("making a " + std::to_string(l.atlas_side) + " glyph atlas failed");
			}
			uploads.push_back({atlas.get(), true, {}, VK_NULL_HANDLE, {}});
			forget_sets();
		}

		// Everything to copy, at 16-byte offsets (whole BC7 blocks, and any device's alignment)
		VkDeviceSize total = 0;
		if (l.atlas_side != 0) {
			for (uint32_t i = 0; i < l.glyph_count; i++) {
				total += round_up(static_cast<VkDeviceSize>(l.glyphs[i].width) * l.glyphs[i].height, 16);
			}
		}
		const auto level_bytes = [](const lc_list_level& level, uint32_t format) -> VkDeviceSize {
			if (format == LC_LIST_FORMAT_BC7) {
				return static_cast<VkDeviceSize>((level.width + 3) / 4) * ((level.height + 3) / 4) * 16;
			}
			return static_cast<VkDeviceSize>(level.width) * level.height * 4;
		};
		for (uint32_t i = 0; i < l.image_count; i++) {
			const auto& op = l.images[i];
			for (uint32_t j = 0; op.op == LC_LIST_IMAGE_SET && j < op.level_count; j++) {
				total += round_up(level_bytes(l.levels[op.first_level + j], op.format), 16);
			}
		}
		Buffer staging;
		if (total != 0 && !take(staging, total)) {
			return fail("allocating " + std::to_string(total) + " bytes to upload the menu's glyphs and pictures failed");
		}
		VkDeviceSize at = 0;
		if (l.atlas_side != 0 && atlas) {
			Upload u {atlas.get(), false, {}, staging.buffer, {}};
			for (uint32_t i = 0; i < l.glyph_count; i++) {
				const auto&        g     = l.glyphs[i];
				const VkDeviceSize bytes = static_cast<VkDeviceSize>(g.width) * g.height;
				if (bytes == 0) {
					continue;
				}
				std::memcpy(staging.mapped + at, lc_list_glyph_pixels(&l, i), bytes);
				VkBufferImageCopy copy {};
				copy.bufferOffset     = at;
				copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
				copy.imageOffset      = {static_cast<int32_t>(g.x), static_cast<int32_t>(g.y), 0};
				copy.imageExtent      = {g.width, g.height, 1};
				u.copies.push_back(copy);
				at += round_up(bytes, 16);
			}
			if (!u.copies.empty()) {
				uploads.push_back(std::move(u));
			}
		}
		for (uint32_t i = 0; i < l.image_count; i++) {
			const auto& op = l.images[i];
			if (op.op == LC_LIST_IMAGE_DROP) {
				if (auto it = images.find(op.key); it != images.end()) {
					retire(it->second);
					images.erase(it);
				}
				continue;
			}
			if (op.op != LC_LIST_IMAGE_SET || op.level_count == 0) {
				continue;
			}
			const auto& top = l.levels[op.first_level];
			if (op.format == LC_LIST_FORMAT_BC7 && !bc7) {
				recycle(staging);
				return fail("the launcher sent BC7 art, which this GPU can't sample");
			}
			if (top.width > max_side || top.height > max_side) {
				recycle(staging);
				return fail("the launcher sent a picture bigger than this GPU's images");
			}
			auto t = std::make_unique<Texture>();
			const VkFormat format = op.format == LC_LIST_FORMAT_BC7 ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_R8G8B8A8_UNORM;
			if (!make_image(t->image, format, top.width, top.height, op.level_count, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
			    !give_set(*t)) {
				free_texture(*t);
				recycle(staging);
				return fail("making a " + std::to_string(top.width) + "x" + std::to_string(top.height) + " picture failed");
			}
			Upload u {t.get(), false, {}, staging.buffer, {}};
			for (uint32_t j = 0; j < op.level_count; j++) {
				const uint32_t     n     = op.first_level + j;
				const VkDeviceSize bytes = level_bytes(l.levels[n], op.format);
				std::memcpy(staging.mapped + at, lc_list_level_data(&l, n), bytes);
				VkBufferImageCopy copy {};
				copy.bufferOffset     = at;
				copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, j, 0, 1};
				copy.imageExtent      = {l.levels[n].width, l.levels[n].height, 1};
				u.copies.push_back(copy);
				at += round_up(bytes, 16);
			}
			uploads.push_back(std::move(u));
			auto& slot = images[op.key];
			retire(slot);
			slot = std::move(t);
		}
		if (staging.buffer != VK_NULL_HANDLE) {
			staged.push_back(staging);
		}

		// What drawing it again takes
		Buffer instance_bytes;
		const VkDeviceSize size = static_cast<VkDeviceSize>(l.instance_count) * LC_LIST_INSTANCE_BYTES;
		if (size != 0) {
			if (!take(instance_bytes, size)) {
				return fail("allocating the menu's instances failed");
			}
			std::memcpy(instance_bytes.mapped, l.instances, size);
		}
		recycle(instances);
		instances = instance_bytes;
		width     = l.width;
		height    = l.height;
		flags     = l.flags;
		std::memcpy(globals, l.globals, sizeof(globals));
		std::memcpy(bg, l.bg, sizeof(bg));
		std::copy(std::begin(l.sources), std::end(l.sources), std::begin(sources));
		blurs.assign(l.blurs, l.blurs + l.blur_count);
		draws.assign(l.draws, l.draws + l.draw_count);
		passes.clear();
		for (uint32_t i = 0; i < l.pass_count; i++) {
			const auto& p = l.passes[i];
			passes.push_back({{{static_cast<int32_t>(p.x), static_cast<int32_t>(p.y)}, {p.width, p.height}}, p.first_draw, p.draw_count});
		}
		taken  = true;
		serial = l.serial;
		dirty  = true;
		return true;
	}

	bool shows_live() const {
		const bool source = sources[0].kind == LC_LIST_SOURCE_LIVE || sources[1].kind == LC_LIST_SOURCE_LIVE;
		return source || std::any_of(blurs.begin(), blurs.end(), [](const lc_list_blur& b) { return b.source == LC_LIST_SOURCE_LIVE; });
	}

	// ---- Drawing ----

	void record_uploads(VkCommandBuffer command) {
		for (const auto& u: uploads) {
			Image&                 im    = u.target->image;
			const VkPipelineStageFlags from = im.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			barrier(vk, command, im.image, im.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, from,
			        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im.levels);
			if (u.clear) {
				const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, im.levels, 0, 1};
				vk.CmdClearColorImage(command, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &u.color, 1, &range);
			}
			if (!u.copies.empty()) {
				vk.CmdCopyBufferToImage(command, u.source, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(u.copies.size()),
				                        u.copies.data());
			}
			barrier(vk, command, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
			        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_QUEUE_FAMILY_IGNORED,
			        VK_QUEUE_FAMILY_IGNORED, im.levels);
			im.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		uploads.clear();
		for (auto& b: staged) {
			recycle(b);
		}
		staged.clear();
	}

	// A blur of `source` (`w` x `h`) at each strength's `levels` (0: not made), made unless one is
	// kept for it. Down the most levels once, then, weakest first, back up from each strength's
	// level into an image of its own; one or two levels down is the result as it is. The levels on
	// the way up are shared, so the weaker strength goes up before the stronger one writes over
	// them. Step for step crates/render's Blur::blur_image, so the two draw the same.
	bool chain_for(std::unique_ptr<Chain>& c, VkImageView source, uint32_t w, uint32_t h, const uint16_t (&levels)[2], bool& made) {
		made = false;
		if (c && c->levels[0] == levels[0] && c->levels[1] == levels[1] && c->source == source) {
			return true;
		}
		retire(c);
		auto chain       = std::make_unique<Chain>();
		chain->levels[0] = levels[0];
		chain->levels[1] = levels[1];
		chain->source    = source;
		const uint32_t                             source_w = w;
		const uint32_t                             source_h = h;
		std::vector<std::pair<uint32_t, uint32_t>> sizes;
		const uint32_t                             most = std::max<uint32_t>({levels[0], levels[1], 1u});
		for (uint32_t i = 0; i < most; i++) {
			w = std::max(w / 2, 1u);
			h = std::max(h / 2, 1u);
			sizes.emplace_back(w, h);
		}
		struct From {
			VkImageView view;
			uint32_t    w, h;
		};
		// Targets by index: the levels down, then each strength's own; their images are made below
		std::vector<std::pair<uint32_t, uint32_t>> target_sizes = sizes;
		std::vector<std::pair<uint32_t, Chain::Step>> steps; // The source's target (NONE: the picture), the step
		for (uint32_t i = 0; i < sizes.size(); i++) {
			steps.push_back({i == 0 ? Chain::NONE : i - 1, {true, i, VK_NULL_HANDLE}});
		}
		const int order[2] = {levels[0] <= levels[1] ? 0 : 1, levels[0] <= levels[1] ? 1 : 0};
		for (const int n: order) {
			const uint32_t l = levels[n];
			const int      m = 1 - n;
			if (l == 0) {
				continue;
			}
			if (levels[m] == l && chain->results[m] != Chain::NONE) {
				chain->results[n] = chain->results[m];
				continue;
			}
			if (l <= 2) {
				chain->results[n] = l - 1;
				continue;
			}
			const auto own = static_cast<uint32_t>(target_sizes.size());
			target_sizes.push_back(sizes[1]);
			for (uint32_t i = l - 2; i >= 1; i--) {
				steps.push_back({i + 1, {false, i == 1 ? own : i, VK_NULL_HANDLE}});
			}
			chain->results[n] = own;
		}
		chain->targets.resize(target_sizes.size());
		for (size_t i = 0; i < target_sizes.size(); i++) {
			if (!make_image(chain->targets[i], BLUR_FORMAT, target_sizes[i].first, target_sizes[i].second, 1,
			                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_FORMAT_UNDEFINED, blur_pass)) {
				free_chain(*chain);
				return false;
			}
		}
		const auto count = static_cast<uint32_t>(steps.size());
		if (!make_buffer(chain->params, PARAMS_STRIDE * count, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true)) {
			free_chain(*chain);
			return false;
		}
		std::vector<VkDescriptorSet> sets(count);
		if (allocate(blur_layout, {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}},
		             chain->pool, count, sets.data()) == VK_NULL_HANDLE) {
			free_chain(*chain);
			return false;
		}
		for (uint32_t i = 0; i < count; i++) {
			const uint32_t f    = steps[i].first;
			const From     from = f == Chain::NONE ? From {source, source_w, source_h}
			                                       : From {chain->targets[f].view, target_sizes[f].first, target_sizes[f].second};
			// One source texel in uv, and the upsample's spread
			const float params[4] = {1.0f / static_cast<float>(from.w), 1.0f / static_cast<float>(from.h), 1.0f, 0.0f};
			std::memcpy(chain->params.mapped + PARAMS_STRIDE * i, params, sizeof(params));
			const auto                   image = sampled(from.view);
			const VkDescriptorImageInfo  samp {blur_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
			const VkDescriptorBufferInfo ubo {chain->params.buffer, PARAMS_STRIDE * i, 16};
			const VkWriteDescriptorSet   writes[] = {
                image_write(sets[i], 0, &image, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
                image_write(sets[i], 1, &samp, VK_DESCRIPTOR_TYPE_SAMPLER),
                buffer_write(sets[i], 2, &ubo),
            };
			vk.UpdateDescriptorSets(info.device, 3, writes, 0, nullptr);
			steps[i].second.set = sets[i];
			chain->steps.push_back(steps[i].second);
		}
		c    = std::move(chain);
		made = true;
		forget_sets();
		return true;
	}

	void run(VkCommandBuffer command, Chain& c) {
		for (const auto& step: c.steps) {
			Image& t = c.targets[step.target];
			barrier(vk, command, t.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
			const VkClearValue    black {{{0, 0, 0, 1}}};
			VkRenderPassBeginInfo rbi {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
			rbi.renderPass      = blur_pass;
			rbi.framebuffer     = t.framebuffer;
			rbi.renderArea      = {{0, 0}, {t.width, t.height}};
			rbi.clearValueCount = 1;
			rbi.pClearValues    = &black;
			vk.CmdBeginRenderPass(command, &rbi, VK_SUBPASS_CONTENTS_INLINE);
			const VkViewport viewport {0, 0, static_cast<float>(t.width), static_cast<float>(t.height), 0, 1};
			vk.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, step.down ? down : up);
			vk.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, blur_layout_p, 0, 1, &step.set, 0, nullptr);
			vk.CmdSetViewport(command, 0, 1, &viewport);
			vk.CmdSetScissor(command, 0, 1, &rbi.renderArea);
			vk.CmdDraw(command, 3, 1, 0, 0);
			vk.CmdEndRenderPass(command);
			barrier(vk, command, t.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
			t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
	}

	// Makes the list's blurs that aren't kept; true if any was made
	bool blur(VkCommandBuffer command, bool& ok) {
		bool any = false;
		for (const auto& b: blurs) {
			std::unique_ptr<Chain>* chain  = nullptr;
			VkImageView             source = VK_NULL_HANDLE;
			uint32_t                w = 0, h = 0;
			bool                    again = false;
			if (b.source == LC_LIST_SOURCE_LIVE && live.image != VK_NULL_HANDLE) {
				chain  = &live_blur;
				source = live.view;
				w      = live.width;
				h      = live.height;
				// The game's frame changes: blurred again whenever it's copied
				again = live_fresh;
			} else if (b.source == LC_LIST_SOURCE_IMAGE) {
				auto it = images.find(b.image);
				if (it == images.end()) {
					continue;
				}
				chain  = &it->second->blur;
				source = it->second->image.view;
				w      = it->second->image.width;
				h      = it->second->image.height;
			} else {
				continue;
			}
			bool made = false;
			if (!chain_for(*chain, source, w, h, b.levels, made)) {
				ok = false;
				say(LC_LOG_ERROR, "making a blur for the menu's glass failed");
				return any;
			}
			if (made || again) {
				run(command, **chain);
				any = true;
			}
		}
		return any;
	}

	bool ensure_canvas() {
		if (canvas.image != VK_NULL_HANDLE && canvas.width == width && canvas.height == height) {
			return true;
		}
		retire(canvas);
		if (canvas_pool != VK_NULL_HANDLE) {
			later([this, p = canvas_pool]() { vk.DestroyDescriptorPool(info.device, p, nullptr); });
			canvas_pool = VK_NULL_HANDLE;
		}
		canvas_set = VK_NULL_HANDLE;
		if (!make_image(canvas, CANVAS_FORMAT, width, height, 1,
		                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FORMAT_UNDEFINED,
		                canvas_pass)) {
			return false;
		}
		canvas_set = allocate(composite_layout, {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}, canvas_pool);
		if (canvas_set == VK_NULL_HANDLE) {
			return false;
		}
		const VkDescriptorImageInfo image {composite_sampler, canvas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		const auto                  write = image_write(canvas_set, 0, &image, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
		vk.UpdateDescriptorSets(info.device, 1, &write, 0, nullptr);
		dirty = true;
		return true;
	}

	// Picture `n`'s blur at each strength there is, into the background's views
	static void blurred_views(const Chain* c, int n, VkImageView (&views)[6]) {
		for (int s = 0; c && s < 2; s++) {
			if (const VkImageView v = c->result(s); v != VK_NULL_HANDLE) {
				views[2 + 2 * s + n] = v;
			}
		}
	}

	// What the background's pictures and blurs show (crates/render's update_background)
	void background_views(VkImageView (&views)[6]) const {
		const VkImageView none = placeholder.image.view;
		std::fill(std::begin(views), std::end(views), none);
		for (int n = 0; n < 2; n++) {
			if (sources[n].kind == LC_LIST_SOURCE_LIVE && live.image != VK_NULL_HANDLE) {
				views[n] = live.view;
				blurred_views(live_blur.get(), n, views);
			} else if (sources[n].kind == LC_LIST_SOURCE_IMAGE) {
				if (auto it = images.find(sources[n].image); it != images.end()) {
					views[n] = it->second->image.view;
					blurred_views(it->second->blur.get(), n, views);
				}
			}
		}
	}

	bool draw_canvas(VkCommandBuffer command) {
		VkImageView views[6];
		background_views(views);
		const VkDescriptorSet set0 = globals_for(atlas ? atlas->image.view : no_glyphs.image.view);
		const VkDescriptorSet set2 = bg_for(views);
		if (set0 == VK_NULL_HANDLE || set2 == VK_NULL_HANDLE) {
			return false;
		}

		if (canvas.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
			barrier(vk, command, canvas.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
			        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
			const VkClearColorValue       clear {};
			const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			vk.CmdClearColorImage(command, canvas.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
			barrier(vk, command, canvas.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
			        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
		} else {
			// Last drawn over the game's frame by lc_vk_draw
			barrier(vk, command, canvas.image, canvas.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
			        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
		}

		// The uniforms, in the command stream: the last present's draws may still read them
		constexpr VkPipelineStageFlags shaders_read = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		memory_barrier(vk, command, 0, 0, shaders_read, VK_PIPELINE_STAGE_TRANSFER_BIT);
		vk.CmdUpdateBuffer(command, globals_ubo.buffer, 0, GLOBALS_BYTES, globals);
		vk.CmdUpdateBuffer(command, bg_ubo.buffer, 0, BG_BYTES, bg);
		memory_barrier(vk, command, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, shaders_read);

		VkRenderPassBeginInfo rbi {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rbi.renderPass  = canvas_pass;
		rbi.framebuffer = canvas.framebuffer;
		rbi.renderArea  = {{0, 0}, {width, height}};
		vk.CmdBeginRenderPass(command, &rbi, VK_SUBPASS_CONTENTS_INLINE);
		const VkViewport viewport {0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
		vk.CmdSetViewport(command, 0, 1, &viewport);
		vk.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, blend);
		vk.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, sdf_layout, 0, 1, &set0, 0, nullptr);
		vk.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, sdf_layout, 2, 1, &set2, 0, nullptr);
		if (instances.buffer != VK_NULL_HANDLE) {
			const VkDeviceSize offset = 0;
			vk.CmdBindVertexBuffers(command, 0, 1, &instances.buffer, &offset);
		}
		const bool replace_first = (flags & LC_LIST_REPLACE_FIRST) != 0;
		for (const auto& p: passes) {
			vk.CmdSetScissor(command, 0, 1, &p.scissor);
			for (uint32_t n = 0; n < p.draw_count; n++) {
				const auto& d     = draws[p.first_draw + n];
				const Texture* tex = &blank;
				if (d.has_image != 0) {
					if (auto it = images.find(d.image); it != images.end()) {
						tex = it->second.get();
					}
				}
				vk.CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, sdf_layout, 1, 1, &tex->set, 0, nullptr);
				uint32_t first = d.first_instance;
				// Each pass's first instance is its background, which, see-through, can't go over
				// what was there
				if (n == 0 && replace_first && first < d.end_instance) {
					vk.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, replace);
					vk.CmdDraw(command, 4, 1, 0, first);
					vk.CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, blend);
					first++;
				}
				if (first < d.end_instance) {
					vk.CmdDraw(command, 4, d.end_instance - first, 0, first);
				}
			}
		}
		vk.CmdEndRenderPass(command);
		barrier(vk, command, canvas.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
		canvas.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		return true;
	}

	void destroy() {
		timer.destroy(vk, info.device);
		collect(0, true);
		uploads.clear();
		for (auto& b: staged) {
			free_buffer(b);
		}
		staged.clear();
		if (atlas) {
			free_texture(*atlas);
		}
		for (auto& [key, t]: images) {
			free_texture(*t);
		}
		images.clear();
		for (Texture* t: {&blank, &noise, &no_glyphs, &placeholder}) {
			free_texture(*t);
		}
		if (live_blur) {
			free_chain(*live_blur);
		}
		free_image(live);
		free_image(canvas);
		free_buffer(instances);
		free_buffer(globals_ubo);
		free_buffer(bg_ubo);
		for (auto& b: spare) {
			free_buffer(b);
		}
		spare.clear();
		for (VkDescriptorPool p: {canvas_pool, globals_pool, bg_pool}) {
			if (p != VK_NULL_HANDLE) {
				vk.DestroyDescriptorPool(info.device, p, nullptr);
			}
		}
		for (VkPipeline p: {blend, replace, down, up}) {
			if (p != VK_NULL_HANDLE) {
				vk.DestroyPipeline(info.device, p, nullptr);
			}
		}
		for (VkRenderPass p: {canvas_pass, blur_pass}) {
			if (p != VK_NULL_HANDLE) {
				vk.DestroyRenderPass(info.device, p, nullptr);
			}
		}
		for (VkShaderModule m: {sdf_module, blur_module}) {
			if (m != VK_NULL_HANDLE) {
				vk.DestroyShaderModule(info.device, m, nullptr);
			}
		}
		for (VkPipelineLayout l: {sdf_layout, blur_layout_p}) {
			if (l != VK_NULL_HANDLE) {
				vk.DestroyPipelineLayout(info.device, l, nullptr);
			}
		}
		for (VkDescriptorSetLayout l: {globals_layout, image_layout, bg_layout, blur_layout}) {
			if (l != VK_NULL_HANDLE) {
				vk.DestroyDescriptorSetLayout(info.device, l, nullptr);
			}
		}
		for (VkSampler sampler: {linear, noise_sampler, blur_sampler}) {
			if (sampler != VK_NULL_HANDLE) {
				vk.DestroySampler(info.device, sampler, nullptr);
			}
		}
	}
};

ListPlayer::ListPlayer() = default;
ListPlayer::~ListPlayer() {
	destroy();
}

bool ListPlayer::create(const Fns& fns, const lc_vk_info& info, const Log& log, VkDescriptorSetLayout composite_layout, VkSampler composite_sampler,
                        std::string& why) {
	destroy();
	s                    = std::make_unique<State>();
	s->vk                = fns;
	s->info              = info;
	s->log               = log;
	s->composite_layout  = composite_layout;
	s->composite_sampler = composite_sampler;
	s->vk.GetPhysicalDeviceMemoryProperties(info.physical_device, &s->memory);
	VkPhysicalDeviceProperties props {};
	s->vk.GetPhysicalDeviceProperties(info.physical_device, &props);
	s->max_side = props.limits.maxImageDimension2D;
	s->timer.create(s->vk, info);

	const auto has = [&](VkFormat format, VkFormatFeatureFlags want) {
		VkFormatProperties p {};
		s->vk.GetPhysicalDeviceFormatProperties(info.physical_device, format, &p);
		return (p.optimalTilingFeatures & want) == want;
	};
	// Sampled BC7 needs the textureCompressionBC feature enabled on the device as well
	s->bc7 = has(VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
	if (!has(BLUR_FORMAT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
		why = "this GPU can't draw into half-float images, which the menu's glass needs";
	} else if (!s->make_layouts() || !s->make_samplers() || !s->make_pipelines() || !s->make_stand_ins()) {
		why = "creating the draw-list player's pipelines or resources failed";
	} else {
		s->say(LC_LOG_INFO, std::string("drawing the launcher's menu from draw lists") + (s->bc7 ? "" : " (no BC7 art)"));
		return true;
	}
	destroy();
	return false;
}

void ListPlayer::destroy() {
	if (s) {
		s->destroy();
		s.reset();
	}
}

bool ListPlayer::begin(uint64_t completed, uint32_t generation, const lc_list* list, bool fresh, bool new_frame) {
	const bool was_showing = s->showing;
	s->collect(completed, false);
	s->timer.begin(s->vk, s->info.device, completed);
	if (generation != s->generation) {
		s->generation = generation;
		s->drop_all();
	}
	s->live_fresh = false;
	s->showing    = false;
	s->copy       = false;
	if (list == nullptr) {
		return true;
	}
	// Taken once: a second begin in the same present hands the same list over, still fresh
	if (fresh && !(s->taken && s->serial == list->serial)) {
		if (!s->take(*list)) {
			s->drop_all();
			return false;
		}
	}
	s->showing = s->taken;
	// The game's frame is copied when it's one the copy doesn't hold: a new one, or the first
	// shown since the menu opened. A paused game presents the same frame over and over, which is
	// copied and blurred once.
	// Owed until a copy is made, though this present doesn't make it (the emulator skips it).
	const bool held = s->live.image != VK_NULL_HANDLE && s->live.width == s->width && s->live.height == s->height;
	s->stale        = s->stale || new_frame || !was_showing;
	s->copy         = s->showing && s->shows_live() && (s->stale || !held);
	return true;
}

bool ListPlayer::wants_frame() const {
	return s && s->copy;
}

bool ListPlayer::capture(VkCommandBuffer command, VkImage source, VkFormat format, VkExtent2D extent) {
	if (!wants_frame()) {
		return true;
	}
	if (source == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0) {
		// The emulator can't give its frame (an HDR one, say): the glass can't be drawn, so the
		// menu goes back to the shell, as a capture it can't make does
		s->say(LC_LOG_ERROR, "the emulator has no frame to copy for the menu's glass");
		s->showing = false;
		return false;
	}
	// Held as the shell's pictures are, sRGB-encoded, read through a UNORM view: an image whose
	// values are linear is blitted into an sRGB one, which encodes them
	const VkFormat want    = holds_linear(format) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
	const bool     scaled  = extent.width != s->width || extent.height != s->height;
	const auto     feature = [&](VkFormat f) {
        VkFormatProperties p {};
        s->vk.GetPhysicalDeviceFormatProperties(s->info.physical_device, f, &p);
        return p.optimalTilingFeatures;
	};
	const VkFormatFeatureFlags from = VK_FORMAT_FEATURE_BLIT_SRC_BIT | (scaled ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0);
	if ((feature(format) & from) != from || (feature(want) & VK_FORMAT_FEATURE_BLIT_DST_BIT) == 0) {
		s->say(LC_LOG_ERROR, "the device can't blit from the frame's format (" + std::to_string(format) + ") for the menu's glass");
		s->showing = false; // The menu goes back to the shell: not drawn in the game either
		return false;
	}
	Image& live = s->live;
	if (live.image == VK_NULL_HANDLE || live.width != s->width || live.height != s->height || live.format != want) {
		s->retire(live);
		s->retire(s->live_blur);
		if (!s->make_image(live, want, s->width, s->height, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		                   VK_FORMAT_R8G8B8A8_UNORM)) {
			s->say(LC_LOG_ERROR, "making a copy of the game's frame for the menu's glass failed");
			s->showing = false;
			return false;
		}
	}
	const auto& f = s->vk;
	s->timer.stamp(f, command, Mark::GlassStart, false);
	barrier(f, command, live.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkImageBlit blit {};
	blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blit.srcOffsets[1]  = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
	blit.dstOffsets[1]  = {static_cast<int32_t>(s->width), static_cast<int32_t>(s->height), 1};
	f.CmdBlitImage(command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, live.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
	               scaled ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
	barrier(f, command, live.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	live.layout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	s->live_fresh = true;
	s->stale      = false;
	return true;
}

VkDescriptorSet ListPlayer::prepare(VkCommandBuffer command, VkImageView* view, bool& ok) {
	ok = true;
	bool changed = !s->uploads.empty();
	s->record_uploads(command);
	if (!s->showing) {
		return VK_NULL_HANDLE;
	}
	if (!s->ensure_canvas()) {
		ok = false;
		s->say(LC_LOG_ERROR, "making the menu's canvas failed");
		return VK_NULL_HANDLE;
	}
	// The glass's part started with the frame's copy, if there was one
	if (!s->live_fresh) {
		s->timer.stamp(s->vk, command, Mark::GlassStart, false);
	}
	const bool blurred = s->blur(command, ok);
	if (!ok) {
		return VK_NULL_HANDLE;
	}
	if (blurred || s->live_fresh) {
		s->timer.stamp(s->vk, command, Mark::GlassEnd, false);
	}
	changed = blurred || changed || (s->live_fresh && s->shows_live());
	if (s->dirty || changed) {
		s->timer.stamp(s->vk, command, Mark::DrawStart, false);
		if (!s->draw_canvas(command)) {
			ok = false;
			s->say(LC_LOG_ERROR, "drawing the menu failed");
			return VK_NULL_HANDLE;
		}
		s->timer.stamp(s->vk, command, Mark::DrawEnd, false);
		s->dirty = false;
	}
	if (view != nullptr) {
		*view = s->canvas.view;
	}
	return s->canvas_set;
}

void ListPlayer::stamp_composite(VkCommandBuffer command, bool start) {
	s->timer.stamp(s->vk, command, start ? Mark::CompositeStart : Mark::CompositeEnd, true);
}

void ListPlayer::end(uint64_t serial) {
	for (auto& g: s->garbage) {
		if (g.serial == 0) {
			g.serial = serial;
		}
	}
	s->live_fresh = false;
	s->timer.end(serial);
}

bool ListPlayer::gpu_times(GpuTimes& out) {
	return s && s->timer.times(out);
}

} // namespace lc::vk
