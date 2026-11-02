// The Vulkan helper's draw-list player (list_vk.cpp): draws the shell's lists (launcher_list.h) as
// the shell's own renderer does (crates/render), with its shaders, into a canvas of the list's size
// that lc_vk_draw then lays over the game's frame like a shell's buffer.
#pragma once

#include "vk_internal.h"

#include <functional>
#include <memory>
#include <string>

namespace lc::vk {

// Milliseconds of GPU time the menu took, from the latest present that measured each, as
// crates/render's GpuTimes has the shell's own: `render` spans copying and blurring the game's
// frame for glass (`glass`) and drawing the canvas (`draw`); `composite` is laying it over the frame
struct GpuTimes {
	float render    = 0;
	float glass     = 0;
	float draw      = 0;
	float composite = 0;
};

// Where a present's timestamps go, in pairs
enum class Mark : uint32_t { GlassStart, GlassEnd, DrawStart, DrawEnd, CompositeStart, CompositeEnd };

class ListPlayer {
public:
	using Log = std::function<void(lc_log_level, const std::string&)>;

	ListPlayer();
	~ListPlayer();
	ListPlayer(const ListPlayer&)            = delete;
	ListPlayer& operator=(const ListPlayer&) = delete;

	// `composite_layout` and `composite_sampler` are what lc_vk_draw draws a buffer with: the
	// canvas gets a set of that layout. False, with the reason in `why`, if it can't play lists
	// on this device.
	bool create(const Fns& fns, const lc_vk_info& info, const Log& log, VkDescriptorSetLayout composite_layout, VkSampler composite_sampler,
	            std::string& why);
	// Once the device is idle
	void destroy();

	// Starts a present: frees what the GPU is done with (`completed`), and takes this present's
	// list, applying its changes if it's `fresh`. A new `generation` drops everything taken
	// before. `new_frame`: the game's frame isn't the one it presented last. False if the list
	// can't be drawn (logged): the menu goes back to the shell.
	bool begin(uint64_t completed, uint32_t generation, const lc_list* list, bool fresh, bool new_frame);
	// Whether this present's list shows a game frame its copy doesn't hold, which lc_vk_capture
	// copies for it
	bool wants_frame() const;
	// Copies the game's finished frame (TRANSFER_SRC_OPTIMAL) for the list's glass. False if it
	// can't (logged): nothing is drawn this present
	bool capture(VkCommandBuffer command, VkImage source, VkFormat format, VkExtent2D extent);
	// Records this present's uploads, blurs and, if anything changed, the canvas, outside any
	// render pass. The canvas's set to draw it with (readable by fragment shaders), its view, or
	// VK_NULL_HANDLE for nothing to draw. False in `ok` when something failed (logged).
	VkDescriptorSet prepare(VkCommandBuffer command, VkImageView* view, bool& ok);
	// Timestamps around lc_vk_draw's composite of the canvas, inside its render pass
	void stamp_composite(VkCommandBuffer command, bool start);
	// Ends a present submitted with `serial`
	void end(uint64_t serial);
	// The GPU times the latest finished presents measured, if they changed since the last call
	bool gpu_times(GpuTimes& out);

private:
	struct State;
	std::unique_ptr<State> s;
};

} // namespace lc::vk
