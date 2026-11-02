// Reading draw lists (launcher_list.h): the layout is the shell's render::wire.
#include "launcher_list.h"

#include <cstring>

namespace {

constexpr uint32_t MAGIC        = 0x4C444C31; // 'LDL1'
constexpr size_t   HEADER_BYTES = 768;
constexpr size_t   SECTIONS     = 7;
constexpr size_t   TABLE        = 704;
constexpr size_t   RECORD[SECTIONS] = {sizeof(lc_list_blur),   sizeof(lc_list_pass),  sizeof(lc_list_draw),
                                       LC_LIST_INSTANCE_BYTES, sizeof(lc_list_glyphs), sizeof(lc_list_image),
                                       sizeof(lc_list_level)};

static_assert(sizeof(lc_list_source) == 16 && sizeof(lc_list_blur) == 16 && sizeof(lc_list_pass) == 32 &&
              sizeof(lc_list_draw) == 24 && sizeof(lc_list_glyphs) == 32 && sizeof(lc_list_image) == 32 &&
              sizeof(lc_list_level) == 16);

template <typename T>
T get(const uint8_t* at) {
	T v;
	std::memcpy(&v, at, sizeof(T));
	return v;
}

bool little_endian() {
	const uint32_t one = 1;
	uint8_t        first;
	std::memcpy(&first, &one, 1);
	return first == 1;
}

// [at, at + len) lies inside `size`
bool inside(uint64_t at, uint64_t len, uint64_t size) {
	return at <= size && len <= size - at;
}

// The largest screen, image or atlas side a list may name, so a player can make textures from
// them as they are
constexpr uint32_t MAX_SIDE = 16384;
constexpr uint32_t MAX_ATLAS = 8192;
// The most blur levels the shell makes
constexpr uint32_t MAX_BLUR = 10;
// The background uniform's picture and gradient counts (ui::BG_IMAGES, ui::BG_GRADIENTS), which the
// shader loops over and indexes its arrays with
constexpr float MAX_BG_IMAGES    = 2;
constexpr float MAX_BG_GRADIENTS = 8;

bool power_of_two(uint32_t v) {
	return v != 0 && (v & (v - 1)) == 0;
}

// Levels in a full chain down to 1x1 from w x h
uint32_t chain_length(uint32_t w, uint32_t h) {
	uint32_t n = 1;
	for (uint32_t side = w > h ? w : h; side > 1; side >>= 1) {
		n++;
	}
	return n;
}

// Bytes a level of `w` x `h` needs in `format`, or 0 for a format this library doesn't know
uint64_t level_bytes(uint32_t format, uint32_t w, uint32_t h) {
	switch (format) {
	case LC_LIST_FORMAT_RGBA8: return uint64_t(w) * h * 4;
	case LC_LIST_FORMAT_BC7: return ((uint64_t(w) + 3) / 4) * ((uint64_t(h) + 3) / 4) * 16;
	default: return 0;
	}
}

} // namespace

extern "C" bool lc_list_read(const void* bytes, size_t size, lc_list* list) {
	if (bytes == nullptr || list == nullptr || !little_endian() || reinterpret_cast<uintptr_t>(bytes) % 8 != 0 ||
	    size < HEADER_BYTES) {
		return false;
	}
	const auto* b = static_cast<const uint8_t*>(bytes);
	if (get<uint32_t>(b) != MAGIC || get<uint32_t>(b + 4) != LC_LIST_VERSION) {
		return false;
	}
	const uint32_t used = get<uint32_t>(b + 8);
	if (used < HEADER_BYTES || used > size || used % 16 != 0) {
		return false;
	}

	lc_list l {};
	l.bytes            = b;
	l.size             = used;
	l.flags            = get<uint32_t>(b + 12);
	l.width            = get<uint32_t>(b + 16);
	l.height           = get<uint32_t>(b + 20);
	l.atlas_side       = get<uint32_t>(b + 24);
	l.dropped_glyphs   = get<uint32_t>(b + 28);
	l.atlas_generation = get<uint64_t>(b + 32);
	l.serial           = get<uint64_t>(b + 40);
	if (l.width == 0 || l.height == 0 || l.width > MAX_SIDE || l.height > MAX_SIDE ||
	    (l.atlas_side != 0 && (!power_of_two(l.atlas_side) || l.atlas_side > MAX_ATLAS))) {
		return false;
	}
	l.globals          = reinterpret_cast<const float*>(b + 48);
	l.bg               = reinterpret_cast<const float*>(b + 64);
	const auto count_in = [](float n, float most) { return n >= 0 && n <= most && n == float(int(n)); };
	if (!count_in(l.bg[4], MAX_BG_IMAGES) || !count_in(l.bg[5], MAX_BG_GRADIENTS)) {
		return false;
	}
	std::memcpy(l.sources, b + 672, sizeof(l.sources));
	for (const auto& s: l.sources) {
		if (s.kind > LC_LIST_SOURCE_IMAGE) {
			return false;
		}
	}

	const uint8_t* at[SECTIONS];
	uint32_t       count[SECTIONS];
	for (size_t n = 0; n < SECTIONS; n++) {
		const uint32_t offset = get<uint32_t>(b + TABLE + n * 8);
		count[n]              = get<uint32_t>(b + TABLE + n * 8 + 4);
		if (offset % 16 != 0 || offset < HEADER_BYTES || !inside(offset, uint64_t(count[n]) * RECORD[n], used)) {
			return false;
		}
		at[n] = b + offset;
	}
	l.blurs          = reinterpret_cast<const lc_list_blur*>(at[0]);
	l.blur_count     = count[0];
	l.passes         = reinterpret_cast<const lc_list_pass*>(at[1]);
	l.pass_count     = count[1];
	l.draws          = reinterpret_cast<const lc_list_draw*>(at[2]);
	l.draw_count     = count[2];
	l.instances      = at[3];
	l.instance_count = count[3];
	l.glyphs         = reinterpret_cast<const lc_list_glyphs*>(at[4]);
	l.glyph_count    = count[4];
	l.images         = reinterpret_cast<const lc_list_image*>(at[5]);
	l.image_count    = count[5];
	l.levels         = reinterpret_cast<const lc_list_level*>(at[6]);
	l.level_count    = count[6];

	for (uint32_t i = 0; i < l.blur_count; i++) {
		const auto& r = l.blurs[i];
		if ((r.source != LC_LIST_SOURCE_LIVE && r.source != LC_LIST_SOURCE_IMAGE) || r.levels[0] > MAX_BLUR || r.levels[1] > MAX_BLUR) {
			return false;
		}
	}
	for (uint32_t i = 0; i < l.pass_count; i++) {
		const auto& p = l.passes[i];
		if (!inside(p.x, p.width, l.width) || !inside(p.y, p.height, l.height) ||
		    !inside(p.first_draw, p.draw_count, l.draw_count)) {
			return false;
		}
	}
	for (uint32_t i = 0; i < l.draw_count; i++) {
		const auto& d = l.draws[i];
		if (d.first_instance > d.end_instance || d.end_instance > l.instance_count) {
			return false;
		}
	}
	for (uint32_t i = 0; i < l.glyph_count; i++) {
		const auto& g = l.glyphs[i];
		if (!inside(g.x, g.width, l.atlas_side) || !inside(g.y, g.height, l.atlas_side) || g.pixels % 16 != 0 ||
		    !inside(g.pixels, uint64_t(g.width) * g.height, used)) {
			return false;
		}
	}
	for (uint32_t i = 0; i < l.image_count; i++) {
		const auto& op = l.images[i];
		if (op.op == LC_LIST_IMAGE_DROP) {
			continue;
		}
		if (op.op != LC_LIST_IMAGE_SET || op.level_count == 0 || !inside(op.first_level, op.level_count, l.level_count)) {
			return false;
		}
		// A mip chain as a texture takes it: each level half the one before (rounding down, at
		// least 1), BC7's largest in whole blocks
		const auto& top = l.levels[op.first_level];
		if (top.width == 0 || top.height == 0 || top.width > MAX_SIDE || top.height > MAX_SIDE ||
		    op.level_count > chain_length(top.width, top.height) ||
		    (op.format == LC_LIST_FORMAT_BC7 && (top.width % 4 != 0 || top.height % 4 != 0))) {
			return false;
		}
		for (uint32_t k = 0; k < op.level_count; k++) {
			const auto&    v    = l.levels[op.first_level + k];
			const uint32_t w    = top.width >> k;
			const uint32_t h    = top.height >> k;
			const uint64_t need = level_bytes(op.format, v.width, v.height);
			if (v.width != (w > 0 ? w : 1) || v.height != (h > 0 ? h : 1) || need == 0 || v.bytes < need ||
			    v.data % 16 != 0 || !inside(v.data, v.bytes, used)) {
				return false;
			}
		}
	}

	*list = l;
	return true;
}
