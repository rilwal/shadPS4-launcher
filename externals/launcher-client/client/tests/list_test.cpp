// Reading draw lists, against one the shell's own encoder wrote (data/list.ldl, from
// crates/render/tests/list.rs), so the two can't drift apart; and lists that point outside
// themselves, which are refused.
#include "check.h"
#include "launcher_list.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

// The list's bytes, 8-byte aligned
static std::vector<uint64_t> load(size_t& size) {
	std::ifstream        file(LC_TEST_DATA_DIR "/list.ldl", std::ios::binary);
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	size = bytes.size();
	std::vector<uint64_t> out((bytes.size() + 7) / 8);
	if (!bytes.empty()) {
		std::memcpy(out.data(), bytes.data(), bytes.size());
	}
	return out;
}

static void put_u32(std::vector<uint64_t>& words, size_t at, uint32_t v) {
	std::memcpy(reinterpret_cast<uint8_t*>(words.data()) + at, &v, 4);
}

static uint32_t get_u32(const std::vector<uint64_t>& words, size_t at) {
	uint32_t v;
	std::memcpy(&v, reinterpret_cast<const uint8_t*>(words.data()) + at, 4);
	return v;
}

static void reads_the_shells_list() {
	size_t     size  = 0;
	const auto words = load(size);
	if (size == 0) {
		FAIL("no data/list.ldl");
		return;
	}
	lc_list l {};
	if (!lc_list_read(words.data(), size, &l)) {
		FAIL("data/list.ldl doesn't read");
		return;
	}
	CHECK(l.size == size);
	CHECK(l.serial == 42);
	CHECK(l.flags == (LC_LIST_REPLACE_FIRST | LC_LIST_GLASS));
	CHECK(l.width == 320 && l.height == 200);
	CHECK(l.atlas_generation == 0x123456789abcull && l.atlas_side == 2048 && l.dropped_glyphs == 1);
	CHECK(l.globals[0] == 320.0f && l.globals[1] == 200.0f && l.globals[2] == 0.25f && l.globals[3] == -1.0f);
	CHECK(l.bg[0] == 0.0f && l.bg[1] == 0.5f && l.bg[4] == 1.0f && l.bg[5] == 3.0f &&
	      l.bg[LC_LIST_BG_FLOATS - 1] == (LC_LIST_BG_FLOATS - 1) * 0.5f);
	CHECK(l.sources[0].kind == LC_LIST_SOURCE_LIVE);
	CHECK(l.sources[1].kind == LC_LIST_SOURCE_IMAGE && l.sources[1].image == 7);

	CHECK(l.blur_count == 2);
	if (l.blur_count == 2) {
		CHECK(l.blurs[0].source == LC_LIST_SOURCE_LIVE && l.blurs[0].levels == 3);
		CHECK(l.blurs[1].source == LC_LIST_SOURCE_IMAGE && l.blurs[1].levels == 2 && l.blurs[1].image == 7);
	}

	CHECK(l.pass_count == 2 && l.draw_count == 3);
	if (l.pass_count == 2 && l.draw_count == 3) {
		const auto& a = l.passes[0];
		const auto& b = l.passes[1];
		CHECK(a.x == 0 && a.y == 0 && a.width == 320 && a.height == 200 && a.first_draw == 0 && a.draw_count == 2);
		CHECK(b.x == 10 && b.y == 20 && b.width == 30 && b.height == 40 && b.first_draw == 2 && b.draw_count == 1);
		CHECK(l.draws[0].first_instance == 0 && l.draws[0].end_instance == 2 && !l.draws[0].has_image);
		CHECK(l.draws[1].first_instance == 2 && l.draws[1].end_instance == 3 && l.draws[1].has_image && l.draws[1].image == 7);
		CHECK(l.draws[2].first_instance == 0 && l.draws[2].end_instance == 3 && !l.draws[2].has_image);
	}

	CHECK(l.instance_count == 3);
	bool instances = l.instance_count == 3;
	for (uint32_t i = 0; instances && i < 3 * LC_LIST_INSTANCE_BYTES; i++) {
		instances = l.instances[i] == static_cast<uint8_t>(i);
	}
	CHECK(instances);

	CHECK(l.glyph_count == 2);
	if (l.glyph_count == 2) {
		const auto& g = l.glyphs[0];
		CHECK(g.x == 1 && g.y == 2 && g.width == 3 && g.height == 2);
		const uint8_t want[] = {10, 20, 30, 40, 50, 60};
		CHECK(std::memcmp(lc_list_glyph_pixels(&l, 0), want, sizeof(want)) == 0);
		CHECK(l.glyphs[1].x == 2040 && l.glyphs[1].y == 2047 && l.glyphs[1].width == 8 && l.glyphs[1].height == 1);
		CHECK(lc_list_glyph_pixels(&l, 1)[7] == 7);
	}

	CHECK(l.image_count == 3 && l.level_count == 3);
	if (l.image_count == 3 && l.level_count == 3) {
		const auto& set = l.images[0];
		CHECK(set.op == LC_LIST_IMAGE_SET && set.format == LC_LIST_FORMAT_RGBA8 && set.key == 7);
		CHECK(set.first_level == 0 && set.level_count == 2);
		CHECK(l.images[1].op == LC_LIST_IMAGE_DROP && l.images[1].key == 3);
		const auto& bc7 = l.images[2];
		CHECK(bc7.op == LC_LIST_IMAGE_SET && bc7.format == LC_LIST_FORMAT_BC7 && bc7.key == 8);
		CHECK(bc7.first_level == 2 && bc7.level_count == 1);
		CHECK(l.levels[0].width == 2 && l.levels[0].height == 1 && l.levels[0].bytes == 8);
		CHECK(lc_list_level_data(&l, 0)[0] == 255 && lc_list_level_data(&l, 0)[6] == 255);
		CHECK(l.levels[1].width == 1 && l.levels[1].height == 1 && lc_list_level_data(&l, 1)[0] == 128);
		CHECK(l.levels[2].width == 4 && l.levels[2].height == 4 && l.levels[2].bytes == 16);
		CHECK(lc_list_level_data(&l, 2)[15] == 0x55);
	}
}

// Each change makes the list point outside itself, or at something that isn't there
static void refuses_lists_that_point_outside() {
	size_t     size = 0;
	const auto good = load(size);
	if (size == 0) {
		return;
	}
	lc_list l {};
	CHECK(lc_list_read(good.data(), size, &l));
	const size_t blurs     = get_u32(good, 704 + 0 * 8);
	const size_t passes    = get_u32(good, 704 + 1 * 8);
	const size_t draws     = get_u32(good, 704 + 2 * 8);
	const size_t glyphs    = get_u32(good, 704 + 4 * 8);
	const size_t images    = get_u32(good, 704 + 5 * 8);
	const size_t levels    = get_u32(good, 704 + 6 * 8);

	const auto bits = [](float f) {
		uint32_t u;
		std::memcpy(&u, &f, 4);
		return u;
	};
	// The background uniform's picture and gradient counts, which the shader loops over
	const size_t pictures = 64 + 4 * 4, gradients = 64 + 5 * 4;

	struct Bad {
		const char* what;
		size_t      at;
		uint32_t    value;
	};
	const Bad bad[] = {
	    {"magic", 0, 0x4C444C32},
	    {"version", 4, LC_LIST_VERSION + 1},
	    {"used bytes", 8, static_cast<uint32_t>(size + 16)},
	    {"a source kind", 672, 3},
	    {"an unaligned section", 704 + 3 * 8, get_u32(good, 704 + 3 * 8) + 4},
	    {"a section in the header", 704 + 3 * 8, 512},
	    {"too many instances", 704 + 3 * 8 + 4, static_cast<uint32_t>(size)},
	    {"a scissor off screen", passes + 32 + 0, 300},
	    {"a pass's draws", passes + 32 + 20, 2},
	    {"a draw's instances", draws + 24 + 4, 4},
	    {"a draw backwards", draws + 0 + 0, 3},
	    {"a glyph off the atlas", glyphs + 32 + 0, 2041},
	    {"a glyph's pixels", glyphs + 16, static_cast<uint32_t>(size - 2)},
	    {"an image op", images + 0, 3},
	    {"an image format", images + 4, 2},
	    {"an image's levels", images + 20, 4},
	    {"an image with no levels", images + 20, 0},
	    {"a level too small for its size", levels + 0, 3},
	    {"a level's data", levels + 8, static_cast<uint32_t>(size - 4)},
	    {"a BC7 level too small", levels + 32 + 4, 8},
	    // What a GPU player would make textures and loops from as they are
	    {"an atlas side past the largest", 24, 0xFFFFFFFF},
	    {"an atlas side past the largest", 24, 16384},
	    {"an atlas side that isn't a power of two", 24, 3000},
	    {"a screen past the largest", 16, 0xFFFFFFFF},
	    {"a screen past the largest", 20, 16385},
	    {"an empty screen", 16, 0},
	    {"a blur's levels", blurs + 4, 0xFFFFFFFF},
	    {"a blur's levels", blurs + 4, 11},
	    {"a level bigger than the one before", levels + 16, 4},
	    {"more levels than a chain has", images + 20, 3},
	    {"a BC7 level of part blocks", levels + 32 + 0, 2},
	    {"a level's data off 16 bytes", levels + 8, get_u32(good, levels + 8) + 4},
	    {"a glyph's pixels off 16 bytes", glyphs + 16, get_u32(good, glyphs + 16) + 1},
	    {"used bytes off 16", 8, static_cast<uint32_t>(size - 8)},
	    {"background picture count", pictures, bits(3)},
	    {"background picture count", pictures, bits(1e9f)},
	    {"background picture count", pictures, bits(-1)},
	    {"background picture count", pictures, bits(0.5f)},
	    {"background picture count", pictures, 0x7FC00000}, // NaN
	    {"background gradient count", gradients, bits(9)},
	    {"background gradient count", gradients, bits(1e9f)},
	    {"background gradient count", gradients, 0x7F800000}, // infinity
	};
	for (const auto& b: bad) {
		auto words = good;
		put_u32(words, b.at, b.value);
		lc_list out {};
		out.serial = 99;
		if (lc_list_read(words.data(), size, &out)) {
			FAIL("read a list with a bad %s", b.what);
		}
		CHECK(out.serial == 99);
	}
	{
		// As many pictures and gradients as the uniform holds
		auto words = good;
		put_u32(words, pictures, bits(2));
		put_u32(words, gradients, bits(8));
		CHECK(lc_list_read(words.data(), size, &l));
	}

	CHECK(!lc_list_read(good.data(), size - 16, &l));
	CHECK(!lc_list_read(good.data(), 700, &l));
	CHECK(!lc_list_read(reinterpret_cast<const uint8_t*>(good.data()) + 4, size - 4, &l));
	CHECK(!lc_list_read(nullptr, size, &l));
}

int main() {
	reads_the_shells_list();
	refuses_lists_that_point_outside();
	return finish("list");
}
