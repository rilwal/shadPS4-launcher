/*
 * launcher_list.h: draw lists (docs/plugin-api.md, "Draw lists"), the quick menu as the shell's
 * renderer would draw it, for the game to draw itself over its own frames.
 *
 * With LC_CAP_DRAW_LISTS the shell sends lists instead of drawn buffers, and lc_overlay_begin hands
 * them over already checked (lc_frame.list). lc_list_read is the same check, for tests and tools.
 *
 * A list is the whole screen. Drawing one: write the globals and the background uniform, apply
 * the glyph updates and image ops, upload the instances, make each blur, then draw each pass,
 * scissored, a draw at a time with its image bound. Every pointer below points into the list's
 * own bytes; records are little-endian, as laid out here, so on a little-endian machine they're
 * read in place.
 */
#ifndef LAUNCHER_LIST_H
#define LAUNCHER_LIST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The layout version this library reads: the shell's LIST_VERSION. A list of another version
   (a shell from another build) isn't read. */
#define LC_LIST_VERSION 1u

/* Bytes in one instance, as the shader's vertex layout reads them. */
#define LC_LIST_INSTANCE_BYTES 128u
/* Floats in the background uniform. */
#define LC_LIST_BG_FLOATS 152u

/* Flags */
#define LC_LIST_REPLACE_FIRST 1u /* Each pass's first instance replaces what's under it, no blend */
#define LC_LIST_GLASS         2u /* Some instance is glass, so the blurs show */

/* Where a background picture or a blur comes from */
#define LC_LIST_SOURCE_NONE  0u
#define LC_LIST_SOURCE_LIVE  1u /* The game's own frame, under the menu */
#define LC_LIST_SOURCE_IMAGE 2u /* An image, by key */

/* Image ops */
#define LC_LIST_IMAGE_SET  1u
#define LC_LIST_IMAGE_DROP 2u

/* Image formats. Both hold premultiplied, sRGB-encoded values, kept in a UNORM texture: the
   shader decodes them. */
#define LC_LIST_FORMAT_RGBA8 0u
#define LC_LIST_FORMAT_BC7   1u /* 4x4 blocks of 16 bytes */

typedef struct lc_list_source {
	uint32_t kind; /* LC_LIST_SOURCE_* */
	uint32_t reserved;
	uint64_t image; /* With LC_LIST_SOURCE_IMAGE */
} lc_list_source;

/* A picture to blur `levels` half-size levels down and back up to a quarter of its size, before
   the passes. The live frame blurs every time; an image's blur can be kept until it changes. */
typedef struct lc_list_blur {
	uint32_t source; /* LC_LIST_SOURCE_LIVE or LC_LIST_SOURCE_IMAGE */
	uint32_t levels;
	uint64_t image;
} lc_list_blur;

/* Draws scissored to x, y, width, height (in pixels, on screen) */
typedef struct lc_list_pass {
	uint32_t x;
	uint32_t y;
	uint32_t width;
	uint32_t height;
	uint32_t first_draw;
	uint32_t draw_count;
	uint64_t reserved;
} lc_list_pass;

/* Instances [first_instance, end_instance) drawn with one image bound (a blank one without) */
typedef struct lc_list_draw {
	uint32_t first_instance;
	uint32_t end_instance;
	uint32_t has_image;
	uint32_t reserved;
	uint64_t image;
} lc_list_draw;

/* Glyph pixels for the atlas: R8, `width` x `height`, `width` bytes a row, at (x, y) */
typedef struct lc_list_glyphs {
	uint32_t x;
	uint32_t y;
	uint32_t width;
	uint32_t height;
	uint32_t pixels; /* Offset in the list */
	uint32_t reserved;
	uint64_t reserved2;
} lc_list_glyphs;

/* Adds (or replaces) an image, with levels [first_level, first_level + level_count) of the list's
   levels, largest first; or drops one. A drop of an image the player doesn't have does nothing. */
typedef struct lc_list_image {
	uint32_t op;     /* LC_LIST_IMAGE_* */
	uint32_t format; /* LC_LIST_FORMAT_* */
	uint64_t key;
	uint32_t first_level;
	uint32_t level_count;
	uint64_t reserved;
} lc_list_image;

/* One mip level: `bytes` of data at offset `data` in the list, rows packed (RGBA8: width * 4 bytes
   a row; BC7: a row of blocks, whole blocks, so the level's size rounds up to a multiple of 4) */
typedef struct lc_list_level {
	uint32_t width;
	uint32_t height;
	uint32_t data;
	uint32_t bytes;
} lc_list_level;

/* A list, checked: every count, range and offset below lies inside it. */
typedef struct lc_list {
	const uint8_t *bytes; /* The whole list; offsets count from here */
	size_t         size;

	uint64_t serial; /* Lists from one set of slots count up */
	uint32_t flags;  /* LC_LIST_* */
	uint32_t width;  /* The screen it was laid out for */
	uint32_t height;

	/* The glyph atlas, side x side R8. A new generation is a new, empty atlas: drop the old one
	   (and anything kept from it) before applying the updates. */
	uint64_t atlas_generation;
	uint32_t atlas_side;
	uint32_t dropped_glyphs; /* Glyphs that didn't fit: the shell has a bug, not the player */

	const float   *globals; /* 4: the screen's size, then towards the light that glass rims catch */
	const float   *bg;      /* LC_LIST_BG_FLOATS: the background uniform, as the shader reads it */
	lc_list_source sources[2]; /* What the uniform's two pictures show */

	const lc_list_blur   *blurs;
	uint32_t              blur_count;
	const lc_list_pass   *passes;
	uint32_t              pass_count;
	const lc_list_draw   *draws;
	uint32_t              draw_count;
	const uint8_t        *instances; /* instance_count * LC_LIST_INSTANCE_BYTES */
	uint32_t              instance_count;
	const lc_list_glyphs *glyphs;
	uint32_t              glyph_count;
	const lc_list_image  *images;
	uint32_t              image_count;
	const lc_list_level  *levels;
	uint32_t              level_count;
} lc_list;

/* Reads and checks a list of `size` bytes at `bytes` (8-byte aligned), which must stay as they are
   while `list` is used. False, filling nothing in, unless it's a whole list of LC_LIST_VERSION
   whose every section, range and offset lies inside it, so a player can use it as it is:
   - the screen is at most 16384 on a side, and the atlas a power of two up to 8192 (or 0)
   - each pass's draws are in the draws, each draw's instances in the instances, and each scissor
     on screen
   - each blur is of a source there is, at most 10 levels
   - each glyph update lies in the atlas
   - each image's levels are in the levels, a full mip chain's worth at most: each half the size
     of the one before (rounding down, at least 1), the largest at most 16384 on a side and, in
     BC7, whole blocks; and each level's data is big enough for its size and format
   - every section and every piece of data starts on a multiple of 16 bytes */
bool lc_list_read(const void *bytes, size_t size, lc_list *list);

/* Glyph update `i`'s pixels, and level `i`'s data */
static inline const uint8_t *lc_list_glyph_pixels(const lc_list *list, uint32_t i) {
	return list->bytes + list->glyphs[i].pixels;
}
static inline const uint8_t *lc_list_level_data(const lc_list *list, uint32_t i) {
	return list->bytes + list->levels[i].data;
}

#ifdef __cplusplus
}
#endif

#endif /* LAUNCHER_LIST_H */
