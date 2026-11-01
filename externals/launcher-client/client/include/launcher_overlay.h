/*
 * launcher_overlay.h: the composited overlay (docs/plugin-api.md, "Composited overlay"), without
 * any graphics API.
 *
 * The shell draws its quick menu into three shared BGRA8 buffers (premultiplied alpha) and trades
 * them with the game through a small shared header. When the menu opens, the game copies the frame
 * it paused on into a surface of its own and hands it back, as the menu's background.
 *
 * The library does the protocol: it takes overlay_surface and overlay_visible off the session,
 * fetches the surfaces from the shell's service, runs the header's buffer trade, sends view, and
 * hands the capture over once the shell's surfaces are in use. The emulator only imports the
 * surfaces into its renderer, draws the buffer it's told to, and copies a frame when asked.
 *
 * GPU work is tracked with serials: any increasing number the emulator gives each present's
 * submission (a timeline value, a fence counter), and the last one its GPU has finished. A buffer
 * goes back to the shell only once the submission that last read it has finished.
 *
 * The lc_overlay_* calls below are for the present thread (one thread, or calls serialised by the
 * caller), apart from lc_overlay_get.
 *
 * Linux shares dma-bufs (and the header as a memfd); macOS, IOSurfaces. Elsewhere, and on a
 * platform whose backend isn't in yet, lc_overlay_get returns NULL and the shell's own window
 * comes to the front instead, as for any game.
 */
#ifndef LAUNCHER_OVERLAY_H
#define LAUNCHER_OVERLAY_H

#include "launcher_client.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lc_overlay lc_overlay;
typedef struct lc_surface lc_surface;

/* Whether this platform can share surfaces with the launcher at all. */
bool lc_overlay_supported(void);

/* The session's overlay, if it was connected with LC_CAP_COMPOSITED_OVERLAY on a platform that
   supports it; else NULL. Any thread. It lives as long as the session. */
lc_overlay *lc_overlay_get(lc_session *session);

/* The window's size in pixels, which the shell lays the menu out for. Call it only from a
   renderer that can draw the overlay, once it knows it can: the first one is what makes the
   launcher count on this session to draw the menu. Sent after ready, and again on every change. */
void lc_overlay_view(lc_overlay *overlay, uint32_t width, uint32_t height);

/* What to do this present */
typedef struct lc_frame {
	/* Changes whenever the shell's surfaces do. Import `buffers` again when it changes, and keep
	   the old imports until the GPU is done with them; then say how it went with
	   lc_overlay_surfaces_ready. The surfaces stay valid until a later lc_overlay_begin reports
	   another generation. */
	uint32_t                 generation;
	const lc_surface *const *buffers;
	uint32_t                 buffer_count; /* 0: no overlay */
	uint32_t                 width;        /* The buffers' size */
	uint32_t                 height;

	/* The buffer to draw over this present, as one premultiplied quad filling the window, or -1
	   for none. `fresh` says the shell has written it since it was last drawn: make its writes
	   visible (acquire it from the external queue family, in Vulkan) before sampling it. */
	int32_t draw;
	bool    fresh;

	/* Nonzero: the shell has a newer frame of the menu, but the buffer the game would give back
	   for it is still read by the submission with this serial. Wait for the GPU to finish it (a
	   fence, a timeline value) and call lc_overlay_begin again for this present, with the new
	   `completed`, to take the newer frame. A renderer that keeps more than one frame in flight
	   has to, or the menu stays on the frame it has. Waiting is only ever for one serial, which is
	   usually a frame old. */
	uint64_t wait;

	/* Copy this present's frame for the shell, then lc_overlay_capture (or cancel with NULL). */
	bool capture;
} lc_frame;

/* Starts a present. `completed` is the newest serial the GPU has finished. It can be called again
   before lc_overlay_end, after waiting for `wait`; the frame it fills replaces the first. */
void lc_overlay_begin(lc_overlay *overlay, uint64_t completed, lc_frame *frame);

/* After importing a generation's buffers: whether it worked. On failure the library lets the
   surfaces go and sends no capture, so the shell comes to the front instead. */
void lc_overlay_surfaces_ready(lc_overlay *overlay, uint32_t generation, bool ok);

/* Ends a present submitted with `serial`. `drew` says it sampled the frame's `draw` buffer. */
void lc_overlay_end(lc_overlay *overlay, uint64_t serial, bool drew);

/* A frame copied for the shell: BGRA8 (unorm), `width` x `height`. */
typedef struct lc_capture {
	/* Linux: a dma-buf of one plane at `offset`, `stride` bytes a row, DRM format modifier
	   `modifier` (DRM_FORMAT_MOD_LINEAR is 0). The library takes the descriptor and closes it. */
	int      fd;
	uint32_t stride;
	uint64_t offset;
	uint64_t modifier;
	/* macOS: an IOSurfaceRef, which the library retains. */
	void *iosurface;

	uint32_t width;
	uint32_t height;
} lc_capture;

/* Hands over the frame lc_frame.capture asked for, written by the submission `serial`. It goes to
   the shell once that submission has finished and the shell's surfaces are in use, from a thread
   of the library's own. NULL says the frame couldn't be copied. */
void lc_overlay_capture(lc_overlay *overlay, const lc_capture *capture, uint64_t serial);

/* Whether presents are needed though the game sends no frames (paused, or loading): to show the
   overlay, finish a capture, or catch up with a change. Any thread. */
bool lc_overlay_wants_presents(lc_overlay *overlay);

/* A surface's id, size and memory layout, for importing it. */
uint32_t lc_surface_id(const lc_surface *surface);
uint32_t lc_surface_stride(const lc_surface *surface);
uint64_t lc_surface_offset(const lc_surface *surface);
uint64_t lc_surface_modifier(const lc_surface *surface);
/* Linux: the dma-buf, still the library's (dup it for an import that takes the descriptor). */
int lc_surface_fd(const lc_surface *surface);
/* macOS: the IOSurfaceRef, still the library's (retain it to keep it). */
void *lc_surface_iosurface(const lc_surface *surface);

#ifdef __cplusplus
}
#endif

#endif /* LAUNCHER_OVERLAY_H */
