# launcher-client

The emulator side of the launcher's plugin protocol ([docs/plugin-api.md](../docs/plugin-api.md)), as a small static library with a C header, so every emulator fork shares one implementation instead of carrying its own. It is lifted from the Kyty and RPCS3 forks' code.

It has three layers, and an emulator can stop at any of them:

| Header | What it does | Platforms |
| --- | --- | --- |
| `launcher_client.h` | The session: connects from `LAUNCHER_ENDPOINT` and `LAUNCHER_SESSION`, does the hello, and sends every session message from a writer thread of its own (so no emulator thread waits on the launcher). It waits for `configure` and hands what the launcher asks for to callbacks. | Linux, macOS, Windows |
| `launcher_overlay.h` | The composited overlay, without any graphics API. It fetches the shell's surfaces from its service, runs the header's buffer trade (a buffer goes back only once the GPU is done with it), sends `view`, and hands the capture over once the surfaces are in use. The emulator imports the surfaces, draws the buffer it's told to, and copies a frame when asked. | Linux (dma-bufs), macOS (IOSurfaces) |
| `launcher_runner.h` | The emulator as a plugin runner, so it is the launcher's plugin for its own games: it lists them (`library`), says how to start one (`launch_result` with `spawn`, usually itself), lists a game's achievements without playing it, and describes its settings and its games' settings. The game it spawns connects back as a session. | Linux, macOS, Windows |
| `launcher_overlay_vk.h` | Optional, for Vulkan renderers. It imports the surfaces as images, copies the paused frame, and draws the menu as one premultiplied quad. It loads Vulkan through the emulator's own function pointers. | Linux; macOS (MoltenVK) next |

An emulator with an OpenGL or Direct3D renderer uses `launcher_overlay.h` directly: the surfaces are plain dma-buf descriptors and IOSurfaces, and GPU work is tracked with any increasing serial (a fence counter, a timeline value).

With `LC_CAP_OSK` too, the game's text input can go on the shell's on-screen keyboard, drawn into the game's frames like the menu: `lc_osk_open` when the game asks for text (0 means show the emulator's own), keys through `lc_callbacks.osk_edit`, the text back with `lc_osk_text` after every change. The emulator keeps the text and its rules; see `launcher_client.h`.

Trophies and settings stay in each emulator, since only it knows how to read them.

## Using it

Add it as a submodule (`3rdparty/launcher-client`, pinned to a commit of this repo) and, with CMake:

```cmake
add_subdirectory(3rdparty/launcher-client/client)
target_link_libraries(my-emulator PRIVATE launcher_client)
```

Without CMake, compile `src/*.cpp` (C++17), with `include/` and `src/` on the include path, and add `src/overlay_vk.cpp` if you want the Vulkan helper. It has no dependencies but the platform's own, and Vulkan's headers for the helper (`LAUNCHER_CLIENT_VULKAN_INCLUDE`, or `$VULKAN_SDK`).

A session:

```c
#include "launcher_client.h"

static void on_quit(void *user) { request_exit(); }

if (lc_env_present()) {
    lc_callbacks cb = {0};
    cb.quit = on_quit;
    lc_connect_info info = {0};
    info.name = "Kyty 0.4.1";
    info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
    lc_session *s = lc_connect(&info, &cb, NULL, 0);
    lc_booting(s);

    const lc_kv *values;
    size_t count;
    if (lc_wait_configure(s, 2000, &values, &count)) {
        for (size_t i = 0; i < count; i++) apply_setting(values[i].key, values[i].value);
    }
    /* ...boot... */
    lc_ready(s, 0, native_window_handle);
}
```

A plugin runner connects with `lc_runner_connect` instead, declaring `LC_CAP_LIBRARY` and the rest of what it provides, sends its `lc_library` straight away, and answers `scan`, `launch` and `list_achievements` from `lc_runner_callbacks`. It covers what an emulator that spawns itself needs: the library, `launch` answered with a spawn, achievement lists, global settings (game folders among them, as an `LC_SETTING_PATH` row with a `multiple` folder pick), and the emulator's settings for the games it starts (`lc_game_settings`), which the launcher shows globally and per game and hands each spawned session in its `configure`. With `LC_CAP_COMPOSITED_OVERLAY`, a runner says its sessions draw the quick menu themselves, so launcherd adds no hook to their launches.

Every call takes a NULL session and does nothing, so the rest of the emulator can call `lc_unlocked(lc_get(), ...)` without checking for launcher mode. The overlay with the Vulkan helper, on the present thread, is in `launcher_overlay_vk.h`'s comment.

## Building and testing

```sh
nix develop            # cmake, a compiler, Vulkan headers and glslang
cmake -S client -B build/client && cmake --build build/client && ctest --test-dir build/client
```

- `protocol` checks every message the library sends and reads against `docs/plugin-api/examples`, which `crates/proto` writes from the Rust types, so the two can't drift apart.
- `session` runs a session against a fake launcherd.
- `runner` runs a plugin runner against a fake launcherd: scan, launch, achievement lists and settings.
- `overlay` runs the overlay against a fake launcherd and a fake shell service: the surfaces, the buffer trade and the capture hand-off.
- `vk` draws the menu over a frame on a real GPU and checks the blended pixel. It skips itself where no device can share linear dma-bufs, as on lavapipe.

The shaders are compiled into `src/overlay_vk_shaders.h`, which is checked in. After changing `shaders/`, run `shaders/build.sh`.

Licence: MIT OR Apache-2.0, like the rest of this repository. Both are GPL-compatible, so the GPL emulator forks can link it.
