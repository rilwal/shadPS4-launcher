// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bitset>
#include <string>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_video.h>
#include <launcher_overlay.h>

#include "launcher/input.h"

namespace Launcher::Input {

namespace {

// Keys pressed while the launcher had them. Once it lets them go, their repeats still aren't the
// game's until they're released: Escape held to close the menu mustn't then reach the game.
// Main thread only.
std::bitset<SDL_SCANCODE_COUNT> g_held;

lc_overlay* Get() {
    const auto session = lc_get();
    return session != nullptr ? lc_overlay_get(session) : nullptr;
}

// The W3C KeyboardEvent.key name of a key, or "" for one the launcher has no use for (a modifier
// on its own, say: modifiers go with the other keys)
std::string KeyName(SDL_Keycode key, SDL_Keymod mod) {
    // With Num Lock off, the keypad moves (with it on, its digits come as text). Not on macOS,
    // where the keypad always types and SDL never sets SDL_KMOD_NUM.
#ifndef __APPLE__
    if ((mod & SDL_KMOD_NUM) == 0) {
        switch (key) {
        case SDLK_KP_8:
            return "ArrowUp";
        case SDLK_KP_2:
            return "ArrowDown";
        case SDLK_KP_4:
            return "ArrowLeft";
        case SDLK_KP_6:
            return "ArrowRight";
        case SDLK_KP_7:
            return "Home";
        case SDLK_KP_1:
            return "End";
        case SDLK_KP_9:
            return "PageUp";
        case SDLK_KP_3:
            return "PageDown";
        case SDLK_KP_PERIOD:
            return "Delete";
        default:
            break;
        }
    }
#else
    (void)mod;
#endif
    switch (key) {
    case SDLK_RETURN:
    case SDLK_KP_ENTER:
        return "Enter";
    case SDLK_ESCAPE:
        return "Escape";
    case SDLK_BACKSPACE:
        return "Backspace";
    case SDLK_DELETE:
        return "Delete";
    case SDLK_TAB:
        return "Tab";
    case SDLK_SPACE:
        return " ";
    case SDLK_UP:
        return "ArrowUp";
    case SDLK_DOWN:
        return "ArrowDown";
    case SDLK_LEFT:
        return "ArrowLeft";
    case SDLK_RIGHT:
        return "ArrowRight";
    case SDLK_HOME:
        return "Home";
    case SDLK_END:
        return "End";
    case SDLK_PAGEUP:
        return "PageUp";
    case SDLK_PAGEDOWN:
        return "PageDown";
    default:
        break;
    }
    if (key >= SDLK_F1 && key <= SDLK_F12) {
        return "F" + std::to_string(key - SDLK_F1 + 1);
    }
    // A key that types a character has that character as its keycode
    if ((key & SDLK_SCANCODE_MASK) != 0 || key < 0x21 || key == 0x7f || key > 0x10ffff) {
        return {};
    }
    std::string name;
    if (key < 0x80) {
        name += static_cast<char>(key);
    } else if (key < 0x800) {
        name += static_cast<char>(0xc0 | (key >> 6));
        name += static_cast<char>(0x80 | (key & 0x3f));
    } else if (key < 0x10000) {
        name += static_cast<char>(0xe0 | (key >> 12));
        name += static_cast<char>(0x80 | ((key >> 6) & 0x3f));
        name += static_cast<char>(0x80 | (key & 0x3f));
    } else {
        name += static_cast<char>(0xf0 | (key >> 18));
        name += static_cast<char>(0x80 | ((key >> 12) & 0x3f));
        name += static_cast<char>(0x80 | ((key >> 6) & 0x3f));
        name += static_cast<char>(0x80 | (key & 0x3f));
    }
    return name;
}

uint32_t Mods(SDL_Keymod mod) {
    return ((mod & SDL_KMOD_SHIFT) != 0 ? LC_MOD_SHIFT : 0) |
           ((mod & SDL_KMOD_CTRL) != 0 ? LC_MOD_CTRL : 0) |
           ((mod & SDL_KMOD_ALT) != 0 ? LC_MOD_ALT : 0) |
           ((mod & SDL_KMOD_GUI) != 0 ? LC_MOD_SUPER : 0);
}

void Pointer(lc_overlay* overlay, SDL_Window* window, float x, float y) {
    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSize(window, &width, &height) || width <= 0 || height <= 0) {
        return;
    }
    lc_input input{};
    input.kind = LC_INPUT_POINTER;
    input.x = x / static_cast<float>(width);
    input.y = y / static_cast<float>(height);
    lc_overlay_input(overlay, &input);
}

} // namespace

bool Wanted() {
    lc_overlay* overlay = Get();
    return overlay != nullptr && lc_overlay_wants_input(overlay);
}

bool Forward(SDL_Window* window, const SDL_Event& event) {
    lc_overlay* overlay = Get();
    const bool wanted = overlay != nullptr && lc_overlay_wants_input(overlay);
    const bool key = event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP;
    if (key && event.key.scancode < SDL_SCANCODE_COUNT) {
        if (!event.key.down) {
            g_held.reset(event.key.scancode);
        } else if (wanted) {
            g_held.set(event.key.scancode);
        } else if (event.key.repeat && g_held.test(event.key.scancode)) {
            return true;
        }
    }
    if (!wanted) {
        return false;
    }
    lc_input input{};
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        // Fullscreen stays the window's own
        if (event.key.key == SDLK_F11) {
            return false;
        }
        const std::string name = KeyName(event.key.key, event.key.mod);
        if (!name.empty()) {
            input.kind = LC_INPUT_KEY;
            input.key = name.c_str();
            input.down = event.key.down;
            input.repeat = event.key.repeat;
            input.mods = Mods(event.key.mod);
            lc_overlay_input(overlay, &input);
        }
        return event.key.down;
    }
    case SDL_EVENT_TEXT_INPUT:
        input.kind = LC_INPUT_TEXT;
        input.text = event.text.text;
        lc_overlay_input(overlay, &input);
        return true;
    case SDL_EVENT_TEXT_EDITING:
        // An input method composing: its text comes as SDL_EVENT_TEXT_INPUT
        return true;
    case SDL_EVENT_MOUSE_MOTION:
        Pointer(overlay, window, event.motion.x, event.motion.y);
        return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        Pointer(overlay, window, event.button.x, event.button.y);
        // SDL counts left, middle, right, back, forward from 1
        static constexpr uint32_t buttons[] = {0, 2, 1, 3, 4};
        if (event.button.button >= 1 && event.button.button <= 5) {
            input.kind = LC_INPUT_BUTTON;
            input.button = buttons[event.button.button - 1];
            input.down = event.button.down;
            lc_overlay_input(overlay, &input);
        }
        return event.button.down;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
        const float flip = event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1.0f : 1.0f;
        input.kind = LC_INPUT_WHEEL;
        input.x = event.wheel.x * flip;
        input.y = event.wheel.y * flip;
        lc_overlay_input(overlay, &input);
        return true;
    }
    default:
        return false;
    }
}

} // namespace Launcher::Input
