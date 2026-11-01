// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// shadPS4 as the game launcher's plugin for its games (the launcher's plugin runner protocol):
// it finds the games, has launcherd start them as `shadps4 --game <game>`, lists their trophies
// without playing them, and describes shadPS4's settings for the launcher to show.
namespace Launcher::Plugin {

// Started as a plugin: `--launcher-plugin`, or no arguments with the launcher's environment.
[[nodiscard]] bool IsPluginLaunch(int argc, char* argv[]);

// Runs until launcherd hangs up. Returns the process's exit code.
int Run(int argc, char* argv[]);

} // namespace Launcher::Plugin
