# launcher-client (vendored)

`client/` is a copy of the `client/` folder of the game launcher
(rilwal/game-launcher), the shared C library emulator forks use to talk to it:
a game session (launcher_client.h, launcher_overlay.h) and a plugin runner
(launcher_runner.h). Its protocol is documented in that repository's
`docs/plugin-api.md`.

It's copied rather than a submodule because the launcher's repository is
private and this fork is public. Once the launcher is public, this goes back
to being a submodule. Don't edit it here: change it in the
launcher, then copy it again.

Copied from: 17479ded0f434d4fe344569c87526fbef47600b8 (game-launcher#71) merged with 8d033924 (game-launcher#74 overlay-ui, which carries #73 and #72), 2026-11-02

To update, from a checkout of the launcher:

```sh
externals/launcher-client/update.sh /path/to/game-launcher
```
