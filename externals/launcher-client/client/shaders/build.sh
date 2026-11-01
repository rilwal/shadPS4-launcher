#!/usr/bin/env bash
# Rebuilds src/overlay_vk_shaders.h from the shaders here. Needs glslangValidator (in the flake's
# dev shell). The header is checked in, so emulators build without it.
set -euo pipefail
cd "$(dirname "$0")"
out=../src/overlay_vk_shaders.h
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
{
	echo "// Made by shaders/build.sh from shaders/overlay.vert and overlay.frag; don't edit."
	echo "#pragma once"
	echo "#include <cstdint>"
	for stage in vert frag; do
		glslangValidator -V --quiet --target-env vulkan1.0 -o "$tmp/$stage.spv" "overlay.$stage" >/dev/null
		echo "static const uint32_t OVERLAY_${stage^^}[] = {"
		od -An -v -tx4 "$tmp/$stage.spv" | sed -E 's/ +/ /g; s/^ //; s/ $//; s/([0-9a-f]{8})/0x\1,/g'
		echo "};"
	done
} > "$out"
echo "wrote $out"
