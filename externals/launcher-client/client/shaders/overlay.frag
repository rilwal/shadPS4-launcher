#version 450
// The shell's buffer, premultiplied. Into an sRGB target its values are decoded first, since the
// target encodes them again.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D menu;
layout(push_constant) uniform Push {
	uint decode_srgb;
};

vec3 to_linear(vec3 c) {
	return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

void main() {
	vec4 c = texture(menu, uv);
	if (decode_srgb != 0u) {
		c.rgb = to_linear(c.rgb);
	}
	color = c;
}
