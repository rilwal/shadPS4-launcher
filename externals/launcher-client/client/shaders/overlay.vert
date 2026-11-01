#version 450
// One triangle covering the target; uv runs 0..1 across it
layout(location = 0) out vec2 uv;

void main() {
	uv          = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
