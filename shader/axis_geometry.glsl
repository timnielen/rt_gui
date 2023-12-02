#version 330 core
layout(points) in;
layout(line_strip, max_vertices = 6) out;

uniform mat4 view;
uniform mat4 projection;

out vec3 color;
uniform float size;
void main() {
	//x
	color = vec3(1.0, 0.0, 0.0);
	gl_Position = projection * view * vec4(0.0, 0.0, 0.0, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(size * color, 1.0);
	EmitVertex();
	EndPrimitive();
	//y
	color = vec3(0.0, 1.0, 0.0);
	gl_Position = projection * view * vec4(0.0, 0.0, 0.0, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(size * color, 1.0);
	EmitVertex();
	EndPrimitive();
	//z
	color = vec3(0.0, 0.0, 1.0);
	gl_Position = projection * view * vec4(0.0, 0.0, 0.0, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(size * color, 1.0);
	EmitVertex();
	EndPrimitive();
}