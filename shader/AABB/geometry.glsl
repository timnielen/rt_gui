#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 36) out;

uniform mat4 view;
uniform mat4 projection;

void main() {
	vec4 min = gl_in[0].gl_Position;
	vec4 max = gl_in[1].gl_Position;

	// 1
	gl_Position = projection * view * min;
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, min.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, max.y, min.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 2
	gl_Position = projection * view * min;
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, max.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, min.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 3
	gl_Position = projection * view * vec4(min.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * min;
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, min.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 4
	gl_Position = projection * view * vec4(min.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, max.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 5
	gl_Position = projection * view * vec4(max.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, max.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 6
	gl_Position = projection * view * vec4(max.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * max;
	EmitVertex();
	EndPrimitive();
	// 7
	gl_Position = projection * view * vec4(max.x, min.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * max;
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, max.y, min.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 8
	gl_Position = projection * view * vec4(max.x, min.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * max;
	EmitVertex();
	EndPrimitive();
	// 9
	gl_Position = projection * view * vec4(min.x, max.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, max.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * max;
	EmitVertex();
	EndPrimitive();
	// 10
	gl_Position = projection * view * vec4(min.x, max.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * max;
	EmitVertex();
	gl_Position = projection * view * vec4(min.x, max.y, max.z, 1.0);
	EmitVertex();
	EndPrimitive();
	// 11
	gl_Position = projection * view * vec4(min.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, min.y, min.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * min;
	EmitVertex();
	EndPrimitive();
	// 12
	gl_Position = projection * view * vec4(min.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, min.y, max.z, 1.0);
	EmitVertex();
	gl_Position = projection * view * vec4(max.x, min.y, min.z, 1.0);
	EmitVertex();
	EndPrimitive();
}