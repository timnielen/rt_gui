#version 330 core
layout(lines) in;
layout(line_strip, max_vertices = 16) out;

uniform mat4 view;
uniform mat4 projection;
out vec3 color;

void main() {
	vec4 min = gl_in[0].gl_Position;
	vec4 max = gl_in[1].gl_Position;
	color = max.rgb + min.rgb / 2;
	color = color - min.rgb;
	color /= max.rgb;
	vec4 v1 = projection * view * vec4(max.x, min.y, min.z, 1.0);
	vec4 v2 = projection * view * vec4(max.x, min.y, max.z, 1.0);
	vec4 v3 = projection * view * vec4(min.x, min.y, max.z, 1.0);
	vec4 v4 = projection * view * vec4(min.x, max.y, min.z, 1.0);
	vec4 v5 = projection * view * vec4(max.x, max.y, min.z, 1.0);
	vec4 v6 = projection * view * vec4(min.x, max.y, max.z, 1.0);
	min = projection * view * min;
	max = projection * view * max;
	gl_Position = min;
	EmitVertex();
	gl_Position = v1;
	EmitVertex();
	gl_Position = v5;
	EmitVertex();
	gl_Position = v4;
	EmitVertex();
	gl_Position = v6;
	EmitVertex();
	gl_Position = v3;
	EmitVertex();
	gl_Position = min;
	EmitVertex();
	gl_Position = v4;
	EmitVertex();
	EndPrimitive();

	gl_Position = max;
	EmitVertex();
	gl_Position = v6;
	EmitVertex();
	gl_Position = v3;
	EmitVertex();
	gl_Position = v2;
	EmitVertex();
	gl_Position = v1;
	EmitVertex();
	gl_Position = v5;
	EmitVertex();
	gl_Position = max;
	EmitVertex();
	gl_Position = v2;
	EmitVertex();
	EndPrimitive();
}