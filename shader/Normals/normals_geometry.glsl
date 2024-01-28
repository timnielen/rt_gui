#version 330 core
layout(points) in;
layout(line_strip, max_vertices = 6) out;
in vec3[] Tangent;
in vec3[] Bitangent;
in vec3[] Normal;
in vec2[] TexCoords;
in mat3[] TBN;
out vec3 color;

uniform float normalsLength;

uniform mat4 view;
uniform mat4 projection;
void main() {
	color = vec3(1, 0, 0);
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();

	gl_Position = gl_in[0].gl_Position + normalsLength * projection * view * vec4(Tangent[0], 0.0);
	EmitVertex();

	EndPrimitive();

	color = vec3(0, 1, 0);
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();

	gl_Position = gl_in[0].gl_Position + normalsLength * projection * view * vec4(Bitangent[0], 0.0);
	EmitVertex();

	EndPrimitive();

	color = vec3(0, 0, 1);
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();

	gl_Position = gl_in[0].gl_Position + normalsLength * projection * view * vec4(Normal[0], 0.0);
	EmitVertex();

	EndPrimitive();
}