#version 330 core
layout(points) in;
layout(line_strip, max_vertices = 84) out;

uniform mat4 view;
uniform mat4 projection;

void main() {
	float sideLength = 100.0;
	float spacing = 5.0;
	int spaceCount = int(sideLength / spacing);
	for (int i = 0; i < (spaceCount + 1); i += 1) {
		gl_Position = projection * view * vec4(-sideLength / 2 + i * spacing, 0.0, -sideLength / 2, 1.0);
		EmitVertex();

		gl_Position = projection * view * vec4(-sideLength / 2 + i * spacing, 0.0, sideLength / 2, 1.0);
		EmitVertex();

		EndPrimitive();


		gl_Position = projection * view * vec4(-sideLength / 2, 0.0, sideLength / 2 - i * spacing, 1.0);
		EmitVertex();

		gl_Position = projection * view * vec4(sideLength / 2, 0.0, sideLength / 2 - i * spacing, 1.0);
		EmitVertex();

		EndPrimitive();
	}
}