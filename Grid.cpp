#include "Grid.h"
#include <cmath>
#include <GL/gl3w.h>
#include <iostream>

Grid::Grid() {
	float spacing = sideLength / float(spaceCount);
	std::cout << "spacing " << spacing << std::endl;
	//vertCount *= 3;
	unsigned int floatCount = 3 * 4 * (spaceCount + 1);
	float *vertices = new float[floatCount];
	//float vertices[18];

	for (int i = 0; i < (spaceCount + 1); i += 1) {

		vertices[6 * i + 0] = -sideLength / 2 + i * spacing;
		vertices[6 * i + 1] = 0;
		vertices[6 * i + 2] = -sideLength / 2;
		vertices[6 * i + 3] = -sideLength / 2 + i * spacing;
		vertices[6 * i + 4] = 0; 
		vertices[6 * i + 5] = sideLength / 2;

		vertices[6 * (spaceCount + 1 + i) + 0] = -sideLength / 2;
		vertices[6 * (spaceCount + 1 + i) + 1] = 0;
		vertices[6 * (spaceCount + 1 + i) + 2] = sideLength / 2 - i * spacing;
		vertices[6 * (spaceCount + 1 + i) + 3] = sideLength / 2;
		vertices[6 * (spaceCount + 1 + i) + 4] = 0;
		vertices[6 * (spaceCount + 1 + i) + 5] = sideLength / 2 - i * spacing;
	}
	std::cout << sizeof(vertices) << std::endl;
	
	glGenVertexArrays(1, &VAO);
	
	unsigned int VBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, floatCount * sizeof(float), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	delete[] vertices;
}

void Grid::render(const Shader& shader) {
	shader.setMat4("transform", glm::mat4(1));
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, 4 * (spaceCount + 1));
}