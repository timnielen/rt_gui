#pragma once
#include "Shader.h"
class Grid
{
private:
	unsigned int VAO = 0;
	unsigned int spaceCount = 20;
	float sideLength = 100;
public:
	Grid();
	void render(const Shader& shader);
};

