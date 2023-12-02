#include "Shader.h"
#include <GL/gl3w.h>
#include "File.h"
#include <iostream>


Shader::Shader(const char* vertexPath, const char* fragmentPath) {
	setVertex(vertexPath);
	setFragment(fragmentPath);
	link();
}

void Shader::use() {
	glUseProgram(ID);
}

void Shader::setBool(const std::string& name, bool value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}
void Shader::setInt(const std::string& name, int value) const
{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setMat4(const std::string& name, const glm::mat4 &matrix) const
{
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setVec3(const std::string& name, const glm::vec3& v) const {
	glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(v));
}

void Shader::destroy() {
	glDeleteProgram(ID);
	ID = 0;
}


void Shader::setVertex(const char* path) {
	glDeleteShader(vertex);
	vertex = setShader(path, GL_VERTEX_SHADER);
}
void Shader::setFragment(const char* path) {
	glDeleteShader(fragment);
	fragment = setShader(path, GL_FRAGMENT_SHADER);
}
void Shader::setGeometry(const char* path) {
	glDeleteShader(geometry);
	geometry = setShader(path, GL_GEOMETRY_SHADER);
}

unsigned int Shader::setShader(const char* path, int type) {
	// 1. retrieve the vertex/fragment source code from filePath
	std::string fileContent = readTextFile(path);
	const char* shaderCode = fileContent.c_str();

	unsigned int shader;
	shader = glCreateShader(type);
	glShaderSource(shader, 1, &shaderCode, NULL);
	glCompileShader(shader);

	int  success;
	char infoLog[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	return shader;
}

void Shader::link() {
	glDeleteProgram(ID);
	ID = glCreateProgram();
	if(vertex != 0)
		glAttachShader(ID, vertex);
	if (fragment != 0)
		glAttachShader(ID, fragment);
	if (geometry != 0)
		glAttachShader(ID, geometry);
	glLinkProgram(ID);

	int  success;
	char infoLog[512];
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(ID, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertex);
	glDeleteShader(geometry);
	glDeleteShader(fragment);
}