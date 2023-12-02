#pragma once

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class Shader
{
private:
    unsigned int vertex = 0, geometry = 0, fragment = 0;
    unsigned int setShader(const char* path, int type);
public:
    // the program ID
    unsigned int ID = 0;
    
    // constructor reads and builds the shader
    Shader(const char* vertexPath, const char* fragmentPath);
    Shader() {}

    // use/activate the shader
    void use();
    // utility uniform functions
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setMat4(const std::string& name, const glm::mat4& matrix) const;
    void setVec3(const std::string& name, const glm::vec3& v) const;
    void destroy();
    void setVertex(const char* path);
    void setFragment(const char* path);
    void setGeometry(const char* path);
    void link();
};

