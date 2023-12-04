#pragma once
#include <string>
#include <vector>
#include "Shader.h"

struct Hit {
    bool hit = false;
    float t = 0.0f;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 invDir;
    Ray(const glm::vec3& o, const glm::vec3& dir) : origin(o), direction(dir) {
        invDir = 1.0f / direction;
    }
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    Hit intersect(Ray& r, glm::mat4 transform = glm::mat4(1));
};

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
};

struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};

class Mesh {
public:
    // mesh data
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture>      textures;
    AABB aabb;
    unsigned int aabbVAO;
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures);
    void renderAABB(Shader& shader);
    void render(Shader& shader, bool points = false);
    Hit intersect(Ray& r, glm::mat4 transform = glm::mat4(1));
private:
    //  render data
    unsigned int VAO, VBO, EBO;
    void calcAABB();
    void setupMesh();
};

