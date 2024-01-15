#pragma once
#include <string>
#include <vector>
#include "Shader.h"
#include "Ray.h"


struct BBox {
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
    BBox aabb;
    unsigned int aabbVAO;
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures);
    void renderAABB(Shader& shader);
    void render(Shader& shader, bool points = false);
    Hit intersect(Ray& r, glm::mat4 transform = glm::mat4(1));
    unsigned int VAO, VBO, EBO;
private:
    //  render data
    void calcAABB();
    void setupMesh();
};


glm::vec3 transformPoint(glm::vec3 p, glm::mat4 transform);
