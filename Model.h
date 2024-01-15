#pragma once
#include "Mesh.h"
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "Ray.h"
#include "Hit.h"

class Model
{
public:
    bool flipTextures;
    Model(const char* path, bool flipTextures = true) : flipTextures(flipTextures)
    {
        loadModel(path);
    }
    void render(Shader& shader, bool points = false);
    void renderAABB(Shader& shader);
    glm::vec3 position = glm::vec3(0);
    glm::vec3 rotationAxis = glm::vec3(0, 1, 0);
    float angle = 0;
    glm::vec3 scale = glm::vec3(1);
    Hit intersect(Ray&);
    std::vector<Mesh> getMeshes() const { return meshes; }
    std::vector<glm::mat4> getMeshTransformations() const { return transformations; }
    glm::mat4 getModelTransformation() const;
private:
    // model data
    std::vector<Mesh> meshes;
    std::vector<glm::mat4> transformations;
    std::string directory; 
    std::vector<Texture> textures_loaded;
    void loadModel(std::string path);
    void processNode(aiNode* node, const aiScene* scene, glm::mat4 inheritedTransformation);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);
    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);
};


