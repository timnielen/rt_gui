#pragma once
#include "Mesh.h"
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


class Model
{
public:
    Model(const char* path)
    {
        loadModel(path);
    }
    void render(Shader& shader);
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

