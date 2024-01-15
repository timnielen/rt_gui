#include "Model.h"
#include <iostream>
#include "File.h"
#include <limits>

void printMat4(glm::mat4 m) {
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            std::cout << m[i][j] << " ";
        std::cout << std::endl;
    }
}

void Model::loadModel(std::string path) {
    Assimp::Importer import;
    const aiScene * scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of('/'));

    processNode(scene->mRootNode, scene, glm::mat4(1));
}

void Model::processNode(aiNode* node, const aiScene* scene, const glm::mat4 inheritedTransormation)
{
    glm::mat4 transformation;
    memcpy(&transformation, &node->mTransformation, sizeof(glm::mat4));
    transformation = glm::transpose(transformation);
    transformation = inheritedTransormation * transformation;
    // process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
        transformations.push_back(transformation);
    }
    // then do the same for each of its children
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(node->mChildren[i], scene, transformation);
    }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene)
{
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    for (unsigned int i = 0; i < mesh->mNumVertices; i++)
    {
        Vertex vertex;
        // process vertex positions, normals and texture coordinates
        glm::vec3 vector;
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.Position = vector;
        vector.x = mesh->mNormals[i].x;
        vector.y = mesh->mNormals[i].y;
        vector.z = mesh->mNormals[i].z;
        vertex.Normal = vector;
        vector.x = mesh->mTangents[i].x;
        vector.y = mesh->mTangents[i].y;
        vector.z = mesh->mTangents[i].z;
        vertex.Tangent = vector;
        vector.x = mesh->mBitangents[i].x;
        vector.y = mesh->mBitangents[i].y;
        vector.z = mesh->mBitangents[i].z;
        vertex.Bitangent = vector;
        if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
        {

            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;

            //std::cout << vec.x << " " << vec.y << std::endl;
            vertex.TexCoords = vec;
        }
        else
        {
            std::cout << "Mesh doesn't contain texture coords" << std::endl;
            vertex.TexCoords = glm::vec2(0.0f, 0.0f);
        }
        vertices.push_back(vertex);
    }
    // process indices
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }
    // process material
    if (mesh->mMaterialIndex >= 0)
    {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        std::vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
        std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        std::vector<Texture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
    }

    return Mesh(vertices, indices, textures);
}

std::vector<Texture> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName)
{
    std::vector<Texture> textures;
    for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
    {
        aiString str;
        mat->GetTexture(type, i, &str);
        bool skip = false;
        for (unsigned int j = 0; j < textures_loaded.size(); j++)
        {
            if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
            {
                textures.push_back(textures_loaded[j]);
                skip = true;
                break;
            }
        }
        if (!skip)
        {   // if texture hasn't been loaded already, load it
            Texture texture;
            char path[256];
            strcpy(path, directory.c_str());
            strcat(path, "/");
            strcat(path, str.C_Str());
            std::cout << "loading " << typeName << " texture: " << path << std::endl;
            texture.id = load_texture(path, flipTextures);
            texture.type = typeName;
            texture.path = str.C_Str();
            textures.push_back(texture);
            textures_loaded.push_back(texture); // add to loaded textures
        }
    }
    return textures;
}
void Model::render(Shader& shader, bool points)
{
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, position);
    transform = glm::rotate(transform, angle, rotationAxis);
    transform = glm::scale(transform, scale);
    for (unsigned int i = 0; i < meshes.size(); i++)
    {
        shader.setMat4("model", transform * transformations[i]);
        meshes[i].render(shader, points);
    }
}

void Model::renderAABB(Shader& shader) {
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, position);
    transform = glm::rotate(transform, angle, rotationAxis);
    transform = glm::scale(transform, scale);
    for (unsigned int i = 0; i < meshes.size(); i++)
    {
        shader.setMat4("model", transform * transformations[i]);
        meshes[i].renderAABB(shader);
    }
}

Hit Model::intersect(Ray& ray) {
    Hit result;
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, position);
    transform = glm::rotate(transform, angle, rotationAxis);
    transform = glm::scale(transform, scale);

    result.t = std::numeric_limits<float>::infinity();
    int count = 0;
    for (unsigned int i = 0; i < meshes.size(); i++) {
        Hit h = meshes[i].intersect(ray, transform * transformations[i]);
        if (h.hit)
        {
            result.hit = true;
            count++;
            if (h.t < result.t) result.t = h.t;
        }
    }
    std::cout << count << std::endl;
    return result;
}

glm::mat4 Model::getModelTransformation() const {
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, position);
    transform = glm::rotate(transform, angle, rotationAxis);
    transform = glm::scale(transform, scale);
    return transform;
}
