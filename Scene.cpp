#include "Scene.h"
#include <iostream>
#include "File.h"
#include <limits>
#include "cuda_helper.h"
#include <GL/gl3w.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <chrono>


void printMat4(glm::mat4 m) {
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            std::cout << m[i][j] << " ";
        std::cout << std::endl;
    }
}

void Scene::loadModel(std::string path) {
    Assimp::Importer import;
    const aiScene * scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_GenNormals | aiProcess_CalcTangentSpace | aiProcess_GenBoundingBoxes);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of('/'));
    fileType = path.substr(path.find_last_of('.'));
    processNode(scene->mRootNode, scene, glm::mat4(1));
}

void Scene::processNode(aiNode* node, const aiScene* scene, const glm::mat4 inheritedTransormation)
{
    glm::mat4 transformation;
    memcpy(&transformation, &node->mTransformation, sizeof(glm::mat4));
    transformation = glm::transpose(transformation);
    transformation = inheritedTransormation * transformation;
    // process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        Mesh processedMesh = processMesh(mesh, scene);
        meshes.push_back(processedMesh);
        transformations.push_back(transformation);
    }
    // then do the same for each of its children
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(node->mChildren[i], scene, transformation);
    }
}

Mesh Scene::processMesh(aiMesh* mesh, const aiScene* scene)
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
        if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
        {
            vector.x = mesh->mTangents[i].x;
            vector.y = mesh->mTangents[i].y;
            vector.z = mesh->mTangents[i].z;
            vertex.Tangent = vector;
            vector.x = mesh->mBitangents[i].x;
            vector.y = mesh->mBitangents[i].y;
            vector.z = mesh->mBitangents[i].z;
            vertex.Bitangent = vector;

            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;

            //std::cout << vec.x << " " << vec.y << std::endl;
            vertex.TexCoords = vec;
        }
        else
        {
            vertex.Tangent = glm::vec3(0.0f);
            vertex.Bitangent = glm::vec3(0.0f);
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
    MultiMaterial* material = nullptr;
    if (mesh->mMaterialIndex >= 0)
    {
        aiMaterial* aiMat = scene->mMaterials[mesh->mMaterialIndex];
        material = processMaterial(aiMat, mesh->mMaterialIndex);
    }
    //process AABB
    auto min = mesh->mAABB.mMin;
    auto max = mesh->mAABB.mMax;
    AABB aabb = AABB(Vec3(min.x, min.y, min.z), Vec3(max.x,max.y,max.z));
    return Mesh(vertices, indices, aabb, material);
}

MultiMaterial* Scene::processMaterial(aiMaterial* aiMat, uint index) {
    for (int i = 0; i < materials.size(); i++)
        if (materials[i]->index == index)
            return materials[i];

    MultiMaterial* material;
    cudaMallocManaged((void**)&material, sizeof(MultiMaterial));
    *material = MultiMaterial(index);
    strcpy(material->name, aiMat->GetName().C_Str());
    aiMat->Get(AI_MATKEY_BLEND_FUNC, material->blendMode);
    aiMat->Get(AI_MATKEY_OPACITY, material->opacity);
    aiMat->Get(AI_MATKEY_SHININESS, material->shininess);
    aiMat->Get(AI_MATKEY_SHININESS_STRENGTH, material->shininessStrength);
    aiMat->Get(AI_MATKEY_REFRACTI, material->refractionIndex);
    loadMaterialTextures(aiMat, material);
    materials.push_back(material);
    return material;
}

void Scene::loadMaterialTextures(aiMaterial* aiMat, MultiMaterial* material)
{
    for (int t = 0; t < textureTypeCount; t++) {
        aiColor3D col;
        aiTextureType type;
        switch (t) {
        case textureTypeDiffuse:
            type = aiTextureType_DIFFUSE;
            aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, col);
            break;
        case textureTypeNormal:
            if (std::strcmp(fileType.c_str(), ".obj") == 0)
                type = aiTextureType_HEIGHT;
            else
                type = aiTextureType_NORMALS;
            break;
        case textureTypeSpecular:
            type = aiTextureType_SPECULAR;
            aiMat->Get(AI_MATKEY_COLOR_SPECULAR, col);
            break;
        }

        material->colors[t] = Vec3(col.r, col.g, col.b);
        int textureCount = aiMat->GetTextureCount(type);
        if (textureCount <= 0)
            continue;

        // We assume only one texture per type
        aiString str;
        aiMat->GetTexture(type, 0, &str);
        bool skip = false;
        for (uint j = 0; j < textures_loaded.size(); j++)
        {
            if (std::strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0)
            {
                material->textures[t] = textures_loaded[j].id;
                skip = true;
                break;
            }
        }
        if (!skip)
        {   // if texture hasn't been loaded already, load it
            Texture newTexture;
            char path[256];
            strcpy(path, directory.c_str());
            strcat(path, "/");
            strcat(path, str.C_Str());
            std::cout << "loading texture: " << path << std::endl;
            newTexture.id = load_texture(path, flipTextures);
            std::cout << newTexture.id << std::endl;
            newTexture.path = str.C_Str();
            material->textures[t] = newTexture.id;
            std::cout << material->textures[t] << std::endl;
            textures_loaded.push_back(newTexture); // add to loaded textures
        }

        material->cudaTextures[t].init(material->textures[t]);
        material->cudaTextures[t].map();
    }
}
void Scene::render(Shader& shader, bool points)
{
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, position);
    transform = glm::rotate(transform, angle, rotationAxis);
    transform = glm::scale(transform, scale);
    
    for (uint i = 0; i < meshes.size(); i++)
    {
        const Mesh& mesh = meshes[i];
        shader.setMat4("model", transform * transformations[i]);
        mesh.render(shader, points);
    }
}

void Scene::renderAABB(Shader& shader) {
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

Hit Scene::intersect(Ray& ray) {
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

glm::mat4 Scene::getModelTransformation() const {
    glm::mat4 transform = glm::mat4(1.0f);
    transform = glm::translate(transform, position);
    transform = glm::rotate(transform, angle, rotationAxis);
    transform = glm::scale(transform, scale);
    return transform;
}


void Scene::loadToDevice() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int meshCount = meshes.size();

    Hitable** hlist;
    cudaMalloc((void**)&hlist, meshCount * sizeof(Hitable*));

    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < meshCount; i++) {
        meshes[i].loadToDevice(hlist + i);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto t2 = high_resolution_clock::now();
    std::cout << "loading meshes into cuda took " << duration_cast<milliseconds>(t2 - t1).count() << "ms" << std::endl;

    cudaMalloc((void**)&hitable, sizeof(Hitable*));
    combineHitables << <1, 1 >> > (this->hitable, hlist, meshCount);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    auto t3 = high_resolution_clock::now();
    std::cout << "combining meshes  took " << duration_cast<milliseconds>(t3 - t2).count() << "ms" << std::endl;
}