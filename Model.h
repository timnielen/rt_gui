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

struct Vert {
    Vec3 Position;
    Vec3 Normal;
};

class Triangle : public Hitable {
public:
    Vert a, b, c;
    Material* mat;
    __device__ __host__ Triangle(Vert a, Vert b, Vert c, Material *mat) : a(a), b(b), c(c), mat(mat) {
        Vec3 min = Vec3(fmin(fmin(a.Position.x, b.Position.x), c.Position.x), fmin(fmin(a.Position.y, b.Position.y), c.Position.y), fmin(fmin(a.Position.z, b.Position.z), c.Position.z));
        Vec3 max = Vec3(fmax(fmax(a.Position.x, b.Position.x), c.Position.x), fmax(fmax(a.Position.y, b.Position.y), c.Position.y), fmax(fmax(a.Position.z, b.Position.z), c.Position.z));
        aabb = AABB(min, max);
    }
    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override{
        /*if (!aabb.hit(r, tmin, tmax))
            return false;*/
        Vec3 edge1 = b.Position - a.Position;
        Vec3 edge2 = c.Position - a.Position;
        const float EPSILON = 0.00001f;

        Vec3 rayVecXe2 = cross(r.direction, edge2);
        float det = dot(edge1, rayVecXe2);

        if (det > -EPSILON && det < EPSILON)
            return false;    // This ray is parallel to this triangle.

        float invDet = 1.0f / det;
        Vec3 s = r.origin - a.Position;
        float u = invDet * dot(s, rayVecXe2);

        if (u < 0.0f || u > 1.0f)
            return false;

        Vec3 sXe1 = cross(s, edge1);
        float v = invDet * dot(r.direction, sXe1);

        if (v < 0.0f || u + v > 1.0f)
            return false;

        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = invDet * dot(edge2, sXe1);
        if (t >= tmin && t <= tmax)
        {
            rec.t = t;
            rec.p = r.at(t);
            rec.set_face_normal(r, d_normalize(cross(edge2, edge1)));
            rec.mat = mat;
            return true;
        }

        return false;
    }

};

//class d_Model : Hitable {
//    HitableList* meshes;
//
//    d_Model(const Model& m) {
//
//    }
//};