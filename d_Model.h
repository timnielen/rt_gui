#pragma once
#include "cuda_runtime.h"
#include "Model.h"
#include "Material.h"
#include "BVH.h"



struct d_Vertex {
    __device__ __host__ d_Vertex() {}
    __device__ __host__ d_Vertex(Vertex v) : Position(v.Position), Normal(v.Normal) {}
    d_Vertex(Vertex v, glm::mat4 transform) : Position(transformPoint(v.Position, transform)), Normal (transformPoint(v.Normal, transform)) {}
    Vec3 Position;
    Vec3 Normal;
};

class Triangle : public Hitable {
public:
    int indexA, indexB, indexC;
    d_Vertex* vertices;
    Material* mat;
    __device__ __host__ Triangle() {}
    __device__ __host__ Triangle(int indexA, int indexB, int indexC, d_Vertex* vertices, Material* mat) : indexA(indexA), indexB(indexB), indexC(indexC), vertices(vertices), mat(mat) {
        Vec3 posA = vertices[indexA].Position;
        Vec3 posB = vertices[indexB].Position;
        Vec3 posC = vertices[indexC].Position;

        Vec3 min = Vec3(fmin(fmin(posA.x, posB.x), posC.x), fmin(fmin(posA.y, posB.y), posC.y), fmin(fmin(posA.z, posB.z), posC.z));
        Vec3 max = Vec3(fmax(fmax(posA.x, posB.x), posC.x), fmax(fmax(posA.y, posB.y), posC.y), fmax(fmax(posA.z, posB.z), posC.z));
        aabb = AABB(min, max);
    }
    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override;
};



class d_Model {
public:
    Hitable** hitable;
    d_Vertex** meshVertices;
    d_Model(const Model& m);
};
