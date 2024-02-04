#pragma once
#include <string>
#include <vector>
#include "Shader.h"
#include "Ray.h"
#include "Hit.h"
#include "Material.h"

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
    std::string path;
};

class Mesh {
public:
    // mesh data
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    MultiMaterial* material = nullptr;
    AABB aabb;
    unsigned int aabbVAO;
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, AABB aabb, MultiMaterial* material);
    void renderAABB(Shader& shader);
    void render(Shader& shader, bool points = false) const;
    Hit intersect(Ray& r, glm::mat4 transform = glm::mat4(1));
    unsigned int VAO, VBO, EBO;

    Hitable** triangles = nullptr;
    int triCount;
    cudaGraphicsResource_t cudaEBO;
    cudaGraphicsResource_t cudaVBO;
    void loadToDevice();
    void unmap();
private:
    void setupMesh();
};


glm::vec3 transformPoint(glm::vec3 p, glm::mat4 transform);


class Triangle : public Hitable {
public:
    int indexA, indexB, indexC;
    Vertex* vertices;
    Material* mat;
    float area;
    __device__ __host__ Triangle() {}
    __device__ __host__ Triangle(int indexA, int indexB, int indexC, Vertex* vertices, Material* mat) : indexA(indexA), indexB(indexB), indexC(indexC), vertices(vertices), mat(mat) {
        Vec3 posA = vertices[indexA].Position;
        Vec3 posB = vertices[indexB].Position;
        Vec3 posC = vertices[indexC].Position;

        Vec3 min = Vec3(fmin(fmin(posA.x, posB.x), posC.x), fmin(fmin(posA.y, posB.y), posC.y), fmin(fmin(posA.z, posB.z), posC.z));
        Vec3 max = Vec3(fmax(fmax(posA.x, posB.x), posC.x), fmax(fmax(posA.y, posB.y), posC.y), fmax(fmax(posA.z, posB.z), posC.z));
        aabb = AABB(min, max);

        area = cross(posB - posA, posC - posA).length() * 0.5f;
    }
    __device__ bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const override;
    __device__ Vec3 random(curandState* local_rand_state) {
        float a = curand_uniform(local_rand_state);
        float b = curand_uniform(local_rand_state);
        if (a + b > 1)
        {
            a = 1 - a;
            b = 1 - b;
        }
        return vertices[indexA].Position + a * (vertices[indexB].Position - vertices[indexA].Position) + b * (vertices[indexC].Position - vertices[indexA].Position);
    }
    __device__ virtual float pdf(const Ray& r) {
        HitRecord rec;
        if (!this->hit(r, 0.001, FLT_MAX, rec))
            return 0;

        float distance_squared = rec.t * rec.t * r.direction.length_squared();
        float cosine = fabs(dot(r.direction, rec.normal) * rsqrtf(r.direction.length_squared()));
        return distance_squared / (cosine * area);
    }
};
