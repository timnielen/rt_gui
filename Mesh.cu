#include "Mesh.h"
#include "GL/gl3w.h"
#include <iostream>
#include <cuda_gl_interop.h>
#include "Material.h"
#include "BVH.h"
#include "cuda_helper.h"

#define BLOCK_SIZE 256

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, AABB aabb, MultiMaterial* material) {
    std::cout << "Vert count: " << vertices.size() << std::endl;
    this->vertices = vertices;
    this->indices = indices;
    this->aabb = aabb;
    this->material = material;
    setupMesh();
}

void Mesh::setupMesh() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    // vertex tangent coords
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
    // vertex bitangent coords
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

    glBindVertexArray(0);
}

void Mesh::render(Shader& shader, bool points) const {
    const MultiMaterial* material = this->material == nullptr ? &DEFAULT_MATERIAL : this->material;
    shader.setFloat("material.shininess", material->shininess);
    uint activeTexture = 0;
    for (uint j = 0; j < textureTypeCount; j++) {
        std::string type;
        switch (j) {
        case textureTypeDiffuse:
            type = "diffuse";
            break;
        case textureTypeSpecular:
            type = "specular";
            break;
        case textureTypeNormal:
            type = "normal";
            break;
        default:
            type = "diffuse";
        }
        shader.setBool("material." + type + ".useTex", material->textures[j] != -1);
        shader.setVec3("material." + type + ".baseColor", material->colors[j].toGLM());
        if (material->textures[j] == -1) continue;
        glActiveTexture(GL_TEXTURE0 + activeTexture);
        shader.setInt("material." + type + ".tex", activeTexture++);
        glBindTexture(GL_TEXTURE_2D, material->textures[j]);
    }
    glActiveTexture(GL_TEXTURE0);
    // draw mesh
    glBindVertexArray(VAO);
    if (points)
        glDrawElements(GL_POINTS, indices.size(), GL_UNSIGNED_INT, 0);
    else
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::renderAABB(Shader& shader) {
    shader.setVec3("color", normalize(aabb.max - aabb.min).toGLM());
    glBindVertexArray(aabbVAO);
    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);
}

__global__ void loadMaterial(Material** mat, MultiMaterial* multiMat) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index != 0) return;
    MultiMaterial* newMat = new MultiMaterial();
    *newMat = *multiMat;
    *mat = newMat;
    //*newMat = *multiMat;
    //*mat = newMat;
    //*mat = new Lambertian(Vec3(1.0f));
    //*mat = new Dielectric(1.5f);
}

__global__ void loadTriangles(Hitable** hlist, int* indices, int triCount, Vertex* vertices, Material** mat) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= triCount) return;
    hlist[index] = new Triangle(indices[3 * index], indices[3 * index + 1], indices[3 * index + 2], vertices, *mat);
}

void Mesh::loadToDevice() {

    cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsRegisterFlagsReadOnly);
    cudaGraphicsGLRegisterBuffer(&cudaEBO, EBO, cudaGraphicsRegisterFlagsReadOnly);

    int* indices;
    cudaGraphicsMapResources(1, &cudaEBO, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&indices, &num_bytes, cudaEBO);
    triCount = (num_bytes / sizeof(int)) / 3;

    Vertex* vertices;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes, cudaVBO);

    cudaMallocManaged((void**)&triangles, triCount * sizeof(Hitable*));
    Material** mat;
    cudaMalloc((void**)&mat, sizeof(Material*));
    loadMaterial << <1, 1 >> > (mat, material);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    loadTriangles << <(triCount + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (triangles, indices, triCount, vertices, mat);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    /*if (countTriangles > 1)
    {
        BVH* bvh;
        cudaMallocManaged((void**)&bvh, sizeof(BVH));
        cudaMemcpy(bvh, &BVH(triangles, countTriangles, aabb), sizeof(BVH), cudaMemcpyDefault);
        bvh->init();
        copyBvhToHitable << <1, 1 >> > (output, bvh);
        cudaFree(bvh);
    }
    else
        *output = *triangles;*/
}

void Mesh::unmap() {
    cudaGraphicsUnmapResources(1, &cudaEBO, 0);
    cudaGraphicsUnmapResources(1, &cudaVBO, 0);
}

Hit BBox::intersect(Ray& r, glm::mat4 transform) {
    Hit h;
    glm::vec4 min = transform * glm::vec4(this->min.x, this->min.y, this->min.z, 1.0f);
    glm::vec4 max = transform * glm::vec4(this->max.x, this->max.y, this->max.z, 1.0f);
    float tmin, tmax;
    if (r.invDir.x >= 0) {
        tmin = (min.x - r.origin.x) * r.invDir.x;
        tmax = (max.x - r.origin.x) * r.invDir.x;
    }
    else {
        tmin = (max.x - r.origin.x) * r.invDir.x;
        tmax = (min.x - r.origin.x) * r.invDir.x;
    }
    float tymin, tymax;
    if (r.invDir.y >= 0) {
        tymin = (min.y - r.origin.y) * r.invDir.y;
        tymax = (max.y - r.origin.y) * r.invDir.y;
    }
    else {
        tymin = (max.y - r.origin.y) * r.invDir.y;
        tymax = (min.y - r.origin.y) * r.invDir.y;
    }

    if (tmin > tymax || tymin > tmax)
        return h;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin, tzmax;
    if (r.invDir.z >= 0) {
        tzmin = (min.z - r.origin.z) * r.invDir.z;
        tzmax = (max.z - r.origin.z) * r.invDir.z;
    }
    else {
        tzmin = (max.z - r.origin.z) * r.invDir.z;
        tzmax = (min.z - r.origin.z) * r.invDir.z;
    }

    if (tmin > tzmax || tzmin > tmax)
        return h;

    h.hit = true;
    h.t = (((tmin) < (tzmin)) ? (tmin) : (tzmin));

    return h;
}

glm::vec3 transformPoint(glm::vec3 p, glm::mat4 transform) {
    auto t = transform * glm::vec4(p.x, p.y, p.z, 1.0f);
    return glm::vec3(t.x, t.y, t.z);
}

Hit Mesh::intersect(Ray& r, glm::mat4 transform) {
    Hit h;
    /*if (!aabb.intersect(r, transform).hit)
        return h;*/
    glm::vec3 origin = r.origin.toGLM();
    glm::vec3 direction = r.direction.toGLM();
    for (unsigned int i = 0; i < indices.size(); i += 3) {
        glm::vec3 v1 = transformPoint(vertices[indices[i]].Position, transform);
        glm::vec3 v2 = transformPoint(vertices[indices[i + 1]].Position, transform);
        glm::vec3 v3 = transformPoint(vertices[indices[i + 2]].Position, transform);

        glm::vec3 edge1 = v2 - v1;
        glm::vec3 edge2 = v3 - v1;
        const float EPSILON = 0.00001f;

        glm::vec3 rayVecXe2 = glm::cross(direction, edge2);
        float det = glm::dot(edge1, rayVecXe2);

        if (det > -EPSILON && det < EPSILON)
            continue;    // This ray is parallel to this triangle.

        float invDet = 1.0 / det;
        glm::vec3 s = origin - v1;
        float u = invDet * glm::dot(s, rayVecXe2);

        if (u < 0.0 || u > 1.0)
            continue;

        glm::vec3 sXe1 = glm::cross(s, edge1);
        float v = invDet * glm::dot(direction, sXe1);

        if (v < 0.0 || u + v > 1.0)
            continue;

        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = invDet * glm::dot(edge2, sXe1);
        if (t > EPSILON)
        {
            h.t = min(t, h.t);
            h.hit = true;
        }
    }
    return h;
}

__device__ bool Triangle::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const {
    /*if (!aabb.hit(r, tmin, tmax))
        return false;*/
    Vec3 edge1 = vertices[indexB].Position - vertices[indexA].Position;
    Vec3 edge2 = vertices[indexC].Position - vertices[indexA].Position;
    const float EPSILON = 1.0e-6f;

    Vec3 rayVecXe2 = cross(r.direction, edge2);
    float det = dot(edge1, rayVecXe2);

    if (det > -EPSILON && det < EPSILON)
        return false;    // This ray is parallel to this triangle.

    float invDet = 1.0f / det;
    Vec3 s = r.origin - vertices[indexA].Position;
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
        float w = (1 - u - v);
        glm::vec2 tA = vertices[indexA].TexCoords;
        glm::vec2 tB = vertices[indexB].TexCoords;
        glm::vec2 tC = vertices[indexC].TexCoords;
        rec.uvCoords.x = w * tA.x + u * tB.x + v * tC.x;
        rec.uvCoords.y = w * tA.y + u * tB.y + v * tC.y;

        Vec3 nA = vertices[indexA].Normal;
        Vec3 nB = vertices[indexB].Normal;
        Vec3 nC = vertices[indexC].Normal;
        rec.set_face_normal(r, d_normalize(w * nA + u * nB + v * nC), d_normalize(cross(edge1, edge2)));
        rec.mat = mat;

        return true;
    }

    return false;
}