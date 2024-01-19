#include "d_Model.h"
#include "cuda_helper.h"
#include <GL/gl3w.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include "Sphere.h"
#include <chrono>

#define BLOCK_SIZE 256

__device__ bool Triangle::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const {
    /*if (!aabb.hit(r, tmin, tmax))
        return false;*/
    Vec3 edge1 = vertices[indexB].Position - vertices[indexA].Position;
    Vec3 edge2 = vertices[indexC].Position - vertices[indexA].Position;
    const float EPSILON = 0.00001f;

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
        rec.set_face_normal(r, d_normalize(cross(edge2, edge1)));
        rec.mat = mat;
        return true;
    }

    return false;
}

__device__ void printTriangle(Triangle* triangle) {
    printf("\n");
    Vec3 A = triangle->vertices[triangle->indexA].Position;
    Vec3 B = triangle->vertices[triangle->indexB].Position;
    Vec3 C = triangle->vertices[triangle->indexC].Position;
    printf("A (%.3f, %.3f, %.3f)\n", A.x, A.y, A.z);
    printf("B (%.3f, %.3f, %.3f)\n", B.x, B.y, B.z);
    printf("C (%.3f, %.3f, %.3f)\n", C.x, C.y, C.z);
    printf("\n");
}

__device__ void printTriangles(Triangle** triangles, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%i:\n", i);
        printTriangle(triangles[i]);
    }
    printf("\n");
}
__global__ void loadMaterial(Material** mat) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index != 0) return;
    *mat = new Lambertian(Vec3(1, 0.1f, 0.5f));
}

__global__ void copyVertices(Vertex* vertices, d_Vertex* d_vertices, int countVertices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= countVertices) return;
    d_vertices[index] = d_Vertex(vertices[index]);
}

__global__ void loadTriangles(Hitable** hlist, int* indices, int triCount, d_Vertex* d_vertices, Material** mat) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= triCount) return;
    if(index == 0)
    {
        printf("test\n");
        printf("%p\n", indices);
    }
    hlist[index] = new Triangle(indices[3 * index], indices[3 * index + 1], indices[3 * index + 2], d_vertices, *mat);
}

__global__ void combineHitables(Hitable** output, Hitable** hlist, int count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (count == 1) {
        *output = *hlist;
        return;
    }
    BVH* bvh = new BVH(HitableList(hlist, count));
    constructBVH << <(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (bvh);
    *output = bvh;
}

void printVertex(d_Vertex v) {
    std::cout << "Position: (" << v.Position.x << ", " << v.Position.y << ", " << v.Position.z << ")" << std::endl;
    std::cout << "Normal: (" << v.Normal.x << ", " << v.Normal.y << ", " << v.Normal.z << ")" << std::endl;
}

d_Model::d_Model(const Model &m) {
    auto meshes = m.getMeshes();
    auto meshTransformations = m.getMeshTransformations();
    glm::mat4 modelTransform = m.getModelTransformation();

    int meshCount = meshes.size();

    Hitable** hlist;
    cudaMalloc((void**)&hlist, meshCount * sizeof(Hitable*));

    meshVertices = new d_Vertex * [meshCount];

    cudaGraphicsResource_t cudaEBO;
    cudaGraphicsResource_t cudaVBO;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    for (int i = 0; i < meshCount; i++) {
        std::cout << "mesh: " << i << std::endl;
        Mesh mesh = meshes[i];
        
        cudaGraphicsGLRegisterBuffer(&cudaVBO, mesh.VBO, cudaGraphicsRegisterFlagsReadOnly);
        cudaGraphicsGLRegisterBuffer(&cudaEBO, mesh.EBO, cudaGraphicsRegisterFlagsReadOnly);
        
        int* indices;
        cudaGraphicsMapResources(1, &cudaEBO, 0);

        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&indices, &num_bytes, cudaEBO);
        int countTriangles = (num_bytes / sizeof(int)) / 3;

        Vertex* vertices;
        cudaGraphicsMapResources(1, &cudaVBO, 0);

        cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes, cudaVBO);
        int countVerts = num_bytes / sizeof(Vertex);

        cudaMalloc((void**)&meshVertices[i], countVerts * sizeof(d_Vertex));

        Hitable** triangles;
        cudaMalloc((void**)&triangles, countTriangles * sizeof(Hitable*));

        auto t1 = high_resolution_clock::now();
        {
            copyVertices << <(countVerts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (vertices, meshVertices[i], countVerts);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
        auto t2 = high_resolution_clock::now();
        std::cout << "copying verts took " << duration_cast<milliseconds>(t2 - t1).count() << "ms" << std::endl;

        Material** mat;
        cudaMalloc((void**)&mat, sizeof(Material*));
        loadMaterial << <1, 1 >> > (mat);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        t1 = high_resolution_clock::now();
        {
            loadTriangles << <(countTriangles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (triangles, indices, countTriangles, meshVertices[i], mat);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
        t2 = high_resolution_clock::now();
        std::cout << "loading triangles took " << duration_cast<milliseconds>(t2 - t1).count() << "ms" << std::endl;

        t1 = high_resolution_clock::now();
        {
            combineHitables << <1, 1 >> > (hlist+i, triangles, countTriangles);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }
        t2 = high_resolution_clock::now();
        std::cout << "combining triangles took " << duration_cast<milliseconds>(t2 - t1).count() << "ms" << std::endl;

        cudaGraphicsUnmapResources(1, &cudaEBO, 0);
        cudaGraphicsUnmapResources(1, &cudaVBO, 0);
        cudaGraphicsUnregisterResource(cudaEBO);
        cudaGraphicsUnregisterResource(cudaVBO);
    }

    cudaMalloc((void**)&hitable, sizeof(Hitable*));
    combineHitables << <1, 1 >> > (this->hitable, hlist, meshCount);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}