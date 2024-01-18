#include "d_Model.h"
#include "cuda_helper.h"
#include <GL/gl3w.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include "Sphere.h"

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

__global__ void loadTriangles(int* indices, int countIndices, Vertex* vertices, d_Vertex* d_vertices, int countVertices, Hitable** hlist) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    for (int i = 0; i < countVertices; i++) {
        d_vertices[i] = d_Vertex(vertices[i]);
    }
    Material* mat = new Lambertian(Vec3(1,0.1f,0.5f));
    const int triCount = countIndices / 3;
    Hitable** triangles = new Hitable*[triCount];
    //Triangle* triangles = new Triangle[triCount];
    AABB aabb;
    for (int i = 0; i < triCount; i++) {
        //hitables[i] = triangles + i;
        triangles[i] = new Triangle(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2], d_vertices, mat);
    }
    curandState local_rand_state;
    curand_init(7698, 0, 0, &local_rand_state);
    //printTriangles((Triangle**)triangles, triCount);
    BVH * bvh = new BVH(HitableList(triangles, triCount));
    int blockSize = 256;
    int numBlocks = (triCount + blockSize - 1) / blockSize;
    constructBVH << <numBlocks, blockSize >> > (bvh);
    *hlist = bvh; //new Sphere(Vec3(0,0,-1), 0.5f, mat); // 
}

__global__ void combineMeshes(Hitable** hlist, int size, Hitable** output) {
    curandState local_rand_state;
    curand_init(1348, 0, 0, &local_rand_state);
    /*BVH* bvh = new BVH(HitableList(hlist, size));
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    constructBVH << <numBlocks, blockSize >> > (bvh);*/
    BVH* bvh = (BVH*)*hlist;
    bvh->print();
    *output = *hlist; // new BVH_Node(hlist, 0, size, &local_rand_state);
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

    for (int i = 0; i < meshCount; i++) {
        Mesh mesh = meshes[i];

        cudaGraphicsGLRegisterBuffer(&cudaVBO, mesh.VBO, cudaGraphicsRegisterFlagsReadOnly);
        cudaGraphicsGLRegisterBuffer(&cudaEBO, mesh.EBO, cudaGraphicsRegisterFlagsReadOnly);

        int* indices;
        cudaGraphicsMapResources(1, &cudaEBO, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&indices, &num_bytes, cudaEBO);
        int countIndices = num_bytes / sizeof(int);

        Vertex* vertices;
        cudaGraphicsMapResources(1, &cudaVBO, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes, cudaVBO);
        int countVerts = num_bytes / sizeof(Vertex);

        cudaMalloc((void**)&meshVertices[i], countVerts * sizeof(d_Vertex));

        std::cout << "mesh: " << i << std::endl;
        loadTriangles <<<1, 1 >>> (indices, countIndices, vertices, meshVertices[i], countVerts, hlist+i);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaGraphicsUnmapResources(1, &cudaEBO, 0);
        cudaGraphicsUnmapResources(1, &cudaVBO, 0);
        cudaGraphicsUnregisterResource(cudaEBO);
        cudaGraphicsUnregisterResource(cudaVBO);
    }

    cudaMalloc((void**)&hitable, sizeof(Hitable*));
    combineMeshes << <1, 1 >> > (hlist, meshCount, this->hitable);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}