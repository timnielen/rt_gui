#include "Mesh.h"
#include "GL/gl3w.h"
#include <iostream>

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures) {
    this->vertices = vertices;
    this->indices = indices;
    this->textures = textures;
    calcAABB();
    setupMesh();
}

void Mesh::calcAABB() {
    aabb.max = glm::vec3(vertices[0].Position);
    aabb.min = glm::vec3(vertices[0].Position);
    for (unsigned int i = 1; i < vertices.size(); i++) {
        aabb.max.x = max(vertices[i].Position.x, aabb.max.x);
        aabb.max.y = max(vertices[i].Position.y, aabb.max.y);
        aabb.max.z = max(vertices[i].Position.z, aabb.max.z);

        aabb.min.x = min(vertices[i].Position.x, aabb.min.x);
        aabb.min.y = min(vertices[i].Position.y, aabb.min.y);
        aabb.min.z = min(vertices[i].Position.z, aabb.min.z);
    }
    std::cout << "max: " << aabb.max.x << " " << aabb.max.y << " " << aabb.max.z << std::endl;
    std::cout << "min: " << aabb.min.x << " " << aabb.min.y << " " << aabb.min.z << std::endl;
    std::cout << sizeof(BBox) << std::endl;


    glGenVertexArrays(1, &aabbVAO);
    unsigned int aabbVBO;
    glGenBuffers(1, &aabbVBO);

    glBindVertexArray(aabbVAO);
    glBindBuffer(GL_ARRAY_BUFFER, aabbVBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(BBox), &aabb, GL_STATIC_DRAW);

    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
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

void Mesh::render(Shader& shader, bool points) {
    unsigned int diffuseNr = 1;
    unsigned int specularNr = 1;
    unsigned int normalNr = 1;
    for (unsigned int i = 0; i < textures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); // activate proper texture unit before binding
        // retrieve texture number (the N in diffuse_textureN)
        std::string number;
        std::string name = textures[i].type;
        if (name == "texture_diffuse")
            number = std::to_string(diffuseNr++);
        else if (name == "texture_specular")
            number = std::to_string(specularNr++);
        else if (name == "texture_normal")
            number = std::to_string(normalNr++);

        shader.setInt(("material." + name + number).c_str(), i);
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }
    glActiveTexture(GL_TEXTURE0);

    // draw mesh
    glBindVertexArray(VAO);
    if(points)
        glDrawElements(GL_POINTS, indices.size(), GL_UNSIGNED_INT, 0);
    else
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::renderAABB(Shader& shader) {
    shader.setVec3("color", glm::normalize(aabb.max - aabb.min));
    glBindVertexArray(aabbVAO);
    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);
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
    h.t = min(tmin, tzmin);

    return h;
}

glm::vec3 transformPoint(glm::vec3 p, glm::mat4 transform) {
    auto t = transform * glm::vec4(p.x, p.y, p.z, 1.0f);
    return glm::vec3(t.x, t.y, t.z);
}

Hit Mesh::intersect(Ray& r, glm::mat4 transform) {
    Hit h;
    if (!aabb.intersect(r, transform).hit)
        return h;
    glm::vec3 origin = r.origin.toGLM();
    glm::vec3 direction = r.direction.toGLM();
    for (unsigned int i = 0; i < indices.size(); i+=3) {
        glm::vec3 v1 = transformPoint(vertices[indices[i]].Position, transform);
        glm::vec3 v2 = transformPoint(vertices[indices[i+1]].Position, transform);
        glm::vec3 v3 = transformPoint(vertices[indices[i+2]].Position, transform);
        
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
        float v = invDet * glm::dot(direction,sXe1);

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