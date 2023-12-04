#pragma once
#include "Vec3.h"
struct Hit {
    bool hit = false;
    float t = 0.0f;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    Vec3 invDir;
    Ray(const glm::vec3 o, const glm::vec3 dir) : origin(o.x, o.y, o.z), direction(dir.x, dir.y, dir.z) {
        invDir = 1.0f / direction;
    }
    Ray(const Vec3& o, const Vec3& dir) : origin(o), direction(dir) {
        invDir = 1.0f / direction;
    }
};