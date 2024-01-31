#pragma once
#include "Vec3.h"
#include "Ray.h"

class AABB {
public:
    Vec3 min, max;
    __device__ __host__ AABB(const Vec3& min, const Vec3& max) {
        this->min = min;
        this->max = max;
    }
    __device__ __host__ AABB() {
        this->min = Vec3(-100.0f);
        this->max = Vec3(-100.0f);
    }
    __device__ __host__ AABB(const AABB& a, const AABB& b) {
        this->min.x = fminf(a.min.x, b.min.x);
        this->min.y = fminf(a.min.y, b.min.y);
        this->min.z = fminf(a.min.z, b.min.z);
        this->max.x = fmaxf(a.max.x, b.max.x);
        this->max.y = fmaxf(a.max.y, b.max.y);
        this->max.z = fmaxf(a.max.z, b.max.z);
    }

    __device__ bool hit(const Ray& r, float tmin, float tmax) const {
        for (int a = 0; a < 3; a++) {
            auto invD = r.invDir[a];
            auto orig = r.origin[a];
            float t0, t1;

            if (invD >= 0) {
                t0 = (min[a] - orig) * invD;
                t1 = (max[a] - orig) * invD;
            }
            else {
                t0 = (max[a] - orig) * invD;
                t1 = (min[a] - orig) * invD;
            }

            if (t0 > tmin) tmin = t0;
            if (t1 < tmax) tmax = t1;

            if (tmax < tmin)
                return false;
        }
        return true;
    }

};

