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
        this->min = Vec3(0.0f);
        this->max = Vec3(0.0f);
    }
    __device__ __host__ AABB(const AABB& a, const AABB& b) {
        min.x = fmin(a.min.x, b.min.x);
        min.y = fmin(a.min.y, b.min.y);
        min.z = fmin(a.min.z, b.min.z);
        max.x = fmax(a.max.x, b.max.x);
        max.y = fmax(a.max.y, b.max.y);
        max.z = fmax(a.max.z, b.max.z);
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

