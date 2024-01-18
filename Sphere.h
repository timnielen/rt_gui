#pragma once
#include "Hit.h"
#include "Material.h"

class Sphere : public Hitable {
public:
	Point3 center;
	float radius;
    Material* mat;
	__device__ __host__ Sphere(const Point3& center, const float& radius, Material* mat) : center(center), radius(radius), mat(mat) {
        Vec3 rvec = Vec3(radius);
        if (radius < 0.0f)
            rvec = -rvec;
        aabb = AABB(center - rvec, center + rvec);
    }

    __device__ bool hit(const Ray& r, float ray_tmin, float ray_tmax, HitRecord& rec) const override {
        Vec3 oc = r.origin - center;
        auto a = r.direction.length_squared();
        auto half_b = dot(oc, r.direction);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (-half_b + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        return true;
    }
};