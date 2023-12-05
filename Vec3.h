#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#include "glm/glm.hpp"

class Vec3
{
public:
	float x, y, z;
    operator glm::vec3() {
        return glm::vec3(x, y, z);
    }
	__host__ __device__ Vec3() : x(0), y(0), z(0) {};
	__host__ __device__ Vec3(const float& x, const float& y, const float& z) : x(x), y(y), z(z) {};
    __host__ glm::vec3 toGLM() const {
        return glm::vec3(x, y, z);
    }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ float operator[](int i) const {
        switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        }
    }
    __host__ __device__ float& operator[](int i) {
        switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        }
    }

    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }


    __host__ __device__ Vec3& operator*=(float t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __host__ __device__ Vec3& operator/=(float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return x*x + y*y + z*z;
    }

    __host__ __device__ float4 toColor() const {
        return make_float4(x, y, z, 1.0f);
    }
};

using Point3 = Vec3;


// Vector Utility Functions

inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

__host__ __device__ inline Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}



__host__ __device__ inline Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ inline Vec3 operator+(float t, const Vec3& v) {
    return Vec3(t + v.x, t + v.y, t + v.z);
}

__host__ __device__ inline Vec3 operator-(float t, const Vec3& v) {
    return Vec3(t - v.x, t - v.y, t - v.z);
}

__host__ __device__ inline Vec3 operator+(const Vec3& v, float t) {
    return Vec3(t + v.x, t + v.y, t + v.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3& v, float t) {
    return Vec3(v.x-t, v.y-t, v.z-t);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return (1 / t) * v;
}

__host__ __device__ inline Vec3 operator/(float t, Vec3 v) {
    return Vec3(t/v.x, t/v.y, t/v.z);
}

__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) {
    return u.x * v.x
        + u.y * v.y
        + u.z * v.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x);
}

__host__ __device__ inline Vec3 normalize(Vec3 v) {
    return v / v.length();
}

