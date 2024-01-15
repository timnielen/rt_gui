#include "Hit.h"

__device__ unsigned int morton3D(Vec3 v)
{
	float x = fmin(fmax(v.x * 1024.0f, 0.0f), 1023.0f);
	float y = fmin(fmax(v.y * 1024.0f, 0.0f), 1023.0f);
	float z = fmin(fmax(v.z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}
