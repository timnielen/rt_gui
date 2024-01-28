#include "Material.h"

__device__ bool MultiMaterial::scatter(
	const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
	Vec3 dir = rec.normal + random_on_hemisphere(rec.normal, local_rand_state);

	// Catch degenerate scatter direction
	if (dir.near_zero())
		dir = rec.normal;


	if (textures[textureTypeDiffuse] == -1)
		attenuation = colors[textureTypeDiffuse];
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeDiffuse], rec.uvCoords.x, rec.uvCoords.y);
		//printf("%d\n", cudaTextures[textureTypeDiffuse].texObject);
		attenuation = Vec3(col.x, col.y, col.z);
	}
	scattered = Ray(rec.p, dir);
	return true;
}