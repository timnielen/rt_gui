#include "Material.h"
#define PI 3.14159265359f
__device__ bool MultiMaterial::scatter(
	const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {

	Vec3 normal = rec.normal;
	if (textures[textureTypeNormal] != -1) {
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeNormal], rec.uvCoords.x, rec.uvCoords.y);
		normal = Vec3(col.x, col.y, col.z);
		normal = 2 * normal - 1;
		Vec3 a = Vec3(rec.tangent.x, rec.bitangent.x, rec.normal.x);
		Vec3 b = Vec3(rec.tangent.y, rec.bitangent.y, rec.normal.y);
		Vec3 c = Vec3(rec.tangent.z, rec.bitangent.z, rec.normal.z);
		normal = d_normalize(Vec3(dot(a, normal), dot(b, normal), dot(c, normal)));
	}

	if (textures[textureTypeDiffuse] == -1)
		attenuation = colors[textureTypeDiffuse];
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeDiffuse], rec.uvCoords.x, rec.uvCoords.y);
		//printf("%d\n", cudaTextures[textureTypeDiffuse].texObject);
		attenuation = Vec3(col.x, col.y, col.z);
	}

	float specular;
	if (textures[textureTypeSpecular] == -1)
		specular = colors[textureTypeSpecular].x;
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeSpecular], rec.uvCoords.x, rec.uvCoords.y);
		//printf("%d\n", cudaTextures[textureTypeDiffuse].texObject);
		specular = col.x;
	}
	float roughness;
	if (textures[textureTypeRoughness] == -1)
		roughness = colors[textureTypeRoughness].x;
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeRoughness], rec.uvCoords.x, rec.uvCoords.y);
		//printf("%d\n", cudaTextures[textureTypeDiffuse].texObject);
		roughness = col.x;
	}

	if(curand_uniform(local_rand_state) < specular)
	{
		Vec3 reflected = reflect(d_normalize(r_in.direction), normal);
		scattered = Ray(rec.p, reflected + roughness * random_unit_vector(local_rand_state));
		//attenuation *= dot(scattered.direction, normal);
		return dot(scattered.direction, normal) > 0;
	}
	else {
		Vec3 dir = random_on_hemisphere(normal, local_rand_state);
		// Catch degenerate scatter direction
		if (dir.near_zero())
			dir = normal;
		auto cos_theta = dot(normal, d_normalize(dir));
		//attenuation *= cos_theta < 0 ? 0 : cos_theta / PI;
		scattered = Ray(rec.p, dir);
		return true;
	}	
}