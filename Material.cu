#include "Material.h"
#define PI 3.14159265359f
#define fracPI 0.31830988618f

__device__ Vec3 random_cosine_direction(curandState* local_rand_state) {
	float r1 = curand_uniform(local_rand_state);
	float r2 = curand_uniform(local_rand_state);

	float phi = 2 * PI * r1;
	float x = cosf(phi) * sqrtf(r2);
	float y = sinf(phi) * sqrtf(r2);
	float z = sqrtf(1 - r2);

	return Vec3(x, y, z);
}

__device__ bool MultiMaterial::scatter(
	const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state, bool& skip_pdf) const {

	Vec3 normal = rec.normal;
	if (textures[textureTypeNormal] != -1) {
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeNormal], rec.uvCoords.x, rec.uvCoords.y);
		normal = Vec3(col.x, col.y, col.z);
		normal = 2 * normal - 1;
		normal = d_normalize(rec.tangent * normal.x + rec.bitangent * normal.y + rec.normal * normal.z);
	}

	if (textures[textureTypeDiffuse] == -1)
		attenuation = colors[textureTypeDiffuse];
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeDiffuse], rec.uvCoords.x, rec.uvCoords.y);
		attenuation = Vec3(col.x, col.y, col.z);
	}

	float specular;
	if (textures[textureTypeSpecular] == -1)
		specular = colors[textureTypeSpecular].x;
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeSpecular], rec.uvCoords.x, rec.uvCoords.y);
		specular = col.x;
	}
	float roughness;
	if (textures[textureTypeRoughness] == -1)
		roughness = colors[textureTypeRoughness].x;
	else
	{
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeRoughness], rec.uvCoords.x, rec.uvCoords.y);
		roughness = col.x;
	}

	if(curand_uniform(local_rand_state) < specular) 
	//if(0)
	{
		Vec3 reflected = reflect(d_normalize(r_in.direction), normal);
		scattered = Ray(rec.p, reflected + roughness * random_unit_vector(local_rand_state));
		//attenuation *= dot(scattered.direction, normal);
		skip_pdf = true;
		return dot(scattered.direction, normal) > 0;
	}
	else {
		//Vec3 dir = normal + random_unit_vector(local_rand_state);
		//// Catch degenerate scatter direction
		//if (dir.near_zero())
		//	dir = normal;
		
		/*Vec3 a = (fabsf(normal.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
		Vec3 tangent = d_normalize(cross(normal, a));
		Vec3 cotangent = cross(normal, tangent);

		Vec3 dir = random_cosine_direction(local_rand_state);
		dir = dir.x * tangent + dir.y * cotangent + dir.z * normal;*/

		

		Vec3 dir = normal + random_unit_vector(local_rand_state);
		if (dir.near_zero())
			dir = normal;

		skip_pdf = false;
		scattered = Ray(rec.p, dir);
		return true;
	}	
}

__device__ float MultiMaterial::pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) {
	Vec3 normal = rec.normal;
	if (textures[textureTypeNormal] != -1) {
		float4 col = tex2D<float4>((cudaTextureObject_t)textures[textureTypeNormal], rec.uvCoords.x, rec.uvCoords.y);
		normal = Vec3(col.x, col.y, col.z);
		normal = 2 * normal - 1;
		normal = d_normalize(rec.tangent * normal.x + rec.bitangent * normal.y + rec.normal * normal.z);
	}

	return fmaxf(0.0f, dot(normal, d_normalize(scattered.direction)) / PI);
}