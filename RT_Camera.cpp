#include "RT_Camera.h"
#include "Vec3.h"


__device__ __host__ void RT_Camera::update(const int& x, const int& y) {
	auto focal_length = 1.0;
	auto viewport_height = 2.0;
	auto viewport_width = viewport_height * (static_cast<float>(x) / y);
	camera_center = Point3(0, 0, 0);

	// Calculate the vectors across the horizontal and down the vertical viewport edges.
	viewport_u = Vec3(viewport_width, 0, 0);
	viewport_v = Vec3(0, -viewport_height, 0);

	// Calculate the horizontal and vertical delta vectors from pixel to pixel.
	pixel_delta_u = viewport_u / x;
	pixel_delta_v = viewport_v / y;

	// Calculate the location of the upper left pixel.
	auto viewport_upper_left = camera_center
		- Vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
	pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
}

