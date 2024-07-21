
#ifndef INCLUDED_SDFGENERATOR_INTERNAL_STRUCT_H
#define INCLUDED_SDFGENERATOR_INTERNAL_STRUCT_H

namespace sdfGenerator
{

struct vec3d
{
	float elems_[3];
	__host__ __device__       float& operator[]( const int axis )       { return elems_[axis]; }
	__host__ __device__ const float& operator[]( const int axis ) const { return elems_[axis]; }
};


struct polygon
{
	// Three vertices
	vec3d vertex[3];

	// Center of polygon.
	// Computed as the average of three vertices.
	vec3d center;

	// Normal vector.
	vec3d normal;

	// Params:
	//   * v: Coordinate of three vertices.
	//   * n: Normal vector
	void setter( const vec3d v[3], const vec3d& n )
	{
		for (int i_v = 0; i_v < 3; i_v++) vertex[i_v] = v[i_v];
		for (int axis = 0; axis < 3; axis++) center[axis] = ( v[0][axis] + v[1][axis] + v[2][axis] ) / 3;
		normal = n;
	}
};

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_INTERNAL_STRUCT_H

