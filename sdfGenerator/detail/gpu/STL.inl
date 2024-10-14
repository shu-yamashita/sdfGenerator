#ifndef INCLUDED_SDFGENERATOR_DETAIL_GPU_STL_INL
#define INCLUDED_SDFGENERATOR_DETAIL_GPU_STL_INL

#include <sdfGenerator/gpu/STL.h>

namespace sdfGenerator
{
namespace gpu
{


template <typename T>
void STL::calcSDF( T* sdf, const size_t (&num_cell)[3], const T* const (&coord)[3] ) const
{
	const size_t num_cell_total = num_cell[0] * num_cell[1] * num_cell[2];

	// Arrays on device.
	T* d_sdf;
	triangle* d_triangles;
	T* d_coord[3];

	// Allocate memory on device.
	cudaMalloc( &d_sdf      , sizeof(T       ) * num_cell_total    );
	cudaMalloc( &d_triangles, sizeof(triangle) * triangles_.size() );
	for (size_t i = 0; i < 3; i++) cudaMalloc( &d_coord[i], sizeof(T) * num_cell[i] );

	// Copy data from host to device.
	cudaMemcpy( d_triangles, triangles_.data(), sizeof(triangle) * triangles_.size(), cudaMemcpyHostToDevice );
	for (size_t i = 0; i < 3; i++) cudamemcpy( d_coord[i], coord[i], sizeof(T) * num_cell[i], cudaMemcpyHostToDevice );

	// Calculate sdf on device.
	{
		const dim3 block(8, 8, 8);
		const dim3 grid(
				(num_cell[0] + block.x - 1) / block.x,
				(num_cell[1] + block.y - 1) / block.y,
				(num_cell[2] + block.z - 1) / block.z );

		// TODO: call kernel. Inside kernel, call common function.
		//calcSDF<<<grid, block>>>( d_sdf, d_triangles, d_coord[0], d_coord[1], d_coord[2], num_cell[0], num_cell[1], num_cell[2], polys_.size() );

		cudaDeviceSynchronize();
	}

	// Copy data from device to host.
	cudaMemcpy( sdf, d_sdf, sizeof(T) * num_cell_total, cudaMemcpyDeviceToHost );

	// Free memory on device.
	cudaFree( d_sdf       );
	cudaFree( d_triangles );
	for (size_t i = 0; i < 3; i++) cudaFree( d_coord[i] );
}


} // namespace gpu
} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_GPU_STL_INL
