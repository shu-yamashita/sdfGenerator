#ifndef INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL
#define INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL


#include <sdfGenerator/detail/internal/stl/stl_to_sdf.h>


namespace sdfGenerator
{
namespace internal
{
namespace stl
{


#ifdef __NVCC__
namespace gpu
{


// * Compute signed distance function from STL.
// * All pointers (T*) are the host pointers.
//   h_ means host pointer.
// * Parameters:
//   - sdf: signed distance function.
//   - x, y, z: coordinate at each position.
//   - N: number of positions.
//   - stl: STL data.
template <typename T>
void stl_to_sdf_host_ptr(
		T* h_sdf,
		const T* h_x,
		const T* h_y,
		const T* h_z,
		const size_t N,
		const STL& stl )
{
	// Arrays on device.
	T* d_sdf;
	T* d_x;
	T* d_y;
	T* d_z;

	// Allocate device memory.
	cudaMalloc( &d_sdf, sizeof(T) * N );
	cudaMalloc( &d_x  , sizeof(T) * N );
	cudaMalloc( &d_y  , sizeof(T) * N );
	cudaMalloc( &d_z  , sizeof(T) * N );

	// Copy data from host to device.
	cudaMemcpy( d_sdf, h_sdf, sizeof(T) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( d_x  , h_x  , sizeof(T) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( d_y  , h_y  , sizeof(T) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( d_z  , h_z  , sizeof(T) * N, cudaMemcpyHostToDevice );

	// Compute signed distance function.
	stl_to_sdf_device_ptr( d_sdf, d_x, d_y, d_z, N, stl );

	// Copy data from device to host.
	cudaMemcpy( h_sdf, d_sdf, sizeof(T) * N, cudaMemcpyDeviceToHost );

	// Deallocate device memoery.
	cudaFree( d_sdf );
	cudaFree( d_x );
	cudaFree( d_y );
	cudaFree( d_z );
}


// * Compute signed distance function from STL.
// * All pointers (T*) are the device pointers.
//   d_ means device pointer.
// * Parameters:
//   - sdf: signed distance function.
//   - x, y, z: coordinate at each position.
//   - N: number of positions.
//   - stl: STL data.
template <typename T>
void stl_to_sdf_device_ptr(
		T* d_sdf,
		const T* d_x,
		const T* d_y,
		const T* d_z,
		const size_t N,
		const STL& stl )
{
	// Polygon data on host.
	const STL::triangle* h_triangles = stl.getPolygonData();
	const size_t num_tri = stl.getNumPolygon();

	// Polygon data on device.
	STL::triangle* d_triangles;
	cudaMalloc( &d_triangles, sizeof( STL::triangle ) * num_tri );
	cudaMemcpy( d_triangles, h_triangles, sizeof( STL::triangle ) * num_tri, cudaMemcpyHostToDevice );

	// TODO
	// call kernel function

	cudaFree( d_triangles );
}


} // namespace gpu
#endif // __NVCC__


} // namespace stl
} // namespace internal
} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL
