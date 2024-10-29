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
//   - x, y, z: coordinate at each node.
//   - N_node: number of node.
//   - stl: STL data.
template <typename T>
void stl_to_sdf_host_ptr(
		T* h_sdf,
		const T* h_x,
		const T* h_y,
		const T* h_z,
		const size_t N_node,
		const STL& stl )
{
	// Arrays on device.
	T* d_sdf;
	T* d_x;
	T* d_y;
	T* d_z;

	// Allocate device memory.
	cudaMalloc( &d_sdf, sizeof(T) * N_node );
	cudaMalloc( &d_x  , sizeof(T) * N_node );
	cudaMalloc( &d_y  , sizeof(T) * N_node );
	cudaMalloc( &d_z  , sizeof(T) * N_node );

	// Copy data from host to device.
	cudaMemcpy( d_sdf, h_sdf, sizeof(T) * N_node, cudaMemcpyHostToDevice );
	cudaMemcpy( d_x  , h_x  , sizeof(T) * N_node, cudaMemcpyHostToDevice );
	cudaMemcpy( d_y  , h_y  , sizeof(T) * N_node, cudaMemcpyHostToDevice );
	cudaMemcpy( d_z  , h_z  , sizeof(T) * N_node, cudaMemcpyHostToDevice );

	// Compute signed distance function.
	stl_to_sdf_device_ptr( d_sdf, d_x, d_y, d_z, N_node, stl );

	// Copy data from device to host.
	cudaMemcpy( h_sdf, d_sdf, sizeof(T) * N_node, cudaMemcpyDeviceToHost );

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
//   - x, y, z: coordinate at each node.
//   - N_node: number of nodes.
//   - stl: STL data.
template <typename T>
void stl_to_sdf_device_ptr(
		T* d_sdf,
		const T* d_x,
		const T* d_y,
		const T* d_z,
		const size_t N_node,
		const STL& stl )
{
	// Polygon data on host.
	const STL::triangle* h_triangles = stl.getPolygonData();
	const size_t N_tri = stl.getNumPolygon();

	// Polygon data on device.
	STL::triangle* d_triangles;
	cudaMalloc( &d_triangles, sizeof( STL::triangle ) * N_tri );
	cudaMemcpy( d_triangles, h_triangles, sizeof( STL::triangle ) * N_tri, cudaMemcpyHostToDevice );

	// Call kernel to compute sdf.
	const size_t thread_per_block = 32;
	const dim3 block(thread_per_block, 1, 1);
	const dim3 grid( (N_node + thread_per_block - 1) / thread_per_block, 1, 1 );
	stl_to_sdf_kernel<<<grid, block>>>( d_sdf, d_x, d_y, d_z, N_node, d_triangles, N_tri );
	cudaDeviceSynchronize(); // TODO: CUDA_SAFE_CALL

	cudaFree( d_triangles );
}


// * Compute signed distance from the surface at all nodes.
// * Threads are assigned in 1D.
// * Parameters:
//   - sdf: signed distance function.
//   - x, y, z: coordinates at nodes.
//   - N_node: number of nodes.
//   - triangles: STL triangle data.
//   - N_tri: number of triangles.
template <typename T>
__global__
void stl_to_sdf_kernel(
		T* sdf,
		const T* x,
		const T* y,
		const T* z,
		const size_t N_node,
		const STL::triangle* triangles,
		const size_t N_tri )
{
	const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if ( index >= N_node ) return;
	sdf[index] = compute_sdf_at_given_coord( x[index], y[index], z[index], triangles, N_tri );
}


// * Compute signed distance at a given coordinate
// * Parameters:
//   - x, y, z: coordinates.
//   - triangles: STL triangle data.
//   - N_tri: number of triangles.
__device__
inline float compute_sdf_at_given_coord(
		const float x,
		const float y,
		const float z,
		const STL::triangle* triangles,
		const size_t N_tri )
{
	// TODO: impl.
	return 0.0; // test
}


} // namespace gpu
#endif // __NVCC__


} // namespace stl
} // namespace internal
} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL
