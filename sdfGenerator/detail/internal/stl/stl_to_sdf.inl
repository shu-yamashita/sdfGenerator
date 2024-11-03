#ifndef INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL
#define INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL


#include <sdfGenerator/detail/internal/polygon.h>
#include <sdfGenerator/detail/internal/stl/stl_to_sdf.h>
#include <sdfGenerator/detail/internal/vector.h>


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
	sdfGenerator::internal::vector<float, 3> here;
	here[0] = x; here[1] = y; here[2] = z;

	// The number of recorded nearest triangles.
	constexpr size_t num_memo = 8;

	// Nearest triangles info.
	long long nearest_id[num_memo];
	float nearest_dist[num_memo];
	for (size_t i = 0; i < num_memo; i++)
	{
		nearest_id[i] = -1;
		nearest_dist[i] = 1e10;
	}

	// Search all triangles
	for (size_t id_tri = 0; id_tri < N_tri; id_tri++)
	{
		const float dist_now = calc_distance( triangles[id_tri].center(), here );

		// Update nearest triangles info.
		for (size_t i_memo = 0; i_memo < num_memo; i_memo++)
		{
			if ( nearest_id[i_memo] == -1 || dist_now < nearest_dist[i_memo] )
			{
				for (size_t j_memo = num_memo - 1; j_memo > i_memo; j_memo--)
				{
					nearest_id[j_memo] = nearest_id[j_memo - 1];
					nearest_dist[j_memo] = nearest_dist[j_memo - 1];
				}
				nearest_id[i_memo] = id_tri;
				nearest_dist[i_memo] = dist_now;
				break;
			}
		}
	}

	// sum of inner product ( (unit vector to triangle) * (normal vector of triangle) ).
	float sum_inner_prod = 0.0;
	for (size_t i_memo = 0; i_memo < num_memo; i_memo++)
	{
		float inner_prod = 0.0;
		const long long id_tri = nearest_id[i_memo];
		const float dist_tri = nearest_dist[i_memo];
		if ( id_tri == -1 ) break;
		for (size_t axis = 0; axis < 3; axis++)
		{
			inner_prod += (triangles[id_tri].center()[axis] - here[axis]) * triangles[id_tri].normal()[axis];
		}
		inner_prod /= dist_tri;
		sum_inner_prod += inner_prod;
	}

	const float sign = ( sum_inner_prod > 0.0 ? 1.0 : -1.0 );
	return sign * nearest_dist[0];
}


} // namespace gpu
#endif // __NVCC__


} // namespace stl
} // namespace internal
} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_INL
