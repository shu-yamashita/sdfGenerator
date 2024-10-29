#ifndef INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_H
#define INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_H


#include <sdfGenerator/STL.h>


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
//   - N_node: number of nodes.
//   - stl: STL data.
template <typename T>
void stl_to_sdf_host_ptr(
		T* h_sdf,
		const T* h_x,
		const T* h_y,
		const T* h_z,
		const size_t N_node,
		const STL& stl );


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
		const STL& stl );


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
		const size_t N_tri );


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
		const size_t N_tri );


} // namespace gpu
#endif // __NVCC__


} // namespace stl
} // namespace internal
} // namespace sdfGenerator


#include <sdfGenerator/detail/internal/stl/stl_to_sdf.inl>

#endif // INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_STL_STL_TO_SDF_H
