#ifndef INCLUDED_SDFGENERATOR_STL_TO_SDF_H
#define INCLUDED_SDFGENERATOR_STL_TO_SDF_H


#include <sdfGenerator/STL.h>


namespace sdfGenerator
{

namespace cpu
{


} // namespace cpu


#ifdef __NVCC__
namespace gpu
{


// * Compute signed distance function from STL.
// * All pointers (T*) are the host.
// * Parameters:
//   - sdf: signed distance function.
//   - x, y, z: coordinate at each node.
//   - N_node: number of nodes.
//   - stl: STL data.
template <typename T>
void stl_to_sdf(
		T* sdf,
		const T* x,
		const T* y,
		const T* z,
		size_t N_node,
		const STL& stl);


} // namespace gpu
#endif // __NVCC__


} // namespace sdfGenerator


#include <sdfGenerator/detail/stl_to_sdf.inl>


#endif // INCLUDED_SDFGENERATOR_STL_TO_SDF_H
