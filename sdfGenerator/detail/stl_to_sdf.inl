#ifndef INCLUDED_SDFGENERATOR_DETAIL_STL_TO_SDF_INL
#define INCLUDED_SDFGENERATOR_DETAIL_STL_TO_SDF_INL


#include <sdfGenerator/STL.h>
#include <sdfGenerator/stl_to_sdf.h>


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
//   - x, y, z: coordinate at each position.
//   - N: number of positions.
//   - stl: STL data.
template <typename T>
void stl_to_sdf(
		T* sdf,
		const T* x,
		const T* y,
		const T* z,
		size_t N,
		const STL& stl)
{
	// Check that all pointers are on host.

	// Define and allocate device ptr.
}


} // namespace gpu
#endif // __NVCC__


} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_STL_TO_SDF_INL
