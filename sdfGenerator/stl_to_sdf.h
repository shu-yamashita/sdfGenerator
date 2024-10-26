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


// TODO
//void stl_to_sdf();


} // namespace gpu
#endif // __NVCC__


} // namespace sdfGenerator


#include <sdfGenerator/detail/stl_to_sdf.inl>


#endif // INCLUDED_SDFGENERATOR_STL_TO_SDF_H
