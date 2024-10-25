#ifndef INCLUDED_SDFGENERATOR_CALC_SDF_H
#define INCLUDED_SDFGENERATOR_CALC_SDF_H


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
// void calc_sdf( STL );


} // namespace gpu
#endif // __NVCC__


} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_CALC_SDF_H
