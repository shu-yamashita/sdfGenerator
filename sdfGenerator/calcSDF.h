#ifndef INCLUDED_SDFGENERATOR_CALCSDF_H
#define INCLUDED_SDFGENERATOR_CALCSDF_H


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
// void calcSDF( STL );


} // namespace gpu
#endif // __NVCC__


} // namespace sdfGenerator

#endif
