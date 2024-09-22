#ifndef INCLUDED_SDFGENERATOR_COMMON_VECTOR_H
#define INCLUDED_SDFGENERATOR_COMMON_VECTOR_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <sdfGenerator/common/macro.h>

namespace sdfGenerator
{
namespace common
{


// * Vector in Dim-dimensional space.
template <typename T, size_t Dim>
struct vector
{
protected:
	T elem_[Dim];
public:
	SDFGENERATOR_CUDA_HOST_DEVICE
	vector(): elem_{} {}

	SDFGENERATOR_CUDA_HOST_DEVICE
	T& operator[]( const size_t index )
	{
		assert( index < Dim );
		return elem_[index];
	}

	SDFGENERATOR_CUDA_HOST_DEVICE
	const T& operator[]( const size_t index ) const
	{
		assert( index < Dim );
		return elem_[index];
	}
};


// * Calculate average vector in Dim-dimensional space.
// * Arguments must have type vector<T, Dim>.
template <typename T, size_t Dim, typename... Args>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim> calc_average_vector( const Args&... vectors )
{
	vector<T, Dim> ret;

	for ( const auto& vec : std::initializer_list< vector<T, Dim> >{ vectors... } )
		for ( size_t axis = 0; axis < Dim; axis++ ) ret[axis] += vec[axis];

	const size_t cnt_vector = sizeof...(vectors);
	for ( size_t axis = 0; axis < Dim; axis++ ) ret[axis] /= cnt_vector;

	return ret;
}


// * Calculate distance between two vertices in Dim-dimensional space.
template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
T calc_distance( const vector<T, Dim>& v0, const vector<T, Dim>& v1 )
{
	auto sq = [] ( T x ) { return x * x; };
	T ret = 0;
	for (size_t axis = 0; axis < Dim; axis++) ret += sq(v0[axis] - v1[axis]);
	return sqrt(ret);
}

} // namespace common 
} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_COMMON_VECTOR_H
