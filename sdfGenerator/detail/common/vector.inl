#ifndef INCLUDED_SDFGENERATOR_DETAIL_COMMON_VECTOR_INL
#define INCLUDED_SDFGENERATOR_DETAIL_COMMON_VECTOR_INL

#include <cassert>
#include <cmath>
#include <sdfGenerator/detail/common/macro.h>

namespace sdfGenerator
{
namespace detail
{
namespace common
{

template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
T& vector<T, Dim>::operator[]( const size_t index )
{
	assert( index < Dim );
	return elem_[index];
}


template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
const T& vector<T, Dim>::operator[]( const size_t index ) const
{
	assert( index < Dim );
	return elem_[index];
}


template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim> vector<T, Dim>::operator+( const vector<T, Dim>& other ) const
{
	vector<T, Dim> ret;
	for (size_t i = 0; i < Dim; i++) ret[i] = elem_[i] + other[i];
	return ret;
}


template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim> vector<T, Dim>::operator-( const vector<T, Dim>& other ) const
{
	vector<T, Dim> ret;
	for (size_t i = 0; i < Dim; i++) ret[i] = elem_[i] - other[i];
	return ret;
}


template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
bool vector<T, Dim>::operator==( const vector<T, Dim>& other ) const
{
	bool ret = true;
	for (size_t i = 0; i < Dim; i++) ret &= ( elem_[i] == other[i] );
	return ret;
}


template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim>& vector<T, Dim>::operator+=( const vector<T, Dim>& other )
{
	for (size_t i = 0; i < Dim; i++) elem_[i] += other[i];
	return *this;
}


template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim>& vector<T, Dim>::operator-=( const vector<T, Dim>& other )
{
	for (size_t i = 0; i < Dim; i++) elem_[i] -= other[i];
	return *this;
}


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
} // namespace detail
} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_COMMON_VECTOR_INL
