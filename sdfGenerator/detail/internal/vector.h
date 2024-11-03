#ifndef INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_VECTOR_H
#define INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_VECTOR_H

#include <cstddef>
#include <sdfGenerator/detail/internal/macro.h>

namespace sdfGenerator
{
namespace internal
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
	T& operator[]( const size_t index );

	SDFGENERATOR_CUDA_HOST_DEVICE
	const T& operator[]( const size_t index ) const;

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim> operator+( const vector<T, Dim>& other ) const;

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim> operator-( const vector<T, Dim>& other ) const;

	SDFGENERATOR_CUDA_HOST_DEVICE
	bool operator==( const vector<T, Dim>& other ) const;

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& operator+=( const vector<T, Dim>& other );

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& operator-=( const vector<T, Dim>& other );
};


// * Calculate average vector in Dim-dimensional space.
// * Arguments must have type vector<T, Dim>.
template <typename T, size_t Dim, typename... Args>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim> calc_average_vector( const Args&... vectors );


// * Calculate distance between two vertices in Dim-dimensional space.
template <typename T, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
T calc_distance( const vector<T, Dim>& v0, const vector<T, Dim>& v1 );


} // namespace internal
} // namespace sdfGenerator


#include <sdfGenerator/detail/internal/vector.inl>


#endif // INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_VECTOR_H
