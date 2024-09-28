#ifndef INCLUDED_SDFGENERATOR_DETAIL_COMMON_POLYGON_INL
#define INCLUDED_SDFGENERATOR_DETAIL_COMMON_POLYGON_INL

#include <algorithm>
#include <utility>
#include <vector>
#include <sdfGenerator/detail/common/polygon.h>


namespace sdfGenerator
{
namespace detail
{
namespace common
{


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim>& polygon<T, N, Dim>::vertex( const size_t index )
{
	assert( index < N );
	return vertex_[index];
}


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
const vector<T, Dim>& polygon<T, N, Dim>::vertex( const size_t index ) const
{
	assert( index < N );
	return vertex_[index];
}


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim>& polygon<T, N, Dim>::center() { return center_; }


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
const vector<T, Dim>& polygon<T, N, Dim>::center() const { return center_; }


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
vector<T, Dim>& polygon<T, N, Dim>::normal() { return normal_; }


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
const vector<T, Dim>& polygon<T, N, Dim>::normal() const { return normal_; }


template <typename T, size_t N, size_t Dim>
template <typename... Args>
SDFGENERATOR_CUDA_HOST_DEVICE
void polygon<T, N, Dim>::set_vertices( const Args&... vertices )
{
	static_assert( sizeof...(vertices) == N, "Please set N vertices for N-sided polygon." );

	size_t index = 0;
	for ( const auto& v : std::initializer_list< vector<T, Dim> >{ vertices... } )
	{
		vertex(index) = v;
		index++;
	}
}


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
void polygon<T, N, Dim>::set_center( const vector<T, Dim>& center ) { center_ = center; }


template <typename T, size_t N, size_t Dim>
SDFGENERATOR_CUDA_HOST_DEVICE
void polygon<T, N, Dim>::set_normal( const vector<T, Dim>& normal ) { normal_ = normal; }




// * Return id and length of the longest edge.
//         0
//        /|
// len=2 / |
//      /  | len=sqrt(3)  -> return (1, 2.0)
//     /   |
//    2----1
//    len=1.0
template <typename T, size_t Dim>
std::pair<size_t, T> find_longest_edge_triangle( const polygon<T, 3, Dim>& tri )
{
	std::vector< std::pair<T, size_t> > len_id;
	len_id.emplace_back( calc_distance( tri.vertex(1), tri.vertex(2) ), 0 );
	len_id.emplace_back( calc_distance( tri.vertex(2), tri.vertex(0) ), 1 );
	len_id.emplace_back( calc_distance( tri.vertex(0), tri.vertex(1) ), 2 );

	std::sort( len_id.rbegin(), len_id.rend() );
	return std::make_pair(len_id[0].second, len_id[0].first);
}


} // namespace common 
} // namespace detail
} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_DETAIL_COMMON_POLYGON_INL
