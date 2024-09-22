#ifndef INCLUDED_SDFGENERATOR_COMMON_POLYGON_H
#define INCLUDED_SDFGENERATOR_COMMON_POLYGON_H

#include <algorithm>
#include <utility>
#include <vector>
#include <sdfGenerator/common/vector.h>

namespace sdfGenerator
{
namespace common
{


// * N-sided polygon in Dim-dimensional space.
template <typename T, size_t N, size_t Dim>
struct polygon
{
private:
	vector<T, Dim> vertex_[N], center_, normal_;
public:
	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& vertex( const size_t index )
	{
		assert( index < N );
		return vertex_[index];
	}
	SDFGENERATOR_CUDA_HOST_DEVICE
	const vector<T, Dim>& vertex( const size_t index ) const
	{
		assert( index < N );
		return vertex_[index];
	}

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& center() { return center_; }
	SDFGENERATOR_CUDA_HOST_DEVICE
	const vector<T, Dim>& center() const { return center_; }

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& normal() { return normal_; }
	SDFGENERATOR_CUDA_HOST_DEVICE
	const vector<T, Dim>& normal() const { return normal_; }

	template <typename... Args>
	SDFGENERATOR_CUDA_HOST_DEVICE
	void set_vertices( const Args&... vertices )
	{
		static_assert( sizeof...(vertices) == N, "Please set N vertices for N-sided polygon." );

		size_t index = 0;
		for ( const auto& v : std::initializer_list< vector<T, Dim> >{ vertices... } )
		{
			vertex(index) = v;
			index++;
		}
	}

	SDFGENERATOR_CUDA_HOST_DEVICE
	void set_center( const vector<T, Dim>& center ) { center_ = center; }
	SDFGENERATOR_CUDA_HOST_DEVICE
	void set_normal( const vector<T, Dim>& normal ) { normal_ = normal; }
};


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
} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_COMMON_POLYGON_H
