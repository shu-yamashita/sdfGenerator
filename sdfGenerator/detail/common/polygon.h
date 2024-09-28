#ifndef INCLUDED_SDFGENERATOR_DETAIL_COMMON_POLYGON_H
#define INCLUDED_SDFGENERATOR_DETAIL_COMMON_POLYGON_H

#include <utility>
#include <sdfGenerator/detail/common/vector.h>


namespace sdfGenerator
{
namespace detail
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
	vector<T, Dim>& vertex( const size_t index );

	SDFGENERATOR_CUDA_HOST_DEVICE
	const vector<T, Dim>& vertex( const size_t index ) const;

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& center();

	SDFGENERATOR_CUDA_HOST_DEVICE
	const vector<T, Dim>& center() const;

	SDFGENERATOR_CUDA_HOST_DEVICE
	vector<T, Dim>& normal();

	SDFGENERATOR_CUDA_HOST_DEVICE
	const vector<T, Dim>& normal() const;

	template <typename... Args>
	SDFGENERATOR_CUDA_HOST_DEVICE
	void set_vertices( const Args&... vertices );

	SDFGENERATOR_CUDA_HOST_DEVICE
	void set_center( const vector<T, Dim>& center );

	SDFGENERATOR_CUDA_HOST_DEVICE
	void set_normal( const vector<T, Dim>& normal );
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
std::pair<size_t, T> find_longest_edge_triangle( const polygon<T, 3, Dim>& tri );


} // namespace common 
} // namespace detail
} // namespace sdfGenerator


#include <sdfGenerator/detail/common/polygon.inl>

#endif // INCLUDED_SDFGENERATOR_DETAIL_COMMON_POLYGON_H
