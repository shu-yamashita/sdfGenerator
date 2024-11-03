#ifndef INCLUDED_SDFGENERATOR_STL_H
#define INCLUDED_SDFGENERATOR_STL_H

#include <sdfGenerator/detail/internal/polygon.h>

namespace sdfGenerator
{


class STL
{
public:
	using triangle = sdfGenerator::internal::polygon<float, 3, 3>;

	// * Read STL file and store the polygon data.
	void readSTL( const char* file_path );
	template <typename T>
	void readSTL( const char* file_path, const T (&offset)[3] );
	template <typename T>
	void readSTL( const char* file_path, const T scale_factor );
	template <typename T1, typename T2>
	void readSTL( const char* file_path, const T1 (&offset)[3], const T2 scale_factor );

	// * Return the number of polygons.
	size_t getNumPolygon() const;

	// * Return raw pointer.
	triangle* getPolygonData();
	const triangle* getPolygonData() const;

	// * Refine polygons until the maximum edge length of all polygons is
	//   less than or equal to the threshold.
	template <typename T>
	void refinePolygon( const T threshold );

protected:
	// Polygon data.
	std::vector<triangle> triangles_;

private:
	void refine_and_push( const triangle& tri, const float threshold );
};


} // namespace sdfGenerator


#include <sdfGenerator/detail/STL.inl>


#endif // INCLUDED_SDFGENERATOR_STL_H
