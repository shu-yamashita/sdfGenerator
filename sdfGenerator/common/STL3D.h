#ifndef INCLUDED_SDFGENERATOR_COMMON_STL3D_H
#define INCLUDED_SDFGENERATOR_COMMON_STL3D_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sdfGenerator/common/logger.h>
#include <sdfGenerator/common/polygon.h>
#include <sdfGenerator/common/vector.h>

namespace sdfGenerator
{

namespace common
{

class STL3D
{
public:
	// * Read STL file and store the polygon data.
	void readSTL( const char* file_path );
	template <typename T>
	void readSTL( const char* file_path, const T (&offset)[3] );
	template <typename T>
	void readSTL( const char* file_path, const T scale_factor );
	template <typename T1, typename T2>
	void readSTL( const char* file_path, const T1 (&offset)[3], const T2 scale_factor );

	// * Refine polygons until the maximum edge length of all polygons is
	//   less than or equal to the threshold.
	template <typename T>
	void refinePolygon( const T threshold );
protected:
	std::vector< polygon<float, 3, 3> > polys_;
private:
	void refine_and_push( const polygon<float, 3, 3>& poly, const float threshold );
};


inline void STL3D::readSTL( const char* file_path )
{
	const float offset[3] = { 0.0, 0.0, 0.0 };
	const float scale_factor = 1.0;
	readSTL( file_path, offset, scale_factor );
}


template <typename T>
void STL3D::readSTL( const char* file_path, const T (&offset)[3] )
{
	const float scale_factor = 1.0;
	readSTL( file_path, offset, scale_factor );
}


template <typename T>
void STL3D::readSTL( const char* file_path, const T scale_factor )
{
	const float offset[3] = { 0.0, 0.0, 0.0 };
	readSTL( file_path, offset, scale_factor );
}


template <typename T1, typename T2>
void STL3D::readSTL( const char* file_path, const T1 (&offset)[3], const T2 scale_factor )
{
	processLogger( "readSTL" );

    std::ifstream ifs( file_path, std::ios::in | std::ios::binary );

    if(!ifs)
	{
		std::cerr << "Can not open STL file." << std::endl;
		std::exit(EXIT_FAILURE);
    }

	unsigned int num_polygons;

	// Read number of polygons.
	ifs.seekg(80, std::ios_base::beg);
	ifs.read( (char*) &num_polygons, sizeof(unsigned int) );

	// Read polygon data.
	for (unsigned int id_p = 0; id_p < num_polygons; id_p++)
	{
		vector<float, 3> vbuff[3], nbuff;

		// Read normal vector.
		for (size_t axis = 0; axis < 3; axis++) ifs.read( (char*) &(nbuff[axis]), sizeof(float) );

		// Read vertices.
		for (size_t id_v = 0; id_v < 3; id_v++)
		{
			for (size_t axis = 0; axis < 3; axis++) ifs.read( (char*) &(vbuff[id_v][axis]), sizeof(float) );
			for (size_t axis = 0; axis < 3; axis++) vbuff[id_v][axis] = vbuff[id_v][axis] * scale_factor + offset[axis];
		}

		polygon<float, 3, 3> tmp_polygon;
		tmp_polygon.set_vertices( vbuff[0], vbuff[1], vbuff[2] ); 
		tmp_polygon.set_center( calc_average_vector<float, 3>( vbuff[0], vbuff[1], vbuff[2] ) );
		tmp_polygon.set_normal( nbuff );

		polys_.push_back( tmp_polygon );

		ifs.seekg(2, std::ios_base::cur);
	}

	ifs.close();
}


template <typename T>
void STL3D::refinePolygon( const T threshold )
{
	processLogger( "refinePolygon" );

	std::vector< polygon<float, 3, 3> > polys_buff = polys_;
	polys_.clear();
	for ( const auto& each_poly : polys_buff ) refine_and_push( each_poly, threshold );
}


inline void STL3D::refine_and_push( const polygon<float, 3, 3>& poly, const float threshold )
{
	const std::pair<size_t, float> max_edge = find_longest_edge_triangle( poly );

	if ( max_edge.second <= threshold )
	{
		polys_.push_back( poly );
		return;
	}

	// The polygon is divided into two polygons.
	polygon<float, 3, 3> poly0, poly1;

	// The longest edge connects tmp_v0 and tmp_v1.
	const vector<float, 3> tmp_v0 = poly.vertex( (max_edge.first + 1) % 3 );
	const vector<float, 3> tmp_v1 = poly.vertex( (max_edge.first + 2) % 3 );

	const vector<float, 3> new_vertex = calc_average_vector<float, 3>( tmp_v0, tmp_v1 );

	poly0.set_vertices( poly.vertex(max_edge.first), tmp_v0, new_vertex );
	poly1.set_vertices( poly.vertex(max_edge.first), tmp_v1, new_vertex );
	poly0.set_center( calc_average_vector<float, 3>( poly0.vertex(0), poly0.vertex(1), poly0.vertex(2) ) );
	poly1.set_center( calc_average_vector<float, 3>( poly1.vertex(0), poly1.vertex(1), poly1.vertex(2) ) );
	poly0.set_normal( poly.normal() );
	poly1.set_normal( poly.normal() );

	refine_and_push( poly0, threshold );
	refine_and_push( poly1, threshold );
}

} // namespace common 

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_COMMON_STL3D_H
