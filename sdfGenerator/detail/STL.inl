#ifndef INCLUDED_SDFGENERATOR_DETAIL_STL_INL
#define INCLUDED_SDFGENERATOR_DETAIL_STL_INL

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sdfGenerator/STL.h>
#include <sdfGenerator/detail/internal/logger.h>
#include <sdfGenerator/detail/internal/polygon.h>
#include <sdfGenerator/detail/internal/vector.h>


namespace sdfGenerator
{


inline void STL::readSTL( const char* file_path )
{
	const float offset[3] = { 0.0, 0.0, 0.0 };
	const float scale_factor = 1.0;
	readSTL( file_path, offset, scale_factor );
}


template <typename T>
void STL::readSTL( const char* file_path, const T (&offset)[3] )
{
	const float scale_factor = 1.0;
	readSTL( file_path, offset, scale_factor );
}


template <typename T>
void STL::readSTL( const char* file_path, const T scale_factor )
{
	const float offset[3] = { 0.0, 0.0, 0.0 };
	readSTL( file_path, offset, scale_factor );
}


template <typename T1, typename T2>
void STL::readSTL( const char* file_path, const T1 (&offset)[3], const T2 scale_factor )
{
	internal::processLogger proc_logger( "readSTL" );

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

	std::printf("[sdfGenerator] Polygon Count: %u\n", num_polygons);

	// Read polygon data.
	for (unsigned int id_p = 0; id_p < num_polygons; id_p++)
	{
		internal::vector<float, 3> vbuff[3], nbuff;

		// Read normal vector.
		for (size_t axis = 0; axis < 3; axis++) ifs.read( (char*) &(nbuff[axis]), sizeof(float) );

		// Read vertices.
		for (size_t id_v = 0; id_v < 3; id_v++)
		{
			for (size_t axis = 0; axis < 3; axis++) ifs.read( (char*) &(vbuff[id_v][axis]), sizeof(float) );
			for (size_t axis = 0; axis < 3; axis++) vbuff[id_v][axis] = vbuff[id_v][axis] * scale_factor + offset[axis];
		}

		triangle tri;
		tri.set_vertices( vbuff[0], vbuff[1], vbuff[2] ); 
		tri.set_center( internal::calc_average_vector<float, 3>( vbuff[0], vbuff[1], vbuff[2] ) );
		tri.set_normal( nbuff );

		triangles_.push_back( tri );

		ifs.seekg(2, std::ios_base::cur);
	}

	ifs.close();
}


inline size_t STL::getNumPolygon() const
{
	return triangles_.size();
}


inline STL::triangle* STL::getPolygonData()
{
	return triangles_.data();
}


inline const STL::triangle* STL::getPolygonData() const
{
	return triangles_.data();
}


template <typename T>
void STL::refinePolygon( const T threshold )
{
	internal::processLogger proc_logger( "refinePolygon" );

	const size_t num_tri_before_refine = triangles_.size();
	std::printf("[sdfGenerator] Polygon Count Before Refinement: %zu\n", num_tri_before_refine);

	std::vector<triangle> triangles_buff = triangles_;
	triangles_.clear();
	for ( const auto& each_tri : triangles_buff ) refine_and_push( each_tri, threshold );

	const size_t num_tri_after_refine = triangles_.size();
	const float increase_factor = (float) num_tri_after_refine / num_tri_before_refine;

	std::printf("[sdfGenerator] Polygon Count After Refinement : %zu (Increased %.2f Times)\n", num_tri_after_refine, increase_factor);
}


inline void STL::refine_and_push( const triangle& tri, const float threshold )
{
	const std::pair<size_t, float> max_edge = internal::find_longest_edge_triangle( tri );

	if ( max_edge.second <= threshold )
	{
		triangles_.push_back( tri );
		return;
	}

	// The triangle is divided into two triangles.
	triangle tri0, tri1;

	// The longest edge connects tmp_v0 and tmp_v1.
	const internal::vector<float, 3> tmp_v0 = tri.vertex( (max_edge.first + 1) % 3 );
	const internal::vector<float, 3> tmp_v1 = tri.vertex( (max_edge.first + 2) % 3 );

	const internal::vector<float, 3> new_vertex = internal::calc_average_vector<float, 3>( tmp_v0, tmp_v1 );

	tri0.set_vertices( tri.vertex(max_edge.first), tmp_v0, new_vertex );
	tri1.set_vertices( tri.vertex(max_edge.first), tmp_v1, new_vertex );
	tri0.set_center( internal::calc_average_vector<float, 3>( tri0.vertex(0), tri0.vertex(1), tri0.vertex(2) ) );
	tri1.set_center( internal::calc_average_vector<float, 3>( tri1.vertex(0), tri1.vertex(1), tri1.vertex(2) ) );
	tri0.set_normal( tri.normal() );
	tri1.set_normal( tri.normal() );

	refine_and_push( tri0, threshold );
	refine_and_push( tri1, threshold );
}


} // namespace sdfGenerator


#endif // INCLUDED_SDFGENERATOR_DETAIL_STL_INL
