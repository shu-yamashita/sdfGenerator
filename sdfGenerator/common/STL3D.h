#ifndef SDFGENERATOR_COMMON_STL3D_H
#define SDFGENERATOR_COMMON_STL3D_H

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <utility>

namespace sdfGenerator
{

namespace common
{

template <typename T>
struct vec3
{
	T x, y, z;
};


struct polygon
{
	vec3<float> vertex[3]; // three vertices
	vec3<float> center;    // center of polygon
	vec3<float> normal;    // normal vector

	void set_vertex( const vec3<float>& v0, const vec3<float>& v1, const vec3<float>& v2 )
	{
		vertex[0] = v0;
		vertex[1] = v1;
		vertex[2] = v2;
	}
};


inline
vec3<float> calc_center(
		const vec3<float>& v0,
		const vec3<float>& v1 )
{
	vec3<float> ret;
	ret.x = ( v0.x + v1.x ) / 2;
	ret.y = ( v0.y + v1.y ) / 2;
	ret.z = ( v0.z + v1.z ) / 2;
	return ret;
}


inline
vec3<float> calc_center(
		const vec3<float>& v0,
		const vec3<float>& v1,
		const vec3<float>& v2 )
{
	vec3<float> ret;
	ret.x = ( v0.x + v1.x + v2.x ) / 3;
	ret.y = ( v0.y + v1.y + v2.y ) / 3;
	ret.z = ( v0.z + v1.z + v2.z ) / 3;
	return ret;
}


inline
float edge_len( const polygon& poly, const int vid_first, const int vid_second )
{
	const float dx = poly.vertex[vid_first].x - poly.vertex[vid_second].x;
	const float dy = poly.vertex[vid_first].y - poly.vertex[vid_second].y;
	const float dz = poly.vertex[vid_first].z - poly.vertex[vid_second].z;
	return std::sqrt( dx * dx + dy * dy + dz * dz );
}


// Return id and length of the longest edge.
//        0
//        |\
// len=   | \        ---> return (1, 2.0)
// sqrt(3)|  \len=2.0
//        |   \
//        1----2
//         len=1.0
inline
std::pair<int, float> longest_edge( const polygon& poly )
{
	float len0 = edge_len( poly, 1, 2 );
	float len1 = edge_len( poly, 2, 0 );
	float len2 = edge_len( poly, 0, 1 );

	std::pair<int, float> ret( 0, len0 );
	if      ( len1 >= len0 && len1 >= len2 ) ret = std::make_pair( 1, len1 );
	else if ( len2 >= len0 && len2 >= len1 ) ret = std::make_pair( 2, len2 );

	return ret;
}


class STL3D
{
public:
	void readSTL( const char* file_path );
	void readSTL( const char* file_path, const float offset[3] );
	void readSTL( const char* file_path, const float scale_factor );
	void readSTL( const char* file_path, const float offset[3], const float scale_factor );

	void refinePolygon( const float threshold );
protected:
	std::vector<polygon> polys_;
private:
	void refine_and_push( const polygon& poly, const float threshold );
};


void STL3D::readSTL( const char* file_path )
{
	const float offset[3] = { 0.0, 0.0, 0.0 };
	const float scale_factor = 1.0;
	readSTL( file_path, offset, scale_factor );
}


void STL3D::readSTL( const char* file_path, const float offset[3] )
{
	const float scale_factor = 1.0;
	readSTL( file_path, offset, scale_factor );
}


void STL3D::readSTL( const char* file_path, const float scale_factor )
{
	const float offset[3] = { 0.0, 0.0, 0.0 };
	readSTL( file_path, offset, scale_factor );
}


void STL3D::readSTL( const char* file_path, const float offset[3], const float scale_factor )
{
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
		vec3<float> vbuff[3], nbuff;

		// Import normal vector.
		ifs.read( (char*) &(nbuff.x), sizeof(float) );
		ifs.read( (char*) &(nbuff.y), sizeof(float) );
		ifs.read( (char*) &(nbuff.z), sizeof(float) );

		// Import vertices.
		for (int id_v = 0; id_v < 3; id_v++)
		{
			ifs.read( (char*) &(vbuff[id_v].x), sizeof(float) );
			ifs.read( (char*) &(vbuff[id_v].y), sizeof(float) );
			ifs.read( (char*) &(vbuff[id_v].z), sizeof(float) );

			vbuff[id_v].x = vbuff[id_v].x * scale_factor + offset[0];
			vbuff[id_v].y = vbuff[id_v].y * scale_factor + offset[1];
			vbuff[id_v].z = vbuff[id_v].z * scale_factor + offset[2];
		}

		vec3<float> cbuff = calc_center( vbuff[0], vbuff[1], vbuff[2] );

		polygon tmp_polygon;
		tmp_polygon.set_vertex( vbuff[0], vbuff[1], vbuff[2] ); 
		tmp_polygon.center = cbuff;
		tmp_polygon.normal = nbuff;

		polys_.push_back( tmp_polygon );

		ifs.seekg(2, std::ios_base::cur);
	}

	ifs.close();
}


void STL3D::refinePolygon( const float threshold )
{
	std::vector<polygon> polys_buff = polys_;
	polys_.clear();
	for ( const auto& each_poly : polys_buff ) refine_and_push( each_poly, threshold );
}


void STL3D::refine_and_push( const polygon& poly, const float threshold )
{
	auto max_edge = longest_edge( poly );

	if ( max_edge.second <= threshold )
	{
		polys_.push_back( poly );
		return;
	}

	// polygon is divided into two polygons.
	polygon poly0, poly1;

	// longest edge is connecting tmp_v0 and tmp_v1.
	const vec3<float> tmp_v0 = poly.vertex[ (max_edge.first + 1) % 3 ];
	const vec3<float> tmp_v1 = poly.vertex[ (max_edge.first + 2) % 3 ];

	const vec3<float> new_vertex = calc_center( tmp_v0, tmp_v1 );

	poly0.set_vertex( poly.vertex[max_edge.first], tmp_v0, new_vertex );
	poly1.set_vertex( poly.vertex[max_edge.first], tmp_v1, new_vertex );
	poly0.center = calc_center( poly0.vertex[0], poly0.vertex[1], poly0.vertex[2] );
	poly1.center = calc_center( poly1.vertex[0], poly1.vertex[1], poly1.vertex[2] );
	poly0.normal = poly.normal;
	poly1.normal = poly.normal;

	refine_and_push( poly0, threshold );
	refine_and_push( poly1, threshold );
}

} // namespace common 

} // namespace sdfGenerator

#endif // SDFGENERATOR_COMMON_STL3D_H
