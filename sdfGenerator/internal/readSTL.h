/*
 * Reference
 *   [1] (Qiita) バイナリフォーマットstlファイルをC++で読み込む,
 *       Author: @JmpM (まるや),
 *       url: https://qiita.com/JmpM/items/9355c655614d47b5d67b
 */

#ifndef INCLUDED_SDFGENERATOR_INTERNAL_READ_STL_H
#define INCLUDED_SDFGENERATOR_INTERNAL_READ_STL_H

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <sdfGenerator/internal/struct.h>

namespace sdfGenerator
{


void push_polygon_with_refinement(
		std::vector<polygon>& poly_container,
		const polygon& poly,
		const bool flag_refine,
		const float thres_refine )
{
	if ( !flag_refine )
	{
		poly_container.push_back( poly );
		return;
	}

	auto norm = [&]( const int v0, const int v1 ) -> float 
	{
		const float dx = poly.vertex[v0][0] - poly.vertex[v1][0];
		const float dy = poly.vertex[v0][1] - poly.vertex[v1][1];
		const float dz = poly.vertex[v0][2] - poly.vertex[v1][2];

		return std::sqrt( dx * dx + dy * dy + dz * dz );
	};

	const float edge_len[3] = { norm( 1, 2 ), norm( 2, 0 ), norm( 0, 1 ) };
	int max_len_id = 0, no_max_id1 = 1, no_max_id2 = 2;
	if      ( edge_len[1] > edge_len[0] && edge_len[1] > edge_len[2] )
	{
		max_len_id = 1;
		no_max_id1 = 0;
		no_max_id2 = 2;
	}
	else if ( edge_len[2] > edge_len[0] && edge_len[2] > edge_len[1] )
	{
		max_len_id = 2;
		no_max_id1 = 0;
		no_max_id2 = 1;
	}

	if ( edge_len[max_len_id] <= thres_refine )
	{
		poly_container.push_back( poly );
		return;
	}

	vec3d new_vertex;
	const vec3d& v1 = poly.vertex[no_max_id1];
	const vec3d& v2 = poly.vertex[no_max_id2];
	for (int axis = 0; axis < 3; axis++) new_vertex[axis] = 0.5 * (v1[axis] + v2[axis]);

	polygon poly1, poly2;
	vec3d poly1_v[3] = { poly.vertex[max_len_id], v1, new_vertex };
	vec3d poly2_v[3] = { poly.vertex[max_len_id], v2, new_vertex };
	poly1.setter( poly1_v, poly.normal );
	poly2.setter( poly2_v, poly.normal );

	push_polygon_with_refinement( poly_container, poly1, flag_refine, thres_refine );
	push_polygon_with_refinement( poly_container, poly2, flag_refine, thres_refine );
}


/*
 * Reads an STL file, subdivides the polygon data as necessary, and stores it in polys.
 *
 * Params:
 *   file_path   : stl file path
 *   offset      : distance to translate an object
 *   scale_factor: the factor by which to scale an object
 *   flag_refine : wheter to subdivide polygon
 *   thres_refine: if flag_refine is true,
 *                 subdivide polygons so that its longest edge is less than this length.
 */
void readSTL(
		std::vector<polygon>& polys,
		const char* file_path,
		const float offset[3],
		const float scale_factor,
		const bool  flag_refine,
		const float thres_refine )
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
		vec3d vbuff[3], nbuff;

		// Import normal vector.
		for (int axis = 0; axis < 3; axis++)
			ifs.read( (char*) &nbuff[axis], sizeof(float) );

		// Import vertices.
		for (int id_v = 0; id_v < 3; id_v++)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				ifs.read( (char*) &vbuff[id_v][axis], sizeof(float) );
				vbuff[id_v][axis] *= scale_factor;
				vbuff[id_v][axis] += offset[axis];
			}
		}

		polygon tmp_polygon;
		tmp_polygon.setter( vbuff, nbuff );
		// Refine this polygon and store in polys.
		push_polygon_with_refinement( polys, tmp_polygon, flag_refine, thres_refine );

		ifs.seekg(2, std::ios_base::cur);
	}

	ifs.close();
}

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_INTERNAL_READ_STL_H

