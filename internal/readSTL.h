/*
 * Reference
 *   [1] (Qiita) バイナリフォーマットstlファイルをC++で読み込む,
 *       Author: @JmpM (まるや),
 *       url: https://qiita.com/JmpM/items/9355c655614d47b5d67b
 */

#ifndef INCLUDED_SDFGENERATOR_INTERNAL_READ_STL_H
#define INCLUDED_SDFGENERATOR_INTERNAL_READ_STL_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <internal/struct.h>

namespace sdfGenerator
{

void readSTL(
		std::vector<polygon>& polys,
		const char* file_path,
		const float offset[3],
		const float scale_factor )
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
		polys.push_back(tmp_polygon);

		ifs.seekg(2, std::ios_base::cur);
	}

	ifs.close();
}

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_INTERNAL_READ_STL_H

