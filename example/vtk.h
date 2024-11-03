
/*
 * Reference:
 *   [1] (Qiita) ParaViewでVTKレガシーフォーマットを使う その1,
 *       Author: @kaityo256 (ロボ太),
 *       url: https://qiita.com/kaityo256/items/661833e9e2bfbac31d4b
 */

#ifndef INCLUDED_VTK_H
#define INCLUDED_VTK_H

#include <fstream>

template <typename T>
void vtk_write(
		const char* vtk_filename,
		const T* sdf,
		const size_t (&num_cell)[3],
		const T dh[3] )
{
	std::ofstream ofs( vtk_filename );
	ofs << "# vtk DataFile Version 1.0" << std::endl;
	ofs << vtk_filename << std::endl;
	ofs << "ASCII" << std::endl;
	ofs << "DATASET STRUCTURED_POINTS" << std::endl;
	ofs << "DIMENSIONS " << num_cell[0] + 1 << " " << num_cell[1] + 1 << " " << num_cell[2] + 1 << std::endl;
	ofs << "ORIGIN 0.0 0.0 0.0" << std::endl;
	ofs << "SPACING " << dh[0] << " " << dh[1] << " " << dh[2] << std::endl;

	ofs << "CELL_DATA " << num_cell[0] * num_cell[1] * num_cell[2] << std::endl;
	ofs << "SCALARS ls float" << std::endl;
	ofs << "LOOKUP_TABLE default" << std::endl;

	for (size_t k = 0; k < num_cell[2]; k++)
	{
		for (size_t j = 0; j < num_cell[1]; j++)
		{
			for (size_t i = 0; i < num_cell[0]; i++)
			{
				const size_t index = i + (j + k * num_cell[1]) * num_cell[0];
				ofs << sdf[index] << std::endl;
			}
		}
	}

	ofs.close();
}

#endif

