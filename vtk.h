
/*
 * Output level set functions in a VTK file.
 *
 * Reference ( I referred to [1] to learn how to use the legacy format of VTK. )
 *
 *   [1] (Qiita) ParaViewでVTKレガシーフォーマットを使う その1,
 *       Author: @kaityo256 (ロボ太),
 *       url: https://qiita.com/kaityo256/items/661833e9e2bfbac31d4b
 */

#ifndef INCLUDED_VTK_H
#define INCLUDED_VTK_H

#include <fstream>

void vtk_write(
		const char* vtk_filename,
		const double* ls_cc,
		const int num_cell[3],
		const float dh[3] )
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

	for (int k = 0; k < num_cell[2]; k++)
	{
		for (int j = 0; j < num_cell[1]; j++)
		{
			for (int i = 0; i < num_cell[0]; i++)
			{
				const int index = i + (j + k * num_cell[1]) * num_cell[0];
				ofs << ls_cc[index] << std::endl;
			}
		}
	}

	ofs.close();

}

#endif

