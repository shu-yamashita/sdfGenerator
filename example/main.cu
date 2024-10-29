#include <sdfGenerator/STL.h>
#include <sdfGenerator/stl_to_sdf.h>
#include "./vtk.h"

using real = double;

int main()
{
	const size_t num_cell[3] = { 100, 100, 100 };
	const size_t NXYZ = num_cell[0] * num_cell[1] * num_cell[2];

	real* ls_cc = new real[NXYZ];

	const char* file_path = "./Stanford_Bunny.stl";

	const real domain_len[3] = { 1, 1, 1 };
	const real dh[3] = {
		domain_len[0] / num_cell[0],
		domain_len[1] / num_cell[1],
		domain_len[2] / num_cell[2] };

	real* coord_x = new real[ NXYZ ];
	real* coord_y = new real[ NXYZ ];
	real* coord_z = new real[ NXYZ ];

	for (size_t k = 0; k < num_cell[2]; k++) for (size_t j = 0; j < num_cell[1]; j++) for (size_t i = 0; i < num_cell[0]; i++)
	{
		const size_t index = i + (j + k * num_cell[1]) * num_cell[0];
		coord_x[index] = (i + 0.5) * dh[0];
		coord_y[index] = (j + 0.5) * dh[1];
		coord_z[index] = (k + 0.5) * dh[2];
	}

	const real offset[3] = { 0.3, 0.5, -0.1 };

	const real scale_factor = 0.008;

	sdfGenerator::STL stl;
	stl.readSTL( file_path, offset, scale_factor );
	stl.refinePolygon( dh[0] );
	sdfGenerator::gpu::stl_to_sdf( ls_cc, coord_x, coord_y, coord_z, NXYZ, stl );

	//vtk_write<real>( "bunny.vtk", ls_cc, num_cell, dh );

	delete[] ls_cc;
	delete[] coord_x;
	delete[] coord_y;
	delete[] coord_z;

	return 0;
}

