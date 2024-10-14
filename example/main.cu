#include <sdfGenerator/gpu/STL.h>
#include "./vtk.h"

using real = double;

int main()
{
	const int num_cell[3] = { 100, 100, 100 };
	const int NXYZ = num_cell[0] * num_cell[1] * num_cell[2];

	real* ls_cc = new real[NXYZ];

	const char* file_path = "./Stanford_Bunny.stl";

	const float domain_len[3] = { 1, 1, 1 };
	const float dh[3] = {
		domain_len[0] / num_cell[0],
		domain_len[1] / num_cell[1],
		domain_len[2] / num_cell[2] };

	float* coord_x = new float[ num_cell[0] ];
	float* coord_y = new float[ num_cell[1] ];
	float* coord_z = new float[ num_cell[2] ];

	for ( int id = 0; id < num_cell[0]; id++) coord_x[id] = dh[0] * ( id + 0.5 );
	for ( int id = 0; id < num_cell[1]; id++) coord_y[id] = dh[1] * ( id + 0.5 );
	for ( int id = 0; id < num_cell[2]; id++) coord_z[id] = dh[2] * ( id + 0.5 );

	const float* coord[3] = { coord_x, coord_y, coord_z };

	const float offset[3] = { 0.3, 0.5, -0.1 };

	const float scale_factor = 0.008;

	sdfGenerator::gpu::STL sdf_gene;
	sdf_gene.readSTL( file_path, offset, scale_factor );
	sdf_gene.refinePolygon( dh[0] );
	//sdf_gene.calcSDF( ls_cc, num_cell, coord );

	//vtk_write<real>( "bunny.vtk", ls_cc, num_cell, dh );

	delete[] ls_cc;
	for (int axis = 0; axis < 3; axis++) delete[] (coord[axis]);

	return 0;
}

