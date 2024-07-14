/*
 * Reference ( I refered to [1] for reading stl files and [2] for calculating signed distance function. )
 *
 *   [1] (Qiita) バイナリフォーマットstlファイルをC++で読み込む,
 *       Author: @JmpM (まるや),
 *       url: https://qiita.com/JmpM/items/9355c655614d47b5d67b
 *
 *   [2] Mittal et al., A versatile sharp interface immersed boundary method for incompressible flows with complex boundaries.,
 *       Journal of Computational Physics 227 (2008) 4825-4852,
 *       url: https://www.sciencedirect.com/science/article/pii/S0021999108000235
 */

#ifndef INCLUDED_SDFGENERATOR_INTERNAL_SDF_FROM_STL_H
#define INCLUDED_SDFGENERATOR_INTERNAL_SDF_FROM_STL_H


#include <cstdlib>
#include <vector>
#include <internal/readSTL.h>
#include <internal/struct.h>

namespace sdfGenerator
{

template <typename T>
__global__
void sdf_from_stl(
		T* sdf_cc,
		const polygon* polys,
		const float* coord_x, // x-coordinate of cell-centers.
		const float* coord_y, // y-coordinate of cell-centers.
		const float* coord_z, // z-coordinate of cell-centers.
		const int NX, // number of cells in x-direction.
		const int NY, // number of cells in y-direction.
		const int NZ, // number of cells in z-direction.
		const int num_polygons,
		const int sign_inside )
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;
	const int j = threadIdx.y + blockDim.y * blockIdx.y;
	const int k = threadIdx.z + blockDim.z * blockIdx.z;

	if ( i < 0 || NX <= i || j < 0 || NY <= j || k < 0 || NZ <= k ) return;

	const int index = i + ( j + k * NY ) * NX;

	const float coord[3] = { coord_x[i], coord_y[j], coord_z[k] };

	// The number of recorded nearest polygons.
	constexpr int num_memo = 3;

	// Nearest polygons info
	int nearest_id[num_memo];
	float nearest_dist[num_memo];
	for (int i = 0; i < num_memo; i++)
	{
		nearest_id[i]   = -1;
		nearest_dist[i] = 1e10;
	}

	// Search all polygons.
	for (unsigned int id_p = 0; id_p < num_polygons; id_p++)
	{
		float dist_now = 0.0;
		for (int axis = 0; axis < 3; axis++)
		{
			const float to_center = polys[id_p].center[axis] - coord[axis];
			dist_now += to_center * to_center;
		}
		dist_now = sqrt( dist_now );

		for (int i_memo = 0; i_memo < num_memo; i_memo++)
		{
			if ( nearest_id[i_memo] == -1 || dist_now < nearest_dist[i_memo] )
			{
				for (int j_memo = num_memo - 1; j_memo > i_memo; j_memo--)
				{
					nearest_id[j_memo]   = nearest_id[j_memo - 1];
					nearest_dist[j_memo] = nearest_dist[j_memo - 1];
				}
				nearest_id[i_memo]   = id_p;
				nearest_dist[i_memo] = dist_now;
				break;
			}
		}
	}

	float dot_sum = 0;
	for (int i_memo = 0; i_memo < num_memo; i_memo++)
	{
		float dot = 0.0;
		const int id = nearest_id[i_memo];
		for (int axis = 0; axis < 3; axis++)
			dot += (polys[id].center[axis] - coord[axis]) * polys[id].normal[axis];
		dot_sum += dot / nearest_dist[i_memo];
	}

	int sign = -1;
	if ( dot_sum > 0 ) sign = sign_inside;  // The cell is inside object.
	else               sign = -sign_inside; // The cell is outside object.

	sdf_cc[index] = sign * nearest_dist[0];
}


/*
 * Reads STL files and creates signed distance field from the object surface.
 *
 * Parameters ( All parameters are allocated on host memory. )
 *   sdf_cc      : signed distance field at cell-center.
 *   file_path   : Path of stl file.
 *   num_cell    : Number of cells in each direction.
 *   coord       : Coordinates of cell-centers in each direction.
 *   offset      : Distance to translate stl in each direction.
 *   sign_inside : Sign of level-set function inside an object ( 1 (default) or -1 ).
 *   scale factor: Coefficient to multiply STL coordinate data by ( 1.0 by default ).
 */
template <typename T>
void sdf_from_stl(
		T* sdf_cc,
		const char* file_path,
		const int num_cell[3],
		const float* const coord[3],
		const float offset[3],
		const int sign_inside = 1,
		const float scale_factor = 1.0 )
{
	std::vector<polygon> h_polys;

	const float thres_refine = 0.5 * (coord[0][1] - coord[0][0]);
	readSTL( h_polys, file_path, offset, scale_factor, true, thres_refine );

	const int NXYZ = num_cell[0] * num_cell[1] * num_cell[2];

	// 'd_' means the address of the device memory.
	T* d_sdf_cc;
	polygon* d_polys;
	float* d_coord[3];

	cudaMalloc( &d_sdf_cc, sizeof(T) * NXYZ );
	cudaMalloc( &d_polys, sizeof(polygon) * h_polys.size() );
	for (int axis = 0; axis < 3; axis++)
	{
		const size_t nbytes = sizeof(float) * num_cell[axis];
		cudaMalloc( &(d_coord[axis]), nbytes );
		cudaMemcpy( d_coord[axis], coord[axis], nbytes, cudaMemcpyHostToDevice );
	}
	cudaMemcpy(
			d_polys, h_polys.data(), sizeof( polygon ) * h_polys.size(),
			cudaMemcpyHostToDevice );

	const dim3 block(8, 8, 8);
	const dim3 grid(
			(num_cell[0] + block.x - 1) / block.x,
			(num_cell[1] + block.y - 1) / block.y,
			(num_cell[2] + block.z - 1) / block.z );

	sdf_from_stl<<<grid, block>>>(
			d_sdf_cc, d_polys,
			d_coord[0], d_coord[1], d_coord[2],
			num_cell[0], num_cell[1], num_cell[2],
			h_polys.size(), sign_inside );

	cudaDeviceSynchronize();

	cudaMemcpy( sdf_cc, d_sdf_cc, sizeof(T) * NXYZ, cudaMemcpyDeviceToHost );

	cudaFree( d_sdf_cc );
	cudaFree( d_polys );
	for (int axis = 0; axis < 3; axis++) cudaFree( d_coord[axis] );
}

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_INTERNAL_SDF_FROM_STL_H

