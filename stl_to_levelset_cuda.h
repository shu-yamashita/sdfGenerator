
/*
 * Convert STL to level-set function.
 *
 * Reference ( I refered to [1] for reading stl files and [2] for calculating level set functions. )
 *
 *   [1] (Qiita) バイナリフォーマットstlファイルをC++で読み込む,
 *       Author: @JmpM (まるや),
 *       url: https://qiita.com/JmpM/items/9355c655614d47b5d67b
 *
 *   [2] Mittal et al., A versatile sharp interface immersed boundary method for incompressible flows with complex boundaries.,
 *       Journal of Computational Physics 227 (2008) 4825-4852,
 *       url: https://www.sciencedirect.com/science/article/pii/S0021999108000235
 */


#ifndef INCLUDED_STL_TO_LEVELSET_CUDA_H
#define INCLUDED_STL_TO_LEVELSET_CUDA_H

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace STL_to_LS_CUDA
{

struct vec3d
{
	float elems_[3];
	__host__ __device__
	      float& operator[]( const int axis )       { return elems_[axis]; }
	__host__ __device__
	const float& operator[]( const int axis ) const { return elems_[axis]; }
};


struct polygon
{
	// Center of polygon.
	// Computed as the average of three vertices.
	vec3d center;

	// Normal vector.
	vec3d normal;

	void setter(
			// Coordinate of three vertices.
			const vec3d v[3],
			// Normal vector
			const vec3d& n )
	{
		for (int axis = 0; axis < 3; axis++)
			center[axis] = ( v[0][axis] + v[1][axis] + v[2][axis] ) / 3;
		normal = n;
	}
};


// Calculate the level-set function.
template <typename T>
__global__
void get_levelset_from_stl(
		T* ls_cc,
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
	// Odd number
	constexpr int num_memo = 15;

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

	int cnt_pos = 0, cnt_neg = 0;
	for (int i_memo = 0; i_memo < num_memo; i_memo++)
	{
		float dot = 0.0;
		const int id = nearest_id[i_memo];
		for (int axis = 0; axis < 3; axis++)
			dot += (polys[id].center[axis] - coord[axis]) * polys[id].normal[axis];
		if ( dot > 0 ) cnt_pos++;
		else           cnt_neg++;
	}

	int sign = -1;
	if ( cnt_pos > cnt_neg ) sign = sign_inside;  // The cell is inside object.
	else                     sign = -sign_inside; // The cell is outside object.

	ls_cc[index] = sign * nearest_dist[0];
}


/*
 * Reads STL files and creates level set functions.
 * The level set function is a signed distance function from the object surface.
 *
 * Parameters ( All parameters are allocated on host memory. )
 *   ls_cc       : Level-set function at cell-center.
 *   file_path   : Path of stl file.
 *   num_cell    : Number of cells in each direction.
 *   coord       : Coordinates of cell-centers in each direction.
 *   offset      : Distance to translate stl in each direction.
 *   sign_inside : Sign of level-set function inside an object ( 1 (default) or -1 ).
 *   scale factor: Coefficient to multiply STL coordinate data by ( 1.0 by default ).
 */
template <typename T>
void get_levelset_from_stl(
		T* ls_cc,
		const char* file_path,
		const int num_cell[3],
		const float* const coord[3],
		const float offset[3],
		const int sign_inside = 1,
		const float scale_factor = 1.0 )
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

	// 'h_' means the address of the host memory.
	polygon* h_polys = new polygon[num_polygons];


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

		h_polys[id_p].setter( vbuff, nbuff );
		ifs.seekg(2, std::ios_base::cur);
	}

	ifs.close();

	const int NXYZ = num_cell[0] * num_cell[1] * num_cell[2];

	// 'd_' means the address of the device memory.
	T* d_ls_cc;
	polygon* d_polys;
	float* d_coord[3];

	cudaMalloc( &d_ls_cc, sizeof(T) * NXYZ );
	cudaMalloc( &d_polys, sizeof(polygon) * num_polygons );
	for (int axis = 0; axis < 3; axis++)
	{
		cudaMalloc( &(d_coord[axis]), sizeof(float) * num_cell[axis] );
		cudaMemcpy(
				d_coord[axis], coord[axis], sizeof(float) * num_cell[axis],
				cudaMemcpyHostToDevice );
	}
	cudaMemcpy(
			d_polys, h_polys, sizeof( polygon ) * num_polygons,
			cudaMemcpyHostToDevice );

	const dim3 block(8, 8, 8);
	const dim3 grid(
			(num_cell[0] + block.x - 1) / block.x,
			(num_cell[1] + block.y - 1) / block.y,
			(num_cell[2] + block.z - 1) / block.z );

	// Calc levelset function.
	get_levelset_from_stl<<<grid, block>>>(
			d_ls_cc, d_polys,
			d_coord[0], d_coord[1], d_coord[2],
			num_cell[0], num_cell[1], num_cell[2],
			num_polygons, sign_inside );

	cudaMemcpy(
			ls_cc, d_ls_cc, sizeof(T) * NXYZ,
			cudaMemcpyDeviceToHost );

	delete[] h_polys;
	cudaFree( d_ls_cc );
	cudaFree( d_polys );
	for (int axis = 0; axis < 3; axis++) cudaFree( d_coord[axis] );
}

} // namespace STL_to_LS_CUDA

#endif

