#ifndef INCLUDED_SDFGENERATOR_GPU_STL_H
#define INCLUDED_SDFGENERATOR_GPU_STL_H

#include <sdfGenerator/common/STL.h>

namespace sdfGenerator
{

namespace gpu
{


//template <typename T, typename Coord>
//__global__
//void calcSDF(
//		T* sdf,
//		const common::polygon* polys,
//		const Coord* coord_x, // x-coordinate of cell-centers.
//		const Coord* coord_y, // y-coordinate of cell-centers.
//		const Coord* coord_z, // z-coordinate of cell-centers.
//		const int NX, // number of cells in x-direction.
//		const int NY, // number of cells in y-direction.
//		const int NZ, // number of cells in z-direction.
//		const int num_polygons,
//		const int sign_inside )
//{
//	const int i = threadIdx.x + blockDim.x * blockIdx.x;
//	const int j = threadIdx.y + blockDim.y * blockIdx.y;
//	const int k = threadIdx.z + blockDim.z * blockIdx.z;

//	if ( i < 0 || NX <= i || j < 0 || NY <= j || k < 0 || NZ <= k ) return;

//	const int index = i + ( j + k * NY ) * NX;

//	const Coord coord[3] = { coord_x[i], coord_y[j], coord_z[k] };

//	// The number of recorded nearest polygons.
//	constexpr int num_memo = 8;

//	// Nearest polygons info
//	int nearest_id[num_memo];
//	float nearest_dist[num_memo];
//	for (int i = 0; i < num_memo; i++)
//	{
//		nearest_id[i]   = -1;
//		nearest_dist[i] = 1e10;
//	}

//	// Search all polygons.
//	for (unsigned int id_p = 0; id_p < num_polygons; id_p++)
//	{
//		float dist_now = 0.0;
//		for (int axis = 0; axis < 3; axis++)
//		{
//			const float to_center = polys[id_p].center[axis] - coord[axis];
//			dist_now += to_center * to_center;
//		}
//		dist_now = sqrt( dist_now );

//		for (int i_memo = 0; i_memo < num_memo; i_memo++)
//		{
//			if ( nearest_id[i_memo] == -1 || dist_now < nearest_dist[i_memo] )
//			{
//				for (int j_memo = num_memo - 1; j_memo > i_memo; j_memo--)
//				{
//					nearest_id[j_memo]   = nearest_id[j_memo - 1];
//					nearest_dist[j_memo] = nearest_dist[j_memo - 1];
//				}
//				nearest_id[i_memo]   = id_p;
//				nearest_dist[i_memo] = dist_now;
//				break;
//			}
//		}
//	}

//	float dot_sum = 0;
//	for (int i_memo = 0; i_memo < num_memo; i_memo++)
//	{
//		float dot = 0.0;
//		const int id = nearest_id[i_memo];
//		for (int axis = 0; axis < 3; axis++)
//			dot += (polys[id].center[axis] - coord[axis]) * polys[id].normal[axis];
//		dot_sum += dot / nearest_dist[i_memo];
//	}

//	int sign = -1;
//	if ( dot_sum > 0 ) sign = sign_inside;  // The cell is inside object.
//	else               sign = -sign_inside; // The cell is outside object.

//	sdf[index] = sign * nearest_dist[0];
//}

class STL: public common::STL
{
public:
	template <typename T>
	void calcSDF( T* sdf, const size_t (&num_cell)[3], const T* const (&coord)[3] ) const;
};

template <typename T>
void STL::calcSDF( T* sdf, const size_t (&num_cell)[3], const T* const (&coord)[3] ) const
{
	// TODO
}

// old version
//template <typename T>
//void STL::calcSDF( T* sdf, const size_t (&num_cell)[3], const T* const (&coord)[3] ) const
//{
//	const size_t num_cell_total = num_cell[0] * num_cell[1] * num_cell[2];

//	T* d_sdf;
//	common::polygon* d_polys;
//	Coord* d_coord[3];

//	cudaMalloc( &d_sdf, sizeof(T) * num_cell_total );
//	cudaMalloc( &d_polys, sizeof(common::polygon) * polys_.size() );
//	cudaMemcpy(
//			d_polys, polys_.data(), sizeof(common::polygon) * polys_.size(),
//			cudaMemcpyHostToDevice );
//	for (int axis = 0; axis < 3; axis++)
//	{
//		const size_t nbytes = sizeof(Coord) * num_cell[axis];
//		cudaMalloc( &(d_coord[axis]), nbytes );
//		cudaMemcpy( d_coord[axis], coord[axis], nbytes, cudaMemcpyHostToDevice );
//	}

//	const dim3 block(8, 8, 8);
//	const dim3 grid(
//			(num_cell[0] + block.x - 1) / block.x,
//			(num_cell[1] + block.y - 1) / block.y,
//			(num_cell[2] + block.z - 1) / block.z );

//	calcSDF<<<grid, block>>>(
//			d_sdf, d_polys, d_coord[0], d_coord[1], d_coord[2],
//			num_cell[0], num_cell[1], num_cell[2], polys_.size(), sign_solid );

//	cudaDeviceSynchronize();

//	cudaMemcpy( sdf, d_sdf, sizeof(T) * num_cell_total, cudaMemcpyDeviceToHost );

//	cudaFree( d_sdf );
//	cudaFree( d_polys );
//	for (int axis = 0; axis < 3; axis++) cudaFree( d_coord[axis] );
//}


} // namespace gpu

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_GPU_STL_H
