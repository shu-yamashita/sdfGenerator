
#ifndef INCLUDED_SDFGENERATOR_ALL_H
#define INCLUDED_SDFGENERATOR_ALL_H

#include <internal/struct.h>
#include <internal/sdf_from_stl.h>

namespace sdfGenerator
{

/*
 * Reads STL files and creates signed distance field from the object surface.
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
void sdf_from_stl(
		T* ls_cc,
		const char* file_path,
		const int num_cell[3],
		const float* const coord[3],
		const float offset[3],
		const int sign_inside,
		const float scale_factor );

} // namespace sdfGenerator

#endif // INCLUDED_SDFGENERATOR_ALL_H

