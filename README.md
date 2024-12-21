# sdfGenerator

sdfGenerator is a C++ header-only library for computing the signed distance function (SDF).

> [!WARNING]
> This library is under development. Destructive changes may occur.

<img src="./figs/example_bunny.png" width = 80%>

The `Stanford_Bunny.stl` in this figure can be downloaded from [here](https://commons.wikimedia.org/wiki/File:Stanford_Bunny.stl?uselang=ja).


## Currently available features

* Generate SDF from STL using CUDA [[Document](https://github.com/shu-yamashita/sdfGenerator/tree/main/docs/STL_to_SDF)]

## Examples

The following example reads STL data from `./example.stl` and generates the signed distance function (SDF) at each grid point in the Cartesian grid ($100 \times 100 \times 100$) using a GPU.
```cpp
#include <vector>
#include <sdfGenerator/STL.h>
#include <sdfGenerator/stl_to_sdf.h>

int main()
{
	const size_t NX = 100, NY = 100, NZ = 100; // The number of grid points in each axis.
	const size_t NXYZ = NX * NY * NZ;          // The number of total grid points.

	std::vector<float> coord_x(NXYZ), coord_y(NXYZ), coord_z(NXYZ); // Coordinates at each grid point.

	// Set coordinates at each grid point.
	for (size_t k = 0; k < NZ; k++) for (size_t j = 0; j < NY; j++) for (size_t i = 0; i < NX; i++) {
		const size_t index = i + j*NX + k*NX*NY;
		coord_x[index] = i * 1e-2;
		coord_y[index] = j * 1e-2;
		coord_z[index] = k * 1e-2;
	}

	sdfGenerator::STL stl;

	// Read STL data from the file.
	stl.readSTL( "./example.stl" );

	std::vector<float> sdf(NXYZ); // Signed distance function at each grid point.

	// Calculate the SDF at each grid point and fill the array sdf.
	sdfGenerator::gpu::stl_to_sdf( sdf.data(), coord_x.data(), coord_y.data(), coord_z.data(), NXYZ, stl );
}
```

## Getting the source code

To download the source code, do
```
git clone git@github.com:shu-yamashita/sdfGenerator.git
```

If you want to update the source code later, you need to do
```
git pull
```

## Using sdfGenerator
sdfGenerator is a header-only library, so you can use it by including it in your code.

Compile your project with `-I<sdfGenerator repo root>`.

## Links
* [Github repository](https://github.com/shu-yamashita/sdfGenerator)