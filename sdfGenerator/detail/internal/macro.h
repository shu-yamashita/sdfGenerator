#ifndef INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_MACRO_H
#define INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_MACRO_H


#ifdef __NVCC__
	#define SDFGENERATOR_CUDA_HOST_DEVICE __host__ __device__
#else
	#define SDFGENERATOR_CUDA_HOST_DEVICE
#endif // __NVCC__


#define CUDA_SAFE_CALL(func) do \
	{ \
		cudaError_t err = (func); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
			exit(err); \
		} \
	} while(0)

#endif // INCLUDED_SDFGENERATOR_DETAIL_INTERNAL_MACRO_H
