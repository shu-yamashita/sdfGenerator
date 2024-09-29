#include <cstdlib>
#include <iostream>
#include <sdfGenerator/detail/common/vector.h>


void check_cuda_error()
{
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess )
	{
		std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}


template <typename T, size_t Dim>
__host__ __device__
void TestDefaultConstructor()
{
	sdfGenerator::detail::common::vector<T, Dim> vec;
	for (size_t i = 0; i <  Dim; i++) assert( vec[i] == T() );
}

template <typename T, size_t Dim>
__global__
void TestDefaultConstructor_Kernel()
{
	TestDefaultConstructor<T, Dim>();
}

template <typename T, size_t Dim>
void TestDefaultConstructor_CallKernel()
{
	TestDefaultConstructor_Kernel<T, Dim><<<3, 3>>>();
	check_cuda_error();
}

template <typename T, size_t Dim>
__host__ __device__
void TestSame()
{
	sdfGenerator::detail::common::vector<T, Dim> vec1, vec2;
	for (size_t i = 0; i < Dim; i++) vec1[i] = 0;
	assert( vec1 == vec2 );
}

template <typename T, size_t Dim>
__global__
void TestSame_Kernel()
{
	TestSame<T, Dim>();
}

template <typename T, size_t Dim>
void TestSame_CallKernel()
{
	TestSame_Kernel<T, Dim><<<3, 3>>>();
	check_cuda_error();
}

template <typename T, size_t Dim, size_t N>
__host__ __device__
void TestAddSub()
{
	sdfGenerator::detail::common::vector<T, Dim> sum, sub, sum_op_p, sub_op_m;
	T ans_sum[Dim] = {};
	T ans_sub[Dim] = {};

	for (size_t id = 0; id < N; id++)
	{
		sdfGenerator::detail::common::vector<T, Dim> vec;

		for (size_t axis = 0; axis < Dim; axis++)
		{
			const T tmp = axis * id * 3.14;
			vec[axis] = tmp;
			ans_sum[axis] += tmp;
			ans_sub[axis] -= tmp;
		}
		sum += vec; // operator+=
		sub -= vec; // operator-=
		sum_op_p = sum_op_p + vec; // operator+
		sub_op_m = sub_op_m - vec; // operator-
	}

	// test operator+=
	for (size_t i = 0; i < Dim; i++) assert( ans_sum[i] == sum[i] );
	// test operator-=
	for (size_t i = 0; i < Dim; i++) assert( ans_sub[i] == sub[i] );
	// test operator+
	for (size_t i = 0; i < Dim; i++) assert( ans_sum[i] == sum_op_p[i] );
	// test operator-
	for (size_t i = 0; i < Dim; i++) assert( ans_sub[i] == sub_op_m[i] );
}

template <typename T, size_t Dim, size_t N>
__global__
void TestAddSub_Kernel()
{
	TestAddSub<T, Dim, N>();
}

template <typename T, size_t Dim, size_t N>
void TestAddSub_CallKernel()
{
	TestAddSub_Kernel<T, Dim, N><<<3, 3>>>();
	check_cuda_error();
}

int main()
{
	std::cout << "[ Test ] default constructor (host)" << std::endl;
	TestDefaultConstructor<int, 3>();
	TestDefaultConstructor<float, 4>();
	TestDefaultConstructor<double, 2>();

	std::cout << "[ Test ] default constructor (device)" << std::endl;
	TestDefaultConstructor_CallKernel<int, 3>();
	TestDefaultConstructor_CallKernel<float, 4>();
	TestDefaultConstructor_CallKernel<double, 2>();

	std::cout << "[ Test ] operator== (host)" << std::endl;
	TestSame<int, 4>();
	TestSame<float, 9>();
	TestSame<double, 2>();

	std::cout << "[ Test ] operator== (device)" << std::endl;
	TestSame_CallKernel<int, 4>();
	TestSame_CallKernel<float, 9>();
	TestSame_CallKernel<double, 2>();

	std::cout << "[ Test ] operators +, -, +=, -= (host)" << std::endl;
	TestAddSub<int, 5, 10>();
	TestAddSub<float, 2, 10>();
	TestAddSub<double, 9, 10>();

	std::cout << "[ Test ] operators +, -, +=, -= (device)" << std::endl;
	TestAddSub_CallKernel<int, 5, 10>();
	TestAddSub_CallKernel<float, 2, 10>();
	TestAddSub_CallKernel<double, 9, 10>();
}
