#include <iostream>
#include <sdfGenerator/common/vector.h>

template <typename T, size_t Dim>
void TestDefaultConstructorHost()
{
	sdfGenerator::common::vector<T, Dim> vec;
	for (size_t i = 0; i <  Dim; i++) assert( vec[i] == T() );
}

template <typename T, size_t Dim>
__global__
void TestDefaultConstructorDevice_Kernel()
{
	sdfGenerator::common::vector<T, Dim> vec;
	for (size_t i = 0; i < Dim; i++) assert( vec[i] == T() );
}

template <typename T, size_t Dim>
void TestDefaultConstructorDevice()
{
	TestDefaultConstructorDevice_Kernel<T, Dim><<<1, 1>>>();
	cudaDeviceSynchronize();
}

int main()
{
	std::cout << "[ TestDefaultConstructorHost ]" << std::endl;
	TestDefaultConstructorHost<int, 3>();
	TestDefaultConstructorHost<int, 5>();
	TestDefaultConstructorHost<float, 4>();
	TestDefaultConstructorHost<float, 8>();
	TestDefaultConstructorHost<double, 2>();
	TestDefaultConstructorHost<double, 4>();

	std::cout << "[ TestDefaultConstructorDevice ]" << std::endl;
	TestDefaultConstructorDevice<int, 3>();
	TestDefaultConstructorDevice<int, 5>();
	TestDefaultConstructorDevice<float, 4>();
	TestDefaultConstructorDevice<float, 8>();
	TestDefaultConstructorDevice<double, 2>();
	TestDefaultConstructorDevice<double, 4>();
}
