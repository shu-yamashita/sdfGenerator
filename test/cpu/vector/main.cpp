#include <iostream>
#include <sdfGenerator/detail/internal/vector.h>

template <typename T, size_t Dim>
void TestDefaultConstructorHost()
{
	sdfGenerator::internal::vector<T, Dim> vec;
	for (size_t i = 0; i <  Dim; i++) assert( vec[i] == T() );
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
}
