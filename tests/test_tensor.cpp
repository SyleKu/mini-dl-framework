#include "tensor.h"
#include <iostream>
#include <cmath>

bool almost_equal(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) < eps;
}

bool check_vector(const std::vector<float>& actual, const std::vector<float>& expected) {
	if (actual.size() != expected.size())
	{
		return false;
	}

	for (size_t i = 0; i < actual.size(); i++)
	{
		if (!almost_equal(actual[i], expected[i]))
		{
			return false;
		}
	}
	
	return true;
}

int main() {
	std::cout << "=== Tensor Operation Tests ===" << std::endl;

	auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f }, {2, 2});
	auto b = tensor({5.0f, 6.0f, 7.0f, 8.0f }, {2, 2});

	std::cout << "\nTensor a:" << std::endl;
	a->print();

	std::cout << "\nTensor b:" << std::endl;
	b->print();

	auto c = add(a, b);
	std::cout << "\nadd(a, b):" << std::endl;
	c->print();

	auto d = mul(a, b);
	std::cout << "\nmul(a, b):" << std::endl;
	d->print();

	auto e = matmul(a, b);
	std::cout << "\nmatmul(a, b):" << std::endl;
	e->print();

	bool add_ok = check_vector(c->data, {6.0f, 8.0f, 10.0f, 12.0f});
	bool mul_ok = check_vector(d->data, { 5.0f, 12.0f, 21.0f, 32.0f });
	bool matmul_ok = check_vector(e->data, { 19.0f, 22.0f, 43.0f, 50.0f });

	std::cout << "add test: " << (add_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "mul test: " << (mul_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "matmul test: " << (matmul_ok ? "PASSED" : "FAILED") << std::endl;

	if (!add_ok || !mul_ok || !matmul_ok)
	{
		return 1;
	}

	std::cin.get();
	return 0;
}
