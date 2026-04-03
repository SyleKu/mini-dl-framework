#include "tensor.h"
#include <iostream>

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

	std::cin.get();
	return 0;
}
