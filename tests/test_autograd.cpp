#include "tensor.h"
#include <iostream>
#include <cmath>

bool almost_equal(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) < eps;
}

int main(){
	auto x = tensor({2.0f}, {1}, true);
	auto y = tensor({3.0f}, {1}, true);

	auto z = mul(x, y);
	auto w = add(z, x);

	std::cout << "Before backward:" << std::endl;
	x->print();
	y->print();
	w->print();

	w->backward();

	std::cout << "\nAfter backward:" << std::endl;
	x->print();
	y->print();
	w->print();

	bool x_grad_ok = almost_equal(x->grad[0], 4.0f);
	bool y_grad_ok = almost_equal(y->grad[0], 2.0f);
	bool w_value_ok = almost_equal(w->data[0], 8.0f);

	std::cout << "\nw value test: " << (w_value_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "x.grad test: " << (x_grad_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "y.grad test: " << (y_grad_ok ? "PASSED" : "FAILED") << std::endl;

	if (!w_value_ok || !x_grad_ok || !y_grad_ok)
	{
		return 1;
	}

	return 0;
}
