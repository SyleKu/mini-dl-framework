#include "tensor.h"
#include <iostream>
#include <cmath>

bool almost_equal(float a, float b, float eps = 1e-4f) {
	return std::fabs(a - b) < eps;
}

bool check_vector(const std::vector<float>& actual, const std::vector<float>& expected, float eps = 1e-4f) {
	if (actual.size() != expected.size())
	{
		return false;
	}

	for (size_t i = 0; i < actual.size(); i++)
	{
		if (!almost_equal(actual[i], expected[i], eps)) {
			return false;
		}
	}

	return true;
}

int main() {
	auto x = tensor({2.0f, 4.0f}, {2}, true);
	auto y = tensor({3.0f, 5.0f}, {2}, true);

	auto z = mul(x, y);
	auto s = sum(z);

	s->backward();

	bool x_grad_ok = check_vector(x->grad, {3.0f, 5.0f});
	bool y_grad_ok = check_vector(y->grad, {2.0f, 4.0f});
	bool s_value_ok = almost_equal(s->data[0], 26.0f);

	std::cout << "sum value test: " << (s_value_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "x.grad test: " << (x_grad_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "y.grad test: " << (y_grad_ok ? "PASSED" : "FAILED") << std::endl;

	if (!s_value_ok || !x_grad_ok || !y_grad_ok)
	{
		return 1;
	}

	return 0;
}
