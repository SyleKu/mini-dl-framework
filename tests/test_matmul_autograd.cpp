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
		if (!almost_equal(actual[i], expected[i], eps))
		{
			return false;
		}
	}

	return true;
}

int main() {
	auto a = tensor({ 1.0f, 2.0f }, {1, 2}, true); // 1x2
	auto b = tensor({ 3.0f, 4.0f, 5.0f, 6.0f }, {2, 2}, true); // 2x2

	auto c = matmul(a, b); // 1x2
	auto s = sum(c); // scalar

	s->backward();

	bool s_ok = almost_equal(s->data[0], 29.0f); // [1 2] * [[3 4], [5 6]] = [13 16], sum = 29
	bool a_grad_ok = check_vector(a->grad, {7.0f, 11.0f});
	bool b_grad_ok = check_vector(b->grad, {1.0f, 1.0f, 2.0f, 2.0f});

	std::cout << "sum value test: " << (s_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "a.grad test: " << (a_grad_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "b.grad test: " << (b_grad_ok ? "PASSED" : "FAILED") << std::endl;

	if (!s_ok || !a_grad_ok || !b_grad_ok)
	{
		return 1;
	}

	return 0;
}
