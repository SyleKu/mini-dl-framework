#include "activations.h"
#include <iostream>
#include <cmath>

bool almost_equal(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) < eps;
}

int main() {
	auto x = tensor({-1.0f, 0.0f, 1.0f}, {3}, true);

	auto s = sigmoid(x);
	auto t = tanh_act(x);

	std::cout << "Sigmoid output: " << std::endl;
	s->print();

	std::cout << "\nTanh output: " << std::endl;
	t->print();

	bool sigmoid_ok =
		almost_equal(s->data[0], 1.0f / (1.0f + std::exp(1.0f))) &&
		almost_equal(s->data[1], 0.5f) &&
		almost_equal(s->data[2], 1.0f / (1.0f + std::exp(-1.0f)));

	bool tanh_ok =
		almost_equal(t->data[0], std::tanh(-1.0f)) &&
		almost_equal(t->data[1], 0.0f) &&
		almost_equal(t->data[2], std::tanh(1.0f));

	std::cout << "\nSigmoid test: " << (sigmoid_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "Tanh test: " << (tanh_ok ? "PASSED" : "FAILED") << std::endl;

	if (!sigmoid_ok || !tanh_ok) {
		return 1;
	}

	return 0;
}
