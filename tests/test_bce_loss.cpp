#include "losses.h"
#include <iostream>
#include <cmath>

bool almost_equal(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) < eps;
}

int main() {
	auto pred = tensor({ 0.9f }, { 1 }, true);
	auto target = tensor({ 1.0f }, { 1 }, false);

	auto loss = binary_cross_entropy(pred, target);

	std::cout << "BCE loss output:" << std::endl;
	loss->print();

	bool loss_ok = loss->data[0] > 0.0f;

	std::cout << "\nBCE loss test: " << (loss_ok ? "PASSED" : "FAILED") << std::endl;

	if (!loss_ok)
	{
		return 1;
	}

	return 0;
}
