#include "linear.h"
#include "activation.h"
#include "losses.h"
#include <iostream>
#include <cmath>

bool almost_equal(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) < eps;
}

int main() {
	Linear layer(2, 1);

	auto x = tensor({ 1.0f, 2.0f }, {1, 2}, false);
	auto target = tensor({ 1.0f }, {1, 1}, false);

	auto pred = layer.forward(x);
	auto loss = mse_loss(pred, target);

	std::cout << "Before backward:" << std::endl;
	pred->print();
	loss->print();

	loss->backward();

	std::cout << "\nAfter backward:" << std::endl;
	layer.weights->print();
	layer.bias->print();
	loss->print();

	bool weight_grad_nonzero = false;
	for(float g : layer.weights->grad)
	{
		if (std::fabs(g) > 1e-6f)
		{
			weight_grad_nonzero = true;
			break;
		}
	}

	bool bias_grad_nonzero = false;
	for (float g : layer.bias->grad)
	{
		if (std::fabs(g) > 1e-6f)
		{
			bias_grad_nonzero = true;
			break;
		}
	}

	std::cout << "\nweights grad test: " << (weight_grad_nonzero ? "PASSED" : "FAILED") << std::endl;
	std::cout << "bias grad test: " << (bias_grad_nonzero ? "PASSED" : "FAILED") << std::endl;
	
	if (!weight_grad_nonzero || !bias_grad_nonzero)
	{
		return 1;
	}

	return 0;
}
