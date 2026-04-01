#include "tensor.h"
#include <iostream>

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad)
	: data(data), shape(shape), requires_grad(requires_grad) {
	grad = std::vector<float>(data.size(), 0.0f);
}

void Tensor::print() const {
	for (float v : data) {
		std::cout << v << " ";
	}
}

void Tensor::zero_grad() {
	std::fill(grad.begin(), grad.end(), 0.0f);
}
