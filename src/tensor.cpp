#include "tensor.h"
#include <iostream>

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad)
	: data(data), shape(shape), requires_grad(requires_grad) {
	grad = std::vector<float>(data.size(), 0.0f);
}

void Tensor::print() const {
	std::cout << "Tensor: ";
	for (float v : data) {
		std::cout << v << " ";
	}

	std::cout << std::endl;
}

void Tensor::zero_grad() {
	std::fill(grad.begin(), grad.end(), 0.0f);
}

Tensor add(const Tensor& a, const Tensor& b) {
	std::vector<float> result(a.data.size());

	for (size_t i = 0; i < a.data.size(); i++)
	{
		result[i] = a.data[i] + b.data[i];
	}

	return Tensor(result, a.shape);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
	int m = a.shape[0];
	int n = a.shape[1];
	int p = b.shape[1];

	std::vector<float> result(m * p, 0.0f);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < p; j++)
		{
			for (int k = 0; k < n; k++) {
				result[i * p + j] += a.data[i * n + k] * b.data[k * p + j];
			}
		}
	}

	return Tensor(result, {m, p});
}

