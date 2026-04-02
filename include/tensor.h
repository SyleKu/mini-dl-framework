#pragma once
#include <vector>
#include <functional>
#include <memory>

class Tensor {
public:
	std::vector<float> data;
	std::vector<float> grad;
	std::vector<int> shape;
	bool requires_grad = false;

	Tensor();
	Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad = false);

	void zero_grad();
	void backward();
	void print() const;
};

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
