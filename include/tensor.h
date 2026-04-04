#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <iostream>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
	std::vector<float> data;
	std::vector<float> grad;
	std::vector<int> shape;
	bool requires_grad = false;

	std::vector<TensorPtr> parents;
	std::function<void()> backward_fn;

	Tensor();
	Tensor(
		const std::vector<float>& data, 
		const std::vector<int>& shape, 
		bool requires_grad = false
	);

	void zero_grad();
	void print() const;
	void backward();
};

TensorPtr tensor(
	const std::vector<float>& data,
	const std::vector<int>& shape,
	bool required_grad = false
);

TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr mul(const TensorPtr& a, const TensorPtr& b);
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
TensorPtr sum(const TensorPtr& a);
