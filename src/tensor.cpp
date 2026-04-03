#include "tensor.h"
#include <iostream>

Tensor::Tensor() :requires_grad(false) {}

Tensor::Tensor(const std::vector<float>& data,
				const std::vector<int>& shape,
				bool requires_grad)
	: data(data), shape(shape), requires_grad(requires_grad) {
	grad = std::vector<float>(data.size(), 0.0f);
}

void Tensor::zero_grad() {
	std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::print() const {
	std::cout << "Tensor(data=[ ";
	for (float v : data) {
		std::cout << v << " ";
	}

	std::cout << "], grad=[ ";
	for (float g : grad) {
		std::cout << g << " ";
	}

	std::cout << "], shape=[ ";
	for (int s : shape)
	{
		std::cout << s << " ";
	}

	std::cout << "])" << std::endl;
}

TensorPtr tensor(const std::vector<float>& data,
					const std::vector<int>& shape,
					bool requires_grad) {
	return std::make_shared<Tensor>(data, shape, requires_grad);
}

void Tensor::build_topo(std::vector<Tensor*>& topo, std::unordered_set<Tensor*>& visited) {
	if (visited.count(this))
	{
		return;
	}

	visited.insert(this);

	for (const auto& parent : parents)
	{
		parent->build_topo(topo, visited);
	}

	topo.push_back(this);
}

void Tensor::backward() {
	if (data.size() != 1)
	{
		throw std::runtime_error("backward() currently only supports scalar outputs");
	}
	
	std::vector<Tensor*> topo;
	std::unordered_set<Tensor*> visited;
	build_topo(topo, visited);
	
	std::fill(grad.begin(), grad.end(), 0.0f);
	grad[0] = 1.0f;

	std::reverse(topo.begin(), topo.end());

	for(Tensor* node : topo) {
		if (node->backward_fn) {
			node->backward_fn();
		}
	}
}

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
	if (a->shape != b->shape)
	{
		throw std::invalid_argument("add requires tensors with same shape");
	}

	if (a->data.size() != b->data.size())
	{
		throw std::invalid_argument("add requires tensors with same size");
	}

	std::vector<float> result(a->data.size());
	for (size_t i = 0; i < a->data.size(); i++)
	{
		result[i] = a->data[i] + b->data[i];
	}

	bool req_grad = a->requires_grad || b->requires_grad;
	TensorPtr out = tensor(result, a->shape, req_grad);

	out->parents = {a, b};

	out->backward_fn = [a, b, out]() {
		for (size_t i = 0; i < out->grad.size(); i++)
		{
			if (a->requires_grad)
			{
				a->grad[i] += out->grad[i];
			}

			if (b->requires_grad)
			{
				b->grad[i] += out->grad[i];
			}
		}
	};
		
	return out;
}

TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
	if (a->shape != b->shape)
	{
		throw std::invalid_argument("mul requires tensors with same shape");
	}

	if (a->data.size() != b->data.size())
	{
		throw std::invalid_argument("mul requires tensors with same size");
	}

	std::vector<float> result(a->data.size());
	for (size_t i = 0; i < a->data.size(); i++)
	{
		result[i] = a->data[i] * b->data[i];
	}

	bool req_grad = a->requires_grad || b->requires_grad;
	TensorPtr out = tensor(result, a->shape, req_grad);

	out->parents = {a, b};

	out->backward_fn = [a, b, out]() {
		for (size_t i = 0; i < out->grad.size(); i++)
		{
			if (a->requires_grad)
			{
				a->grad[i] += b->data[i] * out->grad[i];
			}
			if (b->requires_grad)
			{
				b->grad[i] += a->data[i] * out->grad[i];
			}
		}
	};

	return out;
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
	if (a->shape.size() != 2 || b->shape.size() != 2)
	{
		throw std::invalid_argument("matmul requires 2D tensors");
	}

	int m = a->shape[0];
	int n = a->shape[1];
	int n2 = b->shape[0];
	int p = b->shape[1];

	if (n != n2)
	{
		throw std::invalid_argument("matmul shape mismatch");
	}

	if (a->data.size() != static_cast<size_t>(m * n) || 
		b->data.size() != static_cast<size_t>(n2 * p))
	{
		throw std::invalid_argument("tensor data size does not match shape");
	}

	std::vector<float> result(m * p, 0.0f);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < p; j++)
		{
			for (int k = 0; k < n; k++) {
				result[i * p + j] += a->data[i * n + k] * b->data[k * p + j];
			}
		}
	}

	bool req_guard = a->requires_grad || b->requires_grad;
	TensorPtr out = tensor(result, {m, p}, req_guard);

	out->parents = {a, b};

	out->backward_fn = [a, b, out, m, n, p]() {
		if (a->requires_grad)
		{
			for (int i = 0; i < m; i++) {
				for (int k = 0; k < n; k++)
				{
					float grad_val = 0.0f;
					for (int j = 0; j < p; j++)
					{
						grad_val += out->grad[i * p + j] * b->data[k * p + j];
					}
					a->grad[i * n + k] += grad_val;
				}
			}
		}

		if (b->requires_grad)
		{
			for (int k = 0; k < n; k++)
			{
				for (int j = 0; j < p; j++) {
					float grad_val = 0.0f;
					for (int i = 0; i < m; i++)
					{
						grad_val += a->data[i * n + k] * out->grad[i * p + j];
					}
					b->grad[k * p + j] += grad_val;
				}
			}
		}
	};

	return out;
}

TensorPtr sum(const TensorPtr& a) {
	float total = 0.0f;
	for (float v : a->data)
	{
		total += v;
	}

	TensorPtr out = tensor({ total }, { 1 }, a->requires_grad);
	out->parents = { a };

	out->backward_fn = [a, out]() {
		if (a->requires_grad)
		{
			for (size_t i = 0; i < a->grad.size(); i++)
			{
				a->grad[i] += out->grad[0];
			}
		}
	};

	return out;
}
