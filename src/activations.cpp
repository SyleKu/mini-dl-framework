#include "activations.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

TensorPtr relu(const TensorPtr& a) {
	std::vector<float> result(a->data.size());

	for (size_t i = 0; i < a->data.size(); i++)
	{
		result[i] = std::max(0.0f, a->data[i]);
	}

	TensorPtr out = tensor(result, a->shape, a->requires_grad);
	out->parents = { a };

	out->backward_fn = [a, out]() {
		if (a->requires_grad)
		{
			for (size_t i = 0; i < a->data.size(); i++) {
				float grad_mask = (a->data[i] > 0.0f) ? 1.0f : 0.0f;
				a->grad[i] += grad_mask * out->grad[i];
			}
		}
	};

	return out;
}

TensorPtr sigmoid(const TensorPtr& a) {
	std::vector<float> result(a->data.size());

	for (size_t i = 0; i < a->data.size(); i++)
	{
		result[i] = 1.0f / (1.0f + std::exp(-a->data[i]));
	}

	TensorPtr out = tensor(result, a->shape, a->requires_grad);
	out->parents = { a };

	out->backward_fn = [a, out]() {
		if (a->requires_grad)
		{
			for (size_t i = 0; i < a->data.size(); i++) 
			{
				float s = out->data[i];
				a->grad[i] += s * (1.0f - s) * out->grad[i];
			}
		}
	};

	return out;
}

TensorPtr tanh_act(const TensorPtr& a) {
	std::vector<float> result(a->data.size());

	for (size_t i = 0; i < a->data.size(); i++)
	{
		result[i] = std::tanh(a->data[i]);
	}

	TensorPtr out = tensor(result, a->shape, a->requires_grad);
	out->parents = { a };

	out->backward_fn = [a, out]() {
		if (a->requires_grad)
		{
			for (size_t i = 0; i < a->data.size(); i++) {
				float t = out->data[i];
				a->grad[i] += (1.0f - t * t) * out->grad[i];
			}
		}
	};

	return out;
};

TensorPtr softmax(const TensorPtr& a) {
	if (a->shape.size() != 2 || a->shape[0] != 1)
	{
		throw std::invalid_argument("softmax currently expects shape {1, n}");
	}

	int n = a->shape[1];
	std::vector<float> result(n);

	float max_val = a->data[0];
	for (int i = 1; i < n; i++)
	{
		if (a->data[i] > max_val) {
			max_val = a->data[i];
		}
	}

	float sum_exp = 0.0f;
	for (int i = 0; i < n; i++)
	{
		result[i] = std::exp(a->data[i] - max_val);
		sum_exp += result[i];
	}

	for (int i = 0; i < n; i++)
	{
		result[i] /= sum_exp;
	}

	TensorPtr out = tensor(result, a->shape, a->requires_grad);
	out->parents = { a };

	out->backward_fn = [a, out, n]() {
		if (a->requires_grad)
		{
			for (int i = 0; i < n; i++) {
				float grad_i = 0.0f;

				for (int j = 0; j < n; j++)
				{
					float jacobian = 0.0f;

					if (i == j)
					{
						jacobian = out->data[i] * (1.0f - out->data[i]);
					}
					else {
						jacobian = -out->data[i] * out->data[j];
					}

					grad_i += jacobian * out->grad[j];
				}

				a->grad[i] += grad_i;
			}
		}
	};

	return out;

}
