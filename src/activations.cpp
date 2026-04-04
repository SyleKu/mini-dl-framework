#include "activations.h"
#include <algorithm>
#include <cmath>

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
