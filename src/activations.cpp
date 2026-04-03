#include "activations.h"
#include <algorithm>

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
