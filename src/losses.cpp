#include "losses.h"
#include <stdexcept>

TensorPtr mse_loss(const TensorPtr& prediciton, const TensorPtr& target) {
	if (prediciton->shape != target->shape)
	{
		throw std::invalid_argument("mse_loss requires prediction and target to have the same shape");
	}

	size_t n = prediciton->data.size();
	std::vector<float> diff_sq(n, 0.0f);
	float loss_value = 0.0f;

	for (size_t i = 0; i < n; i++)
	{
		float diff = prediciton->data[i] - target->data[i];
		diff_sq[i] = diff * diff;
		loss_value += diff_sq[i];
	}

	loss_value /= static_cast<float>(n);

	bool req_grad = prediciton->requires_grad || target->requires_grad;
	TensorPtr out = tensor({loss_value}, { 1 }, req_grad);
	out->parents = {prediciton, target};

	out->backward_fn = [prediciton, target, out, n]() {
		float scale = 2.0f / static_cast<float>(n);

		if (prediciton->requires_grad)
		{
			for (size_t i = 0; i < prediciton->data.size(); i++) {
				float diff = prediciton->data[i] - target->data[i];
				prediciton->grad[i] += scale * diff * out->grad[0];
			}
		}

		if (target->requires_grad)
		{
			for (size_t i = 0; i < target->data.size(); i++) {
				float diff = prediciton->data[i] - target->data[i];
				target->grad[i] -= scale * diff * out->grad[0];
			}
		}
	};

	return out;
}
