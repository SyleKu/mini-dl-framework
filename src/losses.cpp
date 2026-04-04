#include "losses.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

TensorPtr mse_loss(const TensorPtr& prediciton, const TensorPtr& target) {
	if (prediciton->shape != target->shape)
	{
		throw std::invalid_argument("mse_loss requires prediction and target to have the same shape");
	}

	size_t n = prediciton->data.size();
	float loss_value = 0.0f;

	for (size_t i = 0; i < n; i++)
	{
		float diff = prediciton->data[i] - target->data[i];
		loss_value += diff * diff;
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

TensorPtr binary_cross_entropy(const TensorPtr& prediciton, const TensorPtr& target) {
	if (prediciton->shape != target->shape)
	{
		throw std::invalid_argument("binary_cross_entropy requires matching shapes");
	}

	const float eps = 1e-7f;
	size_t n = prediciton->data.size();
	float loss_value = 0.0f;

	for (size_t i = 0; i < n; i++)
	{
		float p = std::clamp(prediciton->data[i], eps, 1.0f - eps);
		float y = target->data[i];
		loss_value += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
	}

	loss_value /= static_cast<float>(n);

	bool req_grad = prediciton->requires_grad || target->requires_grad;
	TensorPtr out = tensor({ loss_value }, {1}, req_grad);
	out->parents = { prediciton, target };

	out->backward_fn = [prediciton, target, out, n, eps]() {
		if (prediciton->requires_grad)
		{
			for (size_t i = 0; i < prediciton->data.size(); i++) {
				float p = std::clamp(prediciton->data[i], eps, 1.0f - eps);
				float y = target->data[i];

				float grad = (-y / p) + ((1.0f - y) / (1.0f - p));
				prediciton->grad[i] += (grad / static_cast<float>(n)) * out->grad[0];
			}
		}
	};

	return out;
}

TensorPtr cross_entropy_loss(const TensorPtr& prediciton, const TensorPtr& target) {
	if (prediciton->shape != target->shape)
	{
		throw std::invalid_argument("cross_entropy_loss requires matching shapes");
	}

	const float eps = 1e-7f;
	size_t n = prediciton->data.size();
	float loss_value = 0.0f;

	for (size_t i = 0; i < n; i++)
	{
		float p = std::clamp(prediciton->data[i], eps, 1.0f);
		float y = target->data[i];
		loss_value += -y * std::log(p);
	}

	bool req_grad = prediciton->requires_grad || target->requires_grad;
	TensorPtr out = tensor({ loss_value }, { 1 }, req_grad);
	out->parents = {prediciton, target};

	out->backward_fn = [prediciton, target, out, eps]() {
		if (prediciton->requires_grad)
		{
			for (size_t i = 0; i < prediciton->data.size(); i++)
			{
				float p = std::clamp(prediciton->data[i], eps, 1.0f);
				float y = target->data[i];
				prediciton->grad[i] += (-y / p) * out->grad[0];
			}
		}
	};

	return out;
}