#include "linear.h"
#include <random>

Linear::Linear(int in_features, int out_features) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

	std::vector<float> w(in_features * out_features);
	for (auto& val : w)
	{
		val = dist(gen);
	}

	std::vector<float> b(out_features, 0.0f);

	weights = tensor(w, { in_features , out_features }, true);
	bias = tensor(b, { 1 , out_features }, true);
}

TensorPtr Linear::forward(const TensorPtr& x) {
	auto out = matmul(x, weights);
	return add(out, bias);
}

std::vector<TensorPtr> Linear::parameters() {
	return {weights, bias};
}
