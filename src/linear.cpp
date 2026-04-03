#include "linear.h"
#include <stdexcept>

Linear::Linear(int in_features, int out_features) {
	std::vector<float> w(in_features * out_features, 0.1f);
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
