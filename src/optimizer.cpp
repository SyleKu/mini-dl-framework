#include "optimizer.h"

SGD::SGD(const std::vector<TensorPtr>& parameters, float lr)
	: params(parameters), learning_rate(lr) {
}

void SGD::zero_grad() {
	for (auto& param : params)
	{
		param->zero_grad();
	}

}

void SGD::step() {
	for (auto& param : params)
	{
		for (size_t i = 0; i < param->data.size(); i++) {
			param->data[i] -= learning_rate * param->grad[i];
		}
	}
}
