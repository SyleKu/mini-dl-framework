#pragma once

#include "tensor.h"
#include <vector>

class SGD {
public:
	SGD(const std::vector<TensorPtr>& parameters, float lr);

	void zero_grad();
	void step();

private:
	std::vector<TensorPtr> params;
	float learning_rate;
};
