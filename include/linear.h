#pragma once

#include "module.h"
#include "tensor.h"
#include <vector>

class Linear : public Module {
public:
	TensorPtr weights;
	TensorPtr bias;

	Linear(int in_features, int out_features);

	TensorPtr forward(const TensorPtr& x) override;
	std::vector<TensorPtr> parameters() override;
};
