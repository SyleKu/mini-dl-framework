#pragma once
#include "module.h"
#include "tensor.h"

class Linear : public Module {
public:
	Tensor weights;
	Tensor bias;

	Linear(int in_features, int out_features);
	Tensor forward(const Tensor& x) override;
	std::vector<Tensor*> parameters() override;
};
