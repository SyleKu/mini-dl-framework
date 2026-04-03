#pragma once

#include <vector>
#include <memory>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Module {
public:
	virtual TensorPtr forward(const TensorPtr& x) = 0;
	virtual std::vector<TensorPtr> parameters() = 0;
	virtual ~Module() = default;
};

class Sequential : public Module {
public:
	Sequential(const std::vector<std::shared_ptr<Module>>& modules);

	TensorPtr forward(const TensorPtr& x) override;
	std::vector<TensorPtr> parameters() override;

private:
	std::vector<std::shared_ptr<Module>> layers;
};
