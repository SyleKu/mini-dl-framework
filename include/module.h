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
