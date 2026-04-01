#pragma once
#include <vector>
#include <memory>

class Tensor;

class Module {
public:
	virtual Tensor forward(const Tensor& x) = 0;
	virtual std::vector<Tensor*> parameters() = 0;
	virtual ~Module() * default;
};
