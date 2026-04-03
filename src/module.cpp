#include "module.h"

Sequential::Sequential(const std::vector<std::shared_ptr<Module>>& modules)
	: layers(modules) {}

TensorPtr Sequential::forward(const TensorPtr& x) {
	TensorPtr out = x;
	for (auto& layer : layers)
	{
		out = layer->forward(out);
	}
	return out;
}

std::vector<TensorPtr> Sequential::parameters() {
	std::vector<TensorPtr> params;

	for (auto& layer : layers)
	{
		auto layer_params = layer->parameters();
		params.insert(params.end(), layer_params.begin(), layer_params.end());
	}

	return params;
}
