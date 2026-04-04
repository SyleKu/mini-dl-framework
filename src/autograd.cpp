#include "autograd.h"
#include <unordered_set>
#include <stdexcept>

namespace {

	void build_topo(const TensorPtr& node,
					std::unordered_set<Tensor*>& visited,
					std::vector< TensorPtr>& topo) {

		if (visited.count(node.get())) return;
		visited.insert(node.get());

		for (const auto& parent : node->parents)
		{
			build_topo(parent, visited, topo);
		}

		topo.push_back(node);
	}
}

std::vector<TensorPtr> build_topological_order(const TensorPtr& root) {
	std::vector<TensorPtr> topo;
	std::unordered_set<Tensor*> visited;

	build_topo(root, visited, topo);
	return topo;
}

void run_backward(const TensorPtr& root) {
	if (root->data.size() != 1)
	{
		throw std::runtime_error("backward() currently only supports scalar outputs");
	}

	auto topo = build_topological_order(root);
	
	std::fill(root->grad.begin(), root->grad.end(), 0.0f);
	root->grad[0] = 1.0f;

	for (auto it = topo.rbegin(); it != topo.rend(); ++it)
	{
		if ((*it)->backward_fn)
		{
			(*it)->backward_fn();
		}
	}
}
