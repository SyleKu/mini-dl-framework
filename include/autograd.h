#pragma once

#include "tensor.h"
#include <vector>

std::vector<TensorPtr> build_topological_order(const TensorPtr& root);
void run_backward(const TensorPtr&);
