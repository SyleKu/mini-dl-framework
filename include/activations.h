#pragma once

#include "tensor.h"

TensorPtr relu(const TensorPtr& a);
TensorPtr sigmoid(const TensorPtr& a);
TensorPtr tanh_act(const TensorPtr& a);
TensorPtr softmax(const TensorPtr& a);
