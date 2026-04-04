#pragma once

#include "tensor.h"

TensorPtr mse_loss(const TensorPtr& prediciton, const TensorPtr& target);
TensorPtr binary_cross_entropy(const TensorPtr& prediciton, const TensorPtr& target);
TensorPtr cross_entropy_loss(const TensorPtr& prediciton, const TensorPtr& target);
