#pragma once

#include <vector>

std::vector<float> one_hot_encode(int label, int num_classes);
int argmax(const std::vector<float>& values);
