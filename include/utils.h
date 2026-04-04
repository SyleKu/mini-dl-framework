#pragma once

#include <vector>

std::vector<float> one_hot_encode(int label, int num_classes);
int argmax(const std::vector<float>& values);
std::vector<size_t> shuffled_indices(size_t n);