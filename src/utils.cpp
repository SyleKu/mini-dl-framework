#include "utils.h"
#include <stdexcept>
#include <random>
#include <algorithm>


std::vector<float> one_hot_encode(int label, int num_classes) {
	if (label < 0 || label >= num_classes)
	{
		throw std::invalid_argument("label out of range in one_hot_encode");
	}

	std::vector<float> encoded(num_classes, 0.0f);
	encoded[label] = 1.0f;
	return encoded;
}

int argmax(const std::vector<float>& values) {
	if (values.empty())
	{
		throw std::invalid_argument("argmax received empty vector");
	}


	int best_index = 0;
	float best_value = values[0];
	
	for (size_t i = 1; i < values.size(); i++)
	{
		if (values[i] > best_value)
		{
			best_value = values[i];
			best_index = static_cast<int>(i);
		}
	}
	return best_index;
}

std::vector<size_t> shuffled_indices(size_t n) {
	std::vector<size_t> indices(n);
	for (size_t i = 0; i < n; i++)
	{
		indices[i] = i;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(indices.begin(), indices.end(), gen);

	return indices;
}
