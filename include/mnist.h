#pragma once

#include <vector>
#include <string>

struct MNISTData {
	std::vector<std::vector<float>> images;
	std::vector<int> labels;
};

MNISTData load_mnist_images_and_labels(const std::string& images_path,
										const std::string& labels_path,
										int max_sample = -1);
