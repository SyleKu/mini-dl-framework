#include "mnist.h"
#include <iostream>
#include <filesystem>

int main() {
	try {
		std::filesystem::path project_root = MINI_DL_PROJECT_ROOT;
		std::filesystem::path images_path = project_root / "data/train-images-idx3-ubyte";
		std::filesystem::path labels_path = project_root / "data/train-labels-idx1-ubyte";

		std::cout << "Project root: " << project_root << std::endl;
		std::cout << "Images root: " << images_path << std::endl;
		std::cout << "Labels root: " << labels_path << std::endl;

		if (!std::filesystem::exists(images_path))
		{
			throw std::runtime_error("MNIST images file not found: " + images_path.string());
		}

		if (!std::filesystem::exists(labels_path))
		{
			throw std::runtime_error("MNIST labels file not found: " + labels_path.string());
		}

		auto data = load_mnist_images_and_labels(
			images_path.string(),
			labels_path.string(),
			5
		);

		std::cout << "Loaded samples: " << data.images.size() << std::endl;
		std::cout << "Loaded labels: " << data.labels.size() << std::endl;

		if (!data.images.empty())
		{
			std::cout << "First image size: " << data.images[0].size() << std::endl;
			std::cout << "First label: " << data.labels[0] << std::endl;
		}

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "MNIST loader test failed: " << e.what() << std::endl;
		return 1;
	}
}
