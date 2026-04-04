#include "mnist.h"
#include <iostream>
#include <filesystem>

int main() {
	try {
		std::filesystem::path project_root = MINI_DL_PROJECT_ROOT;

		auto data = load_mnist_images_and_labels(
			(project_root / "data/train-images-idx3-ubyte").string(),
			(project_root / "data/train-labels-idx1-ubyte").string(),
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
