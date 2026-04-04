#include "mnist.h"
#include <iostream>

int main() {
	try {
		auto data = load_mnist_images_and_labels(
			"data/train-images-idx3-ubyte",
			"data/train-labels-idx1-ubyte",
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
