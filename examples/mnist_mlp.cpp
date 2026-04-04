#include "linear.h"
#include "activations.h"
#include "losses.h"
#include "optimizer.h"
#include "mnist.h"
#include "utils.h"

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <exception>

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

		auto train_data = load_mnist_images_and_labels(
			images_path.string(),
			labels_path.string(),
			1000
		);

		std::cout << "Loaded " << train_data.images.size() << " training samples." << std::endl;

		auto layer1 = std::make_shared<Linear>(784, 128);
		auto layer2 = std::make_shared<Linear>(128, 64);
		auto layer3 = std::make_shared<Linear>(64, 10);

		std::vector<TensorPtr> params;
		auto p1 = layer1->parameters();
		auto p2 = layer2->parameters();
		auto p3 = layer3->parameters();

		params.insert(params.end(), p1.begin(), p1.end());
		params.insert(params.end(), p2.begin(), p2.end());
		params.insert(params.end(), p3.begin(), p3.end());

		SGD optimizer(params, 0.01f);

		const int epochs = 5;

		for (int epoch = 0; epoch < epochs; epoch++)
		{
			float total_loss = 0.0f;
			int correct = 0;

			auto indices = shuffled_indices(train_data.images.size());

			for (size_t idx = 0; idx < indices.size(); idx++)
			{
				size_t i = indices[idx];

				optimizer.zero_grad();

				auto x = tensor(train_data.images[i], {1, 784}, false);
				auto target = tensor(one_hot_encode(train_data.labels[i], 10), {1, 10}, false);

				auto h1 = layer1->forward(x);
				auto a1 = relu(h1);

				auto h2 = layer2->forward(a1);
				auto a2 = relu(h2);

				auto logits = layer3->forward(a2);
				auto probs = softmax(logits);

				auto loss = cross_entropy_loss(probs, target);
				total_loss += loss->data[0];

				int predicted = argmax(probs->data);
				if (predicted == train_data.labels[i])
				{
					correct++;
				}

				loss->backward();
				optimizer.step();
			}

			float avg_loss = total_loss / static_cast<float>(train_data.images.size());
			float accuracy = static_cast<float>(correct) / static_cast<float>(train_data.images.size());

			std::cout << "Epoch " << epoch
				<< " | loss  = " << avg_loss
				<< " | accuracy  = " << accuracy * 100.0f << "%"
				<< std::endl;
		}

		std::cout << "\nSample predictions:" << std::endl;
		for (size_t i = 0; i < std::min<size_t>(5, train_data.images.size()); i++)
		{
			auto x = tensor(train_data.images[i], {1, 784}, false);

			auto h1 = layer1->forward(x);
			auto a1 = relu(h1);

			auto h2 = layer2->forward(a1);
			auto a2 = relu(h2);

			auto logits = layer3->forward(a2);
			auto probs = softmax(logits);

			int predicated = argmax(probs->data);
			int actual = train_data.labels[i];

			std::cout << "Sample " << i
				<< " | predicated  = " << predicated
				<< " | actual  = " << actual
				<< std::endl;
		}

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "MNIST training failed: " << e.what() << std::endl;
		return 1;
	}
}
