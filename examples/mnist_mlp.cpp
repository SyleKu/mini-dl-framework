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

int main() {
	try {
		std::filesystem::path project_root = MINI_DL_PROJECT_ROOT;

		auto train_data = load_mnist_images_and_labels(
			(project_root / "data/train-images-idx3-ubyte").string(),
			(project_root / "data/train-labels-idx1-ubyte").string(),
			1000
		);

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

			for (size_t i = 0; i < train_data.images.size(); i++)
			{
				optimizer.zero_grad();

				auto x = tensor(train_data.images[i], {1, 784}, false);
				auto target = tensor(one_hot_encode(train_data.labels[i], 10), {1, 10}, false);

				auto h1 = layer1->forward(x);
				auto a1 = relu(h1);

				auto h2 = layer2->forward(a1);
				auto a2 = relu(h2);

				auto logits = layer3->forward(a2);

				auto loss = mse_loss(logits, target);
				total_loss += loss->data[0];

				int predicted = argmax(logits->data);
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
				<< " | accuracy  = " << accuracy
				<< std::endl;
		}

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "MNIST training failed: " << e.what() << std::endl;
		return 1;
	}
}
