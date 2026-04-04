#include "linear.h"
#include "activations.h"
#include "losses.h"
#include "optimizer.h"
#include <iostream>
#include <memory>
#include <vector>

int main() {
	auto layer1 = std::make_shared<Linear>(2, 8);
	auto layer2 = std::make_shared<Linear>(8, 1);

	std::vector<TensorPtr> params;
	auto p1 = layer1->parameters();
	auto p2 = layer2->parameters();
	params.insert(params.end(), p1.begin(), p1.end());
	params.insert(params.end(), p2.begin(), p2.end());

	SGD optimizer(params, 0.01f);

	std::vector<TensorPtr> inputs = {
		tensor({0.0f, 0.0f}, {1, 2}, false),
		tensor({0.0f, 1.0f}, {1, 2}, false),
		tensor({1.0f, 0.0f}, {1, 2}, false),
		tensor({1.0f, 1.0f}, {1, 2}, false)
	};

	std::vector<TensorPtr> targets = {
		tensor({0.0f}, {1, 1}, false),
		tensor({1.0f}, {1, 1}, false),
		tensor({1.0f}, {1, 1}, false),
		tensor({0.0f}, {1, 1}, false)
	};

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		float total_loss = 0.0f;

		for (size_t i = 0; i < inputs.size(); i++)
		{
			optimizer.zero_grad();

			auto h1 = layer1->forward(inputs[i]);
			auto a1 = tanh_act(h1);
			auto out = layer2->forward(a1);
			auto pred = sigmoid(out);

			auto loss = mse_loss(pred, targets[i]);
			total_loss += loss->data[0];

			loss->backward();
			optimizer.step();
		}

		if (epoch % 20 == 0)
		{
			std::cout << "Epoch " << epoch
				<< " | loss = " << total_loss / static_cast<float>(inputs.size())
				<< std::endl;
		}
	}

	std::cout << "\nFinal predictions:" << std::endl;
	for (size_t i = 0; i < inputs.size(); i++)
	{
		auto h1 = layer1->forward(inputs[i]);
		auto a1 = tanh_act(h1);
		auto out = layer2->forward(a1);
		auto pred = sigmoid(out);

		std::cout << inputs[i]->data[0] << " xor " << inputs[i]->data[1]
			<< " -> " << pred->data[0] << std::endl;
	}

	return 0;
}