#include "activations.h"
#include "losses.h"
#include <iostream>



int main() {
	auto logits = tensor({ 1.0f, 2.0f, 0.5f }, { 1, 3 }, true);
	auto probs = softmax(logits);
	auto target = tensor({ 0.0f, 1.0f, 0.0f }, { 1, 3 }, false);

	auto loss = cross_entropy_loss(probs, target);

	std::cout << "Softmax output:" << std::endl;
	probs->print();

	std::cout << "\nCross entropy loss:" << std::endl;
	loss->print();

	bool loss_ok = loss->data[0] > 0.0f;
	std::cout << "\nSoftmax + CrossEntropy test: " << (loss_ok ? "PASSED" : "FAILED") << std::endl;

	if (!loss_ok)
	{
		return 1;
	}
	return 0;
}
