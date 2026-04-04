#include "utils.h"
#include <iostream>

int main() {
	auto encoded = one_hot_encode(3, 10);

	bool one_hot_ok =
		encoded.size() == 10 &&
		encoded[3] == 1.0f &&
		encoded[0] == 0.0f &&
		encoded[9] == 0.0f;

	int max_index = argmax({0.1f, 0.2f, 0.9f, 0.3f});
	bool argmax_ok = (max_index == 2);

	std::cout << "one_hot_encode test: " << (one_hot_ok ? "PASSED" : "FAILED") << std::endl;
	std::cout << "argmax test: " << (argmax_ok ? "PASSED" : "FAILED") << std::endl;

	if (!one_hot_ok || !argmax_ok)
	{
		return 1;
	}

	return 0;
}
