#include "tensor.h"
#include <iostream>

int main() {
	Tensor a{ {1, 2, 3, 4}, {2, 2} };
	Tensor b{ {5, 6, 7, 8}, {2, 2} };

	Tensor c = add(a, b);
	c.print();

	Tensor d = matmul(a, b);
	d.print();

	return 0;
}
