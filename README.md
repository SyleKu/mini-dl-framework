# mini-dl-framework
Modern deep learning frameworks such as PyTorch abstract away many important implementation details.
This project aims to re-implement core components from first principles to gain a deeper understanding of:

- tensor abstractions
- reverse-mode automatic differentiation
- neural network training pipelines
- performance considerations in numerical computing

---

## Implemented Features

- Tensor class with basic operations (`add`, `mul`, `matmul`, `sum`)
- Reverse-mode autodiff with dynamic computation graph
- Backward propagation for element-wise ops and matrix multiplication
- Linear layer
- ReLU activation
- Mean squared error (MSE) loss
- Initial gradient and autograd tests

---

## Planned Features

- Optimizers (SGD, Adam)
- Sequential model API
- XOR training demo
- MNIST training
- SIMD-optimized matrix operations
- Additional loss funcions (e.g. CrossEntropy)

---

## Architecture

Core components:

- **Tensor**: stores data, gradients, shape, and graph connectivity
- **Autograd Engine**: reverse-mode autodiff over a dynamic computation graph
- **Operations**: add, mul, matmul, sum, relu
- **Modules**: trainable layers such as `Linear`
- **Losses**: scalar objectives such as mean squared error

---

## Roadmap

- [X] Project setup
- [X] Basic tensor operations
- [X] Initial autograd engine
- [X] Sum reduction and gradient checks
- [X] Matmul backward propagation
- [X] Linear layers
- [X] Activation functions
- [X] Loss functions
- [ ] Optimizers
- [ ] XOR training demos
- [ ] MNIST training

---

## Current Status

The framework currently supports:
- scalar and tensor-based forward computation
- reverse-mode automatic differentiation
- gradient propagation through `add`, `mul`, `sum`, and `matmul`
- simple trainable layers and loss evaluation

Initial tests have been added for:
- tensor operations
- scalar autograd correctness
- gradient checks
- matrix multiplication backward propagation

---

## Goals

The goal is not to compete with production frameworks, but to build a minimal, well-structured system that exposes the 
internal mechanics of deep learning systems.

---

## Future Work

- SIMD acceleration (AVX)
- Convolution layers
- Model serialization
- Performance benchmarking

