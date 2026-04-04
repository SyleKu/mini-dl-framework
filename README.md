# mini-dl-framework
Modern deep learning frameworks such as PyTorch abstract away many important implementation details.
This project aims to re-implement core components from first principles to gain a deeper understanding of:

- tensor abstractions
- reverse-mode automatic differentiation
- neural network training pipelines
- performance considerations in numerical computing

---
## Results

### XOR Training

A two-layer MLP was trained on the XOR problem.

Final prediction after training:

- `0 xor 0 -> 0.0068`
- `0 xor 1 -> 0.9884`
- `1 xor 0 -> 0.9898`
- `1 xor 1 -> 0.0123`

![XOR Training](assets/xor_training.png)


This demonstrates that the framework supports forward computation, gradient proagation,
parameter updates, and successful learning of a non-linearly separable problem.

---

## Implemented Features

- Tensor class with basic operations (`add`, `mul`, `matmul`, `sum`)
- Reverse-mode autodiff with dynamic computation graph
- Backward propagation for element-wise ops and matrix multiplication
- Trainable `Linear` layer
- Actiovation functions: `ReLU`, `Sigmoid` ,`Tanh`
- Loss functions: Mean squared error (`MSE`), Binary Cross Entropy Loss (`BCE`)
- Stochastic gradient descent (`SGD`) optimizer
- End-to-end XOR training demo (non-linear learning)
- IDX-based MNIST data loader
- Initial MLP prototype for MNIST classificaiton

---

## Planned Features

- Softmax activation
- CrossEntropy loss
- Improved evaluation pipeline
- MNIST training
- SIMD-optimized matrix operations

---

## Architecture

Core components:

- **Tensor**: stores data, gradients, shape, and graph connectivity
- **Autograd Engine**: reverse-mode autodiff over a dynamic computation graph
- **Operations**: add, mul, matmul, sum, activations
- **Modules**: trainable layers such as `Linear` and model composition via `Sequential`
- **Losses**: scalar objectives such as MSE
- **Optimizers**: parameter updates via SGD

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
- [X] Optimizers
- [X] XOR training demos
- [X] MNIST data loading
- [X] MNIST MLP prototype
- [ ] Softmax ( CrossEntropy
- [ ] SIMD optimizations

---

## Current Status

The framework currently supports:
- scalar and tensor-based forward computation
- reverse-mode automatic differentiation
- gradient propagation through `add`, `mul`, `sum`, and `matmul`
- simple trainable layers and loss evaluation
- simple optimization with SGD
- an initial XOR learning demo

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


---

## Motivation

This porject was developed to gain a deeper understanding of how deep learning
frameworks work internally, including gradient propagation, computational graphs,
and training dynamics.
