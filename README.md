# mini-dl-framework
Modern deep learning frameworks such as PyTorch abstract away many important implementation details.
This project aims to re-implement core components from first principles to gain a deeper understanding of:

- tensor abstractions
- reverse-mode automatic differentiation
- neural network training pipelines
- performance considerations in numerical computing

---

## Features (Planned)

- Tensor class with basic operations
- Reverse-mode autodiff (computational graph)
- Neural network layers (Linear, ReLU, etc.)
- Loss functions (MSE, CrossEntropy)
- Optimizers (SGD, Adam)
- Training demos (XOR, MNIST)
- Optional SIMD optimizations for matrix operations

---

## Architecture

Core components:

- Tensor: data + gradient tracking
- Autograd Engine: dynamic computation graph
- Modules: neural network layers
- Optimizers: parameter updates

---

## Roadmap

- [X] Project setup
- [ ] Basic tensor operations
- [ ] Autograd engine
- [ ] Linear layers
- [ ] Activation functions
- [ ] Loss functions
- [ ] Optimizers
- [ ] XOR training demos
- [ ] MNIST training

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

