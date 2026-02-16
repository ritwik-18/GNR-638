# 🔥 Custom Autograd + CNN Framework

## 📘 Project Overview

This project implements a **custom deep learning framework from scratch**
using a C++ backend and a Python frontend.  
The framework supports **automatic differentiation, CNNs, training, and evaluation**
without relying on external deep learning libraries such as PyTorch or TensorFlow.

The design follows a strict layered architecture:

Tensor Storage → Autograd Engine → Tensor Ops → NN Layers → Python Bindings → Training Pipeline

---

## 📁 Project Structure

cpp/
├── tensor.hpp
├── tensor.cpp
├── autograd.hpp
├── autograd.cpp
├── ops.hpp
├── ops.cpp
├── nn.hpp
├── nn.cpp
├── cnn.hpp
├── cnn.cpp
└── bindings.cpp

python/
├── framework.py
├── dataset.py
├── model.py
├── train.py
└── evaluate.py


---

## 🧱 C++ Backend

### tensor.hpp / tensor.cpp
- Defines the core `Tensor` abstraction
- Stores data, gradients, shape, and autograd metadata
- Uses `std::shared_ptr` to safely manage tensor lifetimes
- No mathematical operations are implemented here

### autograd.hpp / autograd.cpp
- Implements reverse-mode automatic differentiation
- Builds a computation graph dynamically during forward pass
- Performs backpropagation using topological sorting

### ops.hpp / ops.cpp
Implements tensor operations with full backward support:
- Elementwise operations (add, ReLU)
- Matrix multiplication
- Convolution and max pooling
- Flatten and reshape

Each operation:
- Creates an output tensor
- Records parent tensors
- Defines a backward lambda for gradient propagation

### nn.hpp / nn.cpp
Defines high-level neural network abstractions:
- `Module` base class
- `Linear`, `ReLU`, `Sequential`
- CNN layers delegate computation to tensor ops

### cnn.hpp / cnn.cpp
- Implements `Conv2D` and `MaxPool` layers
- Uses naive convolution and pooling for clarity and correctness
- Fully supports backward propagation

### optim.hpp / optim.cpp
- Implements **Stochastic Gradient Descent (SGD)**
- Updates parameters using gradients computed by autograd

### bindings.cpp
- Exposes the C++ framework to Python using **pybind11**
- Exports tensors, layers, loss, optimizer, and utility functions

---

## 🐍 Python Frontend

### framework.py
- Thin wrapper that imports the compiled C++ extension

### dataset.py
- Loads image datasets from folders
- Resizes images to 32×32
- Normalizes pixel values
- Converts data into framework tensors
- Supports mini-batch loading
- Measures dataset loading time

### model.py
- Defines the CNN architecture using exposed C++ layers
- No training logic included

### train.py
- Loads dataset
- Builds the model
- Prints parameter count and FLOPs
- Trains the model using SGD
- Saves trained weights

### evaluate.py
- Loads trained weights
- Runs inference on test data
- Computes and prints accuracy

---

## ⚙️ Build Instructions

From the `cpp/` directory:

```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC \
$(python3 -m pybind11 --includes) \
tensor.cpp autograd.cpp ops.cpp nn.cpp cnn.cpp \
loss.cpp optim.cpp bindings.cpp \
-o ../python/deep_framework$(python3-config --extension-suffix)


python3 train.py ./data_1

python3 evaluate.py ./data_1 model_final.pkl
