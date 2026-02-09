#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"
#include <vector>

// --------------------
// Core elementwise ops
// --------------------
TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr relu(TensorPtr x);
TensorPtr sum(TensorPtr x);

// --------------------
// Linear algebra ops
// --------------------
TensorPtr matmul(TensorPtr a, TensorPtr b);
TensorPtr flatten(TensorPtr x);
TensorPtr reshape(TensorPtr x, const std::vector<int>& shape);

#endif // OPS_HPP