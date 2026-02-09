#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <vector>
#include "tensor.hpp"

// Build computation graph in topological order
void topo_sort(TensorPtr node, std::vector<TensorPtr>& graph);

// Entry point for reverse-mode autodiff
// CHANGED: Now accepts TensorPtr instead of Tensor&
void backward(TensorPtr loss);

#endif // AUTOGRAD_HPP