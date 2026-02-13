#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

TensorPtr cross_entropy_loss(TensorPtr logits, TensorPtr labels);

#endif
