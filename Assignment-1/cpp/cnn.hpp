#ifndef CNN_HPP
#define CNN_HPP

#include "nn.hpp"
#include "tensor.hpp"
#include <vector>

/* Convolutional Layer (2D) */
class Conv2D : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;

    int in_c, out_c;
    int kH, kW;
    int stride, padding;

    Conv2D(int in_c, int out_c, int kH, int kW,
           int stride=1, int padding=0);

    TensorPtr forward(TensorPtr x) override;
    std::vector<TensorPtr> parameters() override;
};

/* Average Pooling Layer */
class MeanPool2D : public Module {
public:
    int k, stride;
    MeanPool2D(int k, int stride);

    TensorPtr forward(TensorPtr x) override;
    std::vector<TensorPtr> parameters() override { return {}; }
};

/* Max Pooling Layer */
class MaxPool2D : public Module {
public:
    int k, stride;
    MaxPool2D(int k, int stride);

    TensorPtr forward(TensorPtr x) override;
    std::vector<TensorPtr> parameters() override { return {}; }
};

#endif