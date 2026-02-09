#ifndef NN_HPP
#define NN_HPP

#include "tensor.hpp"
#include <vector>

class Module {
public:
    virtual TensorPtr forward(TensorPtr x) = 0;
    virtual std::vector<TensorPtr> parameters() { return {}; }
    virtual ~Module() = default;
};

class Linear : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;

    Linear(int in_features, int out_features);
    TensorPtr forward(TensorPtr x) override;
    std::vector<TensorPtr> parameters() override;
};

class ReLU : public Module {
public:
    TensorPtr forward(TensorPtr x) override;
};

class Sequential : public Module {
public:
    std::vector<Module*> layers; // Using raw pointers for layers is fine here

    Sequential(const std::vector<Module*>& layers);
    TensorPtr forward(TensorPtr x) override;
    std::vector<TensorPtr> parameters() override;
};

#endif