#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "tensor.hpp"
#include <vector>

class Optimizer {
public:
    std::vector<TensorPtr> params;

    Optimizer(const std::vector<TensorPtr>& parameters)
        : params(parameters) {}

    virtual void step() = 0;

    virtual void zero_grad() {
        for (auto& p : params) {
            if (!p->requires_grad) continue;
            for (int i = 0; i < p->size(); ++i)
                p->grad[i] = 0.f;
        }
    }

    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
public:
    float lr;

    SGD(const std::vector<TensorPtr>& parameters, float learning_rate)
        : Optimizer(parameters), lr(learning_rate) {}

    void step() override {
        for (auto& p : params) {
            if (!p->requires_grad) continue;
            for (int i = 0; i < p->size(); ++i)
                p->data[i] -= lr * p->grad[i];
        }
    }
};

#endif
