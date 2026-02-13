#include "nn.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <random>

// ====================
// Linear
// ====================
Linear::Linear(int in_features, int out_features) {

    weight = std::make_shared<Tensor>(
        std::vector<int>{in_features, out_features}, true);

    bias = std::make_shared<Tensor>(
        std::vector<int>{out_features}, true);

    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-0.1f,0.1f);

    for(int i=0;i<weight->size();++i)
        weight->data[i]=dist(gen);

    for(int i=0;i<bias->size();++i)
        bias->data[i]=0.f;
}

TensorPtr Linear::forward(TensorPtr x) {

    // 1️⃣ matmul builds graph correctly (parents = {x, weight})
    TensorPtr out = matmul(x, weight);

    int batch = x->shape[0];
    int out_features = bias->shape[0];

    // 2️⃣ Add bias (forward only)
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < out_features; ++j) {
            out->data[i * out_features + j] += bias->data[j];
        }
    }

    // 3️⃣ Attach bias backward WITHOUT touching parents
    if (bias->requires_grad) {

        std::weak_ptr<Tensor> weak_out = out;
        TensorPtr pb = bias;

        auto old_backward = out->backward_fn;

        out->backward_fn =
        [old_backward, pb, weak_out, batch, out_features]() {

            // First run matmul backward (VERY IMPORTANT)
            if (old_backward) old_backward();

            auto pout = weak_out.lock();
            if (!pout) return;

            // Compute bias gradient
            for (int j = 0; j < out_features; ++j) {

                float grad_sum = 0.f;

                for (int i = 0; i < batch; ++i)
                    grad_sum += pout->grad[i*out_features + j];

                pb->grad[j] += grad_sum;
            }
        };
    }

    return out;
}


std::vector<TensorPtr> Linear::parameters() {
    return { weight, bias };
}

// ====================
// ReLU
// ====================
TensorPtr ReLU::forward(TensorPtr x) {
    return relu(x);
}

// ====================
// Sequential
// ====================
Sequential::Sequential(const std::vector<Module*>& layers_)
    : layers(layers_) {}

TensorPtr Sequential::forward(TensorPtr x) {
    TensorPtr out = x;
    for (Module* layer : layers) {
        out = layer->forward(out);
    }
    return out;
}

std::vector<TensorPtr> Sequential::parameters() {
    std::vector<TensorPtr> params;
    for (Module* layer : layers) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}