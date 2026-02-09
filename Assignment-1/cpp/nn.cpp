#include "nn.hpp"
#include "ops.hpp"
#include <stdexcept>

// ====================
// Linear
// ====================
Linear::Linear(int in_features, int out_features) {
    // Initialize pointers using make_shared
    weight = std::make_shared<Tensor>(std::vector<int>{in_features, out_features}, true);
    bias = std::make_shared<Tensor>(std::vector<int>{out_features}, true);

    // Init
    for (int i = 0; i < weight->size(); ++i) weight->data[i] = 0.01f;
    for (int i = 0; i < bias->size(); ++i) bias->data[i] = 0.0f;
}

TensorPtr Linear::forward(TensorPtr x) {
    // Assume matmul returns TensorPtr
    TensorPtr out = matmul(x, weight);

    int batch = x->shape[0];
    int out_features = bias->shape[0];

    // Add bias
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < out_features; ++j) {
            out->data[i * out_features + j] += bias->data[j];
        }
    }

    if (bias->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        TensorPtr pb = bias;
        
        // Save existing backward function
        auto old_backward = out->backward_fn;
        
        // Add bias to parents
        out->parents.push_back(pb);

        out->backward_fn = [old_backward, pb, weak_out, batch, out_features]() {
            if (old_backward) old_backward();

            auto pout = weak_out.lock();
            if (!pout) return;

            for (int j = 0; j < out_features; ++j) {
                float grad_sum = 0.0f;
                for (int i = 0; i < batch; ++i) {
                    grad_sum += pout->grad[i * out_features + j];
                }
                if (pb->requires_grad)
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