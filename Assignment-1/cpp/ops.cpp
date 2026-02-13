#include "ops.hpp"
#include <stdexcept>

// --------------------
// add
// --------------------
TensorPtr add(TensorPtr a, TensorPtr b) {
    if (a->size() != b->size()) throw std::runtime_error("add: size mismatch");

    auto out = create_tensor(a->shape, a->requires_grad || b->requires_grad);

    for (int i = 0; i < a->size(); ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }

    if (out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {a, b};

        out->backward_fn = [a, b, weak_out]() {
            auto pout = weak_out.lock();
            if (!pout) return;
            for (int i = 0; i < pout->size(); ++i) {
                float g = pout->grad[i];
                if (a->requires_grad) a->grad[i] += g;
                if (b->requires_grad) b->grad[i] += g;
            }
        };
    }
    return out;
}

// --------------------
// relu
// --------------------
TensorPtr relu(TensorPtr x) {
    auto out = create_tensor(x->shape, x->requires_grad);

    for (int i = 0; i < x->size(); ++i) {
        out->data[i] = (x->data[i] > 0.0f) ? x->data[i] : 0.0f;
    }

    if (out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {x};

        out->backward_fn = [x, weak_out]() {
            auto pout = weak_out.lock();
            if (!pout) return;
            for (int i = 0; i < pout->size(); ++i) {
                if (x->data[i] > 0.0f) x->grad[i] += pout->grad[i];
            }
        };
    }
    return out;
}

// --------------------
// matmul
// --------------------
TensorPtr matmul(TensorPtr a, TensorPtr b) {
    if (a->ndim() != 2 || b->ndim() != 2) throw std::runtime_error("matmul: 2D only");
    if (a->shape[1] != b->shape[0]) throw std::runtime_error("matmul: K mismatch");

    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];

    auto out = create_tensor({M, N}, a->requires_grad || b->requires_grad);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }

    if (out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {a, b};

        out->backward_fn = [a, b, weak_out, M, K, N]() {
            auto pout = weak_out.lock();
            if (!pout) return;

            if (a->requires_grad) {
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < K; ++k) {
                        float g = 0.0f;
                        for (int j = 0; j < N; ++j) g += pout->grad[i * N + j] * b->data[k * N + j];
                        a->grad[i * K + k] += g;
                    }
                }
            }
            if (b->requires_grad) {
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        float g = 0.0f;
                        for (int i = 0; i < M; ++i) g += a->data[i * K + k] * pout->grad[i * N + j];
                        b->grad[k * N + j] += g;
                    }
                }
            }
        };
    }
    return out;
}

TensorPtr sum(TensorPtr x) {
    // Basic sum reduction implementation if needed
    auto out = create_tensor({1}, x->requires_grad);
    float s = 0;
    for(float v : x->data) s += v;
    out->data[0] = s;
    
    if(out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {x};
        out->backward_fn = [x, weak_out]() {
            auto pout = weak_out.lock();
            if(!pout) return;
            for(size_t i=0; i<x->grad.size(); ++i) x->grad[i] += pout->grad[0];
        };
    }
    return out;
}

TensorPtr flatten(TensorPtr x) {
    auto out = create_tensor({x->size()}, x->requires_grad);
    out->data = x->data;
    if (out->requires_grad) {
        out->parents = {x};
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_fn = [x, weak_out]() {
            auto pout = weak_out.lock();
            if (!pout) return;
            for (size_t i = 0; i < x->grad.size(); ++i)
                x->grad[i] += pout->grad[i];
        };
    }
    return out;
}
TensorPtr reshape(TensorPtr x, const std::vector<int>& shape) {
    int total = 1;
    for(int s : shape) total *= s;
    if(total != x->size()) throw std::runtime_error("reshape: size mismatch");
    auto out = create_tensor(shape, x->requires_grad);
    out->data = x->data;
    if (out->requires_grad) {
        out->parents = {x};
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_fn = [x, weak_out]() {
            auto pout = weak_out.lock();
            if (!pout) return;
            for (size_t i = 0; i < x->grad.size(); ++i)
                x->grad[i] += pout->grad[i];
        };
    }
    return out;
}