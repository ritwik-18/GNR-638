// #ifndef OPTIM_HPP
// #define OPTIM_HPP

// #include "tensor.hpp"
// #include <vector>

// class Optimizer {
// public:
//     std::vector<TensorPtr> params;

//     Optimizer(const std::vector<TensorPtr>& parameters)
//         : params(parameters) {}

//     virtual void step() = 0;

//     virtual void zero_grad() {
//         for (auto& p : params) {
//             if (!p->requires_grad) continue;
//             for (int i = 0; i < p->size(); ++i)
//                 p->grad[i] = 0.f;
//         }
//     }

//     virtual ~Optimizer() = default;
// };


#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <functional>

// =========================================================
// THREADING ENGINE (Reused from cnn.cpp)
// =========================================================
static void optim_parallel_for(int start, int end, std::function<void(int)> func) {
    int total = end - start;
    if (total <= 0) return;

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;

    // Small tasks run on main thread to avoid overhead
    if (total < (int)n_threads * 1024) { 
        for (int i = start; i < end; ++i) func(i);
        return;
    }

    std::vector<std::thread> threads;
    int chunk_size = (total + n_threads - 1) / n_threads;

    for (unsigned int i = 0; i < n_threads; ++i) {
        int range_start = start + i * chunk_size;
        int range_end = std::min(range_start + chunk_size, end);

        if (range_start >= end) break;

        threads.emplace_back([=]() {
            for (int k = range_start; k < range_end; ++k) {
                func(k);
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

// ========================
// BASE OPTIMIZER
// ========================
class Optimizer {
public:
    std::vector<TensorPtr> params;

    Optimizer(const std::vector<TensorPtr>& parameters)
        : params(parameters) {}

    virtual void step() = 0;

    virtual void zero_grad() {
        for (auto& p : params) {
            if (!p->requires_grad) continue;
            
            // Safety check
            if (p->grad.size() != p->data.size()) {
                p->grad.resize(p->data.size(), 0.0f);
            } else {
                std::fill(p->grad.begin(), p->grad.end(), 0.0f);
            }
        }
    }

    virtual ~Optimizer() = default;
};

// ========================
// ADAM OPTIMIZER
// ========================
class Adam : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;

    // History buffers (Momentum & Velocity)
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;

    Adam(const std::vector<TensorPtr>& parameters,
         float learning_rate = 0.001f,
         float b1 = 0.9f,
         float b2 = 0.999f,
         float epsilon = 1e-8f)
        : Optimizer(parameters),
          lr(learning_rate),
          beta1(b1),
          beta2(b2),
          eps(epsilon),
          t(0)
    {
        // Initialize history for every parameter
        for (auto& p : params) {
            if (p->requires_grad) {
                m.push_back(std::vector<float>(p->size(), 0.f));
                v.push_back(std::vector<float>(p->size(), 0.f));
            } else {
                m.push_back({});
                v.push_back({});
            }
        }
    }

    void step() override {
        t++;

        // 1. Calculate corrections ONCE per step (Optimization)
        float correction1 = 1.0f - std::pow(beta1, t);
        float correction2 = 1.0f - std::pow(beta2, t);

        // Iterate over parameters
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            if (!p->requires_grad) continue;

            // Get raw pointers for speed inside lambda
            float* p_data = p->data.data();
            float* p_grad = p->grad.data();
            float* m_data = m[pi].data();
            float* v_data = v[pi].data();
            int size = p->size();

            // 2. Parallelize the update loop
            optim_parallel_for(0, size, [=](int i) {
                float g = p_grad[i];

                // Update biased first moment estimate
                m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * g;

                // Update biased second raw moment estimate
                v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * g * g;

                // Compute bias-corrected first moment estimate
                float m_hat = m_data[i] / correction1;

                // Compute bias-corrected second raw moment estimate
                float v_hat = v_data[i] / correction2;

                // Update parameters
                p_data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            });
        }
    }
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