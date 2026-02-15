#include "cnn.hpp"
#include "ops.hpp"
#include <cmath>
#include <random>
#include <cstring>
#include <thread>   // Standard C++ threading
#include <vector>
#include <functional>
#include <algorithm>
#include <memory>   // For std::shared_ptr
#include<iostream>
// =========================================================
// THREADING ENGINE: Custom Parallel For Loop
// =========================================================
static void parallel_for(int start, int end, std::function<void(int)> func) {
    int total = end - start;
    if (total <= 0) return;

    // Detect available cores
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4; // Safety fallback

    // If the task is tiny, run on main thread to avoid overhead
    if (total < (int)n_threads) {
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

    // Join all threads
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
}

// =========================================================
// UTILS & MATH
// =========================================================

static inline int idx4(int b, int c, int y, int x, int C, int H, int W) {
    return ((b * C + c) * H + y) * W + x;
}

static float randf() {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    return dist(gen);
}

/* Parallel Matrix Multiplication (C = A * B)
   Splits the rows of C across threads.
*/
static void gemm_cpu(const float* A, const float* B, float* C, 
                     int M, int N, int K, 
                     bool transA, bool transB) {
    
    parallel_for(0, M, [=](int i) {
        float* c_row = &C[i * N];
        for (int p = 0; p < K; ++p) {
            float a_val = transA ? A[p * M + i] : A[i * K + p];
            for (int j = 0; j < N; ++j) {
                float b_val = transB ? B[j * K + p] : B[p * N + j];
                c_row[j] += a_val * b_val;
            }
        }
    });
}

// =========================================================
// IM2COL & COL2IM
// =========================================================

static TensorPtr im2col(TensorPtr x, int kH, int kW, int stride, int padding) {
    int B = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    int rows = B * H_out * W_out;
    int cols = C * kH * kW;

    auto col = create_tensor({rows, cols}, false);

    // Parallelize over output pixels (rows of the column matrix)
    parallel_for(0, rows, [=](int row_idx) {
        int ox = row_idx % W_out;
        int oy = (row_idx / W_out) % H_out;
        int b = row_idx / (W_out * H_out);
        
        float* row_ptr = &col->data[row_idx * cols];
        int col_iter = 0;

        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < kH; ++ky) {
                for (int kx = 0; kx < kW; ++kx) {
                    int iy = oy * stride + ky - padding;
                    int ix = ox * stride + kx - padding;

                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        row_ptr[col_iter] = x->data[idx4(b, c, iy, ix, C, H, W)];
                    } else {
                        row_ptr[col_iter] = 0.0f;
                    }
                    col_iter++;
                }
            }
        }
    });

    return col;
}

/* Col2Im (Backward Pass)
   Parallelized over Batch and Channel to avoid race conditions.
*/
static void col2im(const std::vector<float>& col_grad, TensorPtr img_grad, 
                   int kH, int kW, int stride, int padding) {
    int B = img_grad->shape[0]; int C = img_grad->shape[1]; 
    int H = img_grad->shape[2]; int W = img_grad->shape[3];
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;
    int col_size = C * kH * kW;

    parallel_for(0, B * C, [=](int idx) {
        int c = idx % C;
        int b = idx / C;
        int col_offset = c * kH * kW; // Offset to this channel block in a col row

        for (int oy = 0; oy < H_out; ++oy) {
            for (int ox = 0; ox < W_out; ++ox) {
                int row_idx = b * H_out * W_out + oy * W_out + ox;
                const float* col_ptr = &col_grad[row_idx * col_size];
                
                for (int ky = 0; ky < kH; ++ky) {
                    for (int kx = 0; kx < kW; ++kx) {
                        int iy = oy * stride + ky - padding;
                        int ix = ox * stride + kx - padding;

                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            float val = col_ptr[col_offset + ky * kW + kx];
                            // No mutex needed: this thread exclusively owns this (b, c) slice
                            img_grad->grad[idx4(b, c, iy, ix, C, H, W)] += val;
                        }
                    }
                }
            }
        }
    });
}

// =========================================================
// CONV2D IMPLEMENTATION
// =========================================================

Conv2D::Conv2D(int in_c_, int out_c_, int kH_, int kW_, int stride_, int padding_)
    : in_c(in_c_), out_c(out_c_), kH(kH_), kW(kW_), stride(stride_), padding(padding_) 
{
    weight = create_tensor({out_c, in_c, kH, kW}, true);
    bias = create_tensor({out_c}, true);

    for (float& v : weight->data) v = randf();
    for (float& v : bias->data) v = 0.0f;
}

std::vector<TensorPtr> Conv2D::parameters() {
    return {weight, bias};
}

TensorPtr Conv2D::forward(TensorPtr x) {
    int B = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    // 1. Im2Col
    auto Xcol = im2col(x, kH, kW, stride, padding);

    // 2. GEMM (Xcol * W^T)
    auto out_linear = create_tensor({B * H_out * W_out, out_c}, true);
    gemm_cpu(Xcol->data.data(), weight->data.data(), out_linear->data.data(),
             B * H_out * W_out, out_c, in_c * kH * kW, 
             false, true);

    // 3. Reshape & Bias
    auto out = create_tensor({B, out_c, H_out, W_out}, 
                             x->requires_grad || weight->requires_grad || bias->requires_grad);

    parallel_for(0, B * out_c, [=](int idx) {
        int oc = idx % out_c;
        int b = idx / out_c;
        float b_val = bias->data[oc];

        for (int oy = 0; oy < H_out; ++oy) {
            for (int ox = 0; ox < W_out; ++ox) {
                int linear_idx = (b * H_out * W_out + oy * W_out + ox) * out_c + oc;
                out->data[idx4(b, oc, oy, ox, out_c, H_out, W_out)] = out_linear->data[linear_idx] + b_val;
            }
        }
    });

    // 4. Backward Pass
    if (out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {x, weight, bias};

        out->backward_fn = [=]() {
            auto pout = weak_out.lock();
            if (!pout) return;

            int M = B * H_out * W_out;
            int N = out_c;
            int K = in_c * kH * kW;

            // Flatten incoming gradient
            std::vector<float> dY_flat(M * N);
            parallel_for(0, M, [=, &dY_flat](int i) {
                for (int j = 0; j < N; ++j) {
                    int ox = i % W_out;
                    int oy = (i / W_out) % H_out;
                    int b = i / (W_out * H_out);
                    dY_flat[i * N + j] = pout->grad[idx4(b, j, oy, ox, out_c, H_out, W_out)];
                }
            });

            // Bias Gradients
            if (bias->requires_grad) {
                std::vector<float> db(out_c, 0.0f);
                // Serial accumulation for safety (fast enough for small vector)
                for(int i=0; i<M; ++i) {
                    for(int c=0; c<out_c; ++c) db[c] += dY_flat[i*out_c + c];
                }
                for(int c=0; c<out_c; ++c) bias->grad[c] += db[c];
            }

            // Weight Gradients
            if (weight->requires_grad) {
                auto Xcol_replay = im2col(x, kH, kW, stride, padding);
                gemm_cpu(dY_flat.data(), Xcol_replay->data.data(), weight->grad.data(),
                         out_c, K, M, true, false);
            }

            // Input Gradients (CRITICAL for stacking layers)
            if (x->requires_grad) {
                std::vector<float> dX_col(M * K, 0.0f);
                gemm_cpu(dY_flat.data(), weight->data.data(), dX_col.data(),
                         M, K, out_c, false, false);
                col2im(dX_col, x, kH, kW, stride, padding);
            }
        };
    }

    return out;
}

// =========================================================
// POOLING IMPLEMENTATIONS
// =========================================================

MeanPool2D::MeanPool2D(int k_, int stride_) : k(k_), stride(stride_) {}

TensorPtr MeanPool2D::forward(TensorPtr x) {
    int B = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;

    auto out = create_tensor({B, C, H_out, W_out}, x->requires_grad);

    parallel_for(0, B * C, [=](int idx) {
        int c = idx % C;
        int b = idx / C;
        for (int oy = 0; oy < H_out; ++oy) {
            for (int ox = 0; ox < W_out; ++ox) {
                float s = 0;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        s += x->data[idx4(b, c, oy*stride+ky, ox*stride+kx, C, H, W)];
                    }
                }
                out->data[idx4(b, c, oy, ox, C, H_out, W_out)] = s / (k * k);
            }
        }
    });

    if (out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {x};
        out->backward_fn = [=]() {
            auto pout = weak_out.lock(); if(!pout) return;
            float scale = 1.0f / (k * k);

            parallel_for(0, B * C, [=](int idx) {
                int c = idx % C; int b = idx / C;
                for (int oy = 0; oy < H_out; ++oy) {
                    for (int ox = 0; ox < W_out; ++ox) {
                        float g = pout->grad[idx4(b, c, oy, ox, C, H_out, W_out)] * scale;
                        for (int ky = 0; ky < k; ++ky) {
                            for (int kx = 0; kx < k; ++kx) {
                                x->grad[idx4(b, c, oy*stride+ky, ox*stride+kx, C, H, W)] += g;
                            }
                        }
                    }
                }
            });
        };
    }
    return out;
}

MaxPool2D::MaxPool2D(int k_, int stride_) : k(k_), stride(stride_) {}

TensorPtr MaxPool2D::forward(TensorPtr x) {
    int B = x->shape[0]; int C = x->shape[1]; int H = x->shape[2]; int W = x->shape[3];
    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;

    auto out = create_tensor({B, C, H_out, W_out}, x->requires_grad);
    // Use shared_ptr to avoid deep copying the index vector into lambda
    auto max_indices = std::make_shared<std::vector<int>>(out->size());

    parallel_for(0, B * C, [=](int idx) {
        int c = idx % C; int b = idx / C;
        for (int oy = 0; oy < H_out; ++oy) {
            for (int ox = 0; ox < W_out; ++ox) {
                float best = -1e9; int best_idx = -1;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        int id = idx4(b, c, oy*stride+ky, ox*stride+kx, C, H, W);
                        float v = x->data[id];
                        if (v > best) { best = v; best_idx = id; }
                    }
                }
                int oid = idx4(b, c, oy, ox, C, H_out, W_out);
                out->data[oid] = best;
                (*max_indices)[oid] = best_idx;
            }
        }
    });

    if (out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {x};
        out->backward_fn = [=]() {
            auto pout = weak_out.lock(); if(!pout) return;
            
            // Parallelize over batches to avoid race conditions if stride < k
            parallel_for(0, B, [=](int b) {
                int start = b * C * H_out * W_out;
                int end = (b+1) * C * H_out * W_out;
                for(int i = start; i < end; ++i) {
                    x->grad[(*max_indices)[i]] += pout->grad[i];
                }
            });
        };
    }
    return out;
}