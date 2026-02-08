#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>
#include <string>

class Tensor {
public:
    // --- Constructors ---
    Tensor();
    Tensor(const std::vector<int>& shape, bool requires_grad = false);
    Tensor(const std::vector<float>& data,
           const std::vector<int>& shape,
           bool requires_grad = false);

    // --- Core data ---
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;
    bool requires_grad;

    // --- Shape helpers ---
    int ndim() const;
    int size() const;

    // --- Gradient helpers ---
    void zero_grad();
    bool has_grad() const;

    // --- Utility ---
    std::string shape_str() const;
};

#endif // TENSOR_HPP
