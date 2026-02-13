#include "tensor.hpp"
#include <numeric>
#include <sstream>
#include <stdexcept>
// -------- Constructors --------
Tensor::Tensor()
    : requires_grad(false) {}

Tensor::Tensor(const std::vector<int>& shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_) {

    int total = size();
    data.resize(total, 0.0f);

    if (requires_grad) {
        grad.resize(total, 0.0f);
    }
}
Tensor::Tensor(const std::vector<float>& data_,
               const std::vector<int>& shape_,
               bool requires_grad_)
    : data(data_), shape(shape_), requires_grad(requires_grad_) {

    if (data.size() != size()) {
        throw std::runtime_error("Tensor data size does not match shape");
    }

    if (requires_grad) {
        grad.resize(size(), 0.0f);
    }
}

// -------- Helpers --------

int Tensor::ndim() const {
    return static_cast<int>(shape.size());
}

int Tensor::size() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void Tensor::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
}

bool Tensor::has_grad() const {
    return requires_grad && !grad.empty();
}

std::string Tensor::shape_str() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) oss << ", ";
    }
    oss << ")";
    return oss.str();
}