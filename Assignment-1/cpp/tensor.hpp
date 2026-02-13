#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <numeric> 
#include <sstream>
#include <stdexcept>
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;
    bool requires_grad;

    std::vector<TensorPtr> parents;
    std::function<void()> backward_fn;

    // Constructors
    Tensor(); // Default
    Tensor(const std::vector<int>& shape, bool requires_grad=false);
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad=false);

    int size() const;
    int ndim() const;
    void zero_grad();
    bool has_grad() const;
    std::string shape_str() const;
};

// Factory helper
inline TensorPtr create_tensor(const std::vector<int>& shape, bool requires_grad=false) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

#endif