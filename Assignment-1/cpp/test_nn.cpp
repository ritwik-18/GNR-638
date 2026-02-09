#include "nn.hpp"
#include "ops.hpp"
#include "autograd.hpp"
#include <iostream>

int main() {
    auto x = create_tensor({1, 3}, true);
    x->data = {1.0f, -2.0f, 3.0f};

    Linear l1(3, 2);
    ReLU r1;

    Sequential model({&l1, &r1});

    auto out = model.forward(x);
    auto loss = sum(out);

    backward(loss);

    std::cout << "Weight grad:\n";
    for (float g : l1.weight->grad) std::cout << g << " ";
    std::cout << "\nBias grad:\n";
    for (float g : l1.bias->grad) std::cout << g << " ";
    std::cout << "\n";
}
