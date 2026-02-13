#include <iostream>
#include "tensor.hpp"
#include "ops.hpp"
#include "cnn.hpp"
#include "autograd.hpp"

int main() {

    std::cout << "==== CNN TEST START ====\n";

    // --------------------------------------------------
    // 1️⃣ Create fake input tensor  (B=1,C=1,H=5,W=5)
    // --------------------------------------------------
    TensorPtr x = create_tensor({1,1,5,5}, true);

    for(int i=0;i<x->size();i++)
        x->data[i] = (float)(i+1);   // simple deterministic values

    std::cout << "Input shape: " << x->shape_str() << "\n";

    // --------------------------------------------------
    // 2️⃣ Create layers
    // --------------------------------------------------
    Conv2D conv(1,2,3,3,1,0);     // out channels = 2
    MeanPool2D pool(2,2);

    // --------------------------------------------------
    // 3️⃣ Forward pass
    // --------------------------------------------------
    TensorPtr y = conv.forward(x);
    std::cout << "After Conv shape: " << y->shape_str() << "\n";

    TensorPtr z = pool.forward(y);
    std::cout << "After Pool shape: " << z->shape_str() << "\n";

    // --------------------------------------------------
    // 4️⃣ Loss = sum(z)
    // --------------------------------------------------
    TensorPtr loss = sum(z);

    std::cout << "Loss value: " << loss->data[0] << "\n";

    // --------------------------------------------------
    // 5️⃣ Backward pass
    // --------------------------------------------------
    backward(loss);

    std::cout << "Backward finished.\n";

    // --------------------------------------------------
    // 6️⃣ Print some gradients
    // --------------------------------------------------
    std::cout << "Input grad[0]: " << x->grad[0] << "\n";
    std::cout << "Conv weight grad[0]: " << conv.weight->grad[0] << "\n";
    std::cout << "Conv bias grad[0]: " << conv.bias->grad[0] << "\n";

    std::cout << "==== CNN TEST END ====\n";

    return 0;
}
