#include "loss.hpp"
#include <cmath>
#include <algorithm>

TensorPtr cross_entropy_loss(TensorPtr logits, TensorPtr labels) {

    int B = logits->shape[0];
    int C = logits->shape[1];

    TensorPtr loss = create_tensor({1}, true);

    std::vector<float> softmax(B*C);

    float total_loss = 0.f;

    for(int i=0;i<B;i++){

        float max_val = -1e9f;
        for(int j=0;j<C;j++)
            max_val = std::max(max_val, logits->data[i*C+j]);

        float sum_exp = 0.f;
        for(int j=0;j<C;j++){
            softmax[i*C+j] = std::exp(logits->data[i*C+j]-max_val);
            sum_exp += softmax[i*C+j];
        }

        for(int j=0;j<C;j++)
            softmax[i*C+j] /= sum_exp;

        int label = (int)labels->data[i];
        total_loss += -std::log(softmax[i*C+label] + 1e-9f);
    }

    loss->data[0] = total_loss / B;

    std::weak_ptr<Tensor> weak_loss = loss;
    loss->parents = {logits};

    loss->backward_fn = [weak_loss, logits, labels, softmax, B, C]() {

        auto pl = weak_loss.lock();
        if(!pl) return;

        float g = pl->grad[0] / B;

        for(int i=0;i<B;i++){

            int label = (int)labels->data[i];

            for(int j=0;j<C;j++){

                float grad = softmax[i*C+j];

                if(j==label)
                    grad -= 1.f;

                logits->grad[i*C+j] += grad * g;
            }
        }
    };

    return loss;
}
