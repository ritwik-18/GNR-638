#include <iostream>
#include <vector>
#include <iomanip>

#include "tensor.hpp"
#include "ops.hpp"
#include "cnn.hpp"
#include "autograd.hpp"

// ======================================================
// SGD
// ======================================================
void optimizer_step(std::vector<TensorPtr>& params, float lr) {
    for (auto& p : params) {
        if (!p->requires_grad) continue;
        for (int i = 0; i < p->size(); ++i) {
            p->data[i] -= lr * p->grad[i];
            p->grad[i] = 0.0f;
        }
    }
}

// ======================================================
// Manual square op (LOCAL TEST ONLY)
// ======================================================
TensorPtr square_local(TensorPtr x) {

    auto out = create_tensor(x->shape, x->requires_grad);

    for(int i=0;i<x->size();i++)
        out->data[i] = x->data[i] * x->data[i];

    if(out->requires_grad) {
        std::weak_ptr<Tensor> weak_out = out;
        out->parents = {x};

        out->backward_fn = [x,weak_out]() {
            auto pout = weak_out.lock();
            if(!pout) return;

            for(int i=0;i<x->size();i++)
                x->grad[i] += 2.f * x->data[i] * pout->grad[i];
        };
    }
    return out;
}

// ======================================================
// MSE LOSS USING ONLY add + sum
// ======================================================
TensorPtr mse_loss(TensorPtr pred, TensorPtr target) {

    TensorPtr neg = create_tensor(target->shape,false);

    for(int i=0;i<target->size();i++)
        neg->data[i] = -target->data[i];

    TensorPtr diff = add(pred,neg);
    TensorPtr err  = square_local(diff);

    TensorPtr loss = sum(err);
    loss->data[0] /= pred->size();

    return loss;
}

// ======================================================
// DATASET
// ======================================================
void get_batch(TensorPtr& x, TensorPtr& y) {

    std::fill(x->data.begin(),x->data.end(),0.f);
    std::fill(y->data.begin(),y->data.end(),0.f);

    // vertical line
    for(int h=0;h<4;h++)
        x->data[0*16 + h*4 + 1] = 1.f;

    y->data[0] = 1.f;

    // horizontal line
    for(int w=0;w<4;w++)
        x->data[1*16 + 4 + w] = 1.f;

    y->data[1] = 0.f;
}

// ======================================================
// MAIN
// ======================================================
int main(){

    std::cout<<"==== TRAIN TEST START ====\n";

    TensorPtr inputs  = create_tensor({2,1,4,4},false);
    TensorPtr targets = create_tensor({2,1,1,1},false);

    get_batch(inputs,targets);

    Conv2D conv(1,1,3,3,1,0);
    MeanPool2D pool(2,2);

    std::vector<TensorPtr> params = conv.parameters();

    float lr = 0.1f;

    for(int epoch=0;epoch<100;epoch++){

        TensorPtr f   = conv.forward(inputs);
        TensorPtr out = pool.forward(f);

        TensorPtr loss = mse_loss(out,targets);

        if(epoch%10==0)
            std::cout<<"Epoch "<<epoch
                     <<" | Loss "<<loss->data[0]
                     <<" | Pred ["<<out->data[0]<<","<<out->data[1]<<"]\n";

        backward(loss);
        optimizer_step(params,lr);
    }

    auto final = pool.forward(conv.forward(inputs));

    std::cout<<"Target [1,0]\n";
    std::cout<<"Result ["<<final->data[0]<<","<<final->data[1]<<"]\n";

    std::cout<<"==== TRAIN TEST END ====\n";
}
