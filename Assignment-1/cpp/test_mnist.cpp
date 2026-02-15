#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "tensor.hpp"
#include "ops.hpp"
#include "cnn.hpp"
#include "nn.hpp"
#include "mnist_loader.hpp"
#include "autograd.hpp"
#include "loss.hpp"
#include "optim.hpp"      // ⭐ NEW

// =====================================================
// PREDICT DIGIT
// =====================================================
int predict_digit(Conv2D& c1, MaxPool2D& p1, Linear& fc, TensorPtr img){

    int flat_size = 16*13*13;

    TensorPtr f1 = c1.forward(img);
    f1 = relu(f1);
    f1 = p1.forward(f1);

    TensorPtr flat = reshape(f1,{1,flat_size});
    TensorPtr out  = fc.forward(flat);

    int pred = 0;
    float best = out->data[0];

    for(int j=1;j<10;j++){
        float v = out->data[j];
        if(v > best){
            best = v;
            pred = j;
        }
    }
    return pred;
}

// =====================================================
// MINI-BATCH SLICE
// =====================================================
TensorPtr slice_batch(TensorPtr src, int start, int count){

    std::vector<int> shape = src->shape;
    shape[0] = count;

    TensorPtr out = create_tensor(shape,false);

    int single_size = src->size() / src->shape[0];

    for(int i=0;i<count;i++){
        std::copy(
            src->data.begin() + (start+i)*single_size,
            src->data.begin() + (start+i+1)*single_size,
            out->data.begin() + i*single_size
        );
    }
    return out;
}

// =====================================================
// LOAD CSV ROW -> TENSOR
// =====================================================
bool load_csv_row(std::ifstream& file, TensorPtr& img, int& label){

    static bool header_skipped = false;

    std::string line;

    while(std::getline(file,line)){

        if(!header_skipped){
            header_skipped = true;
            if(line.find("label") != std::string::npos)
                continue;
        }

        if(line.empty()) continue;

        std::stringstream ss(line);
        std::string val;

        img = create_tensor({1,1,28,28},false);

        if(!std::getline(ss,val,',')) return false;

        try{
            label = std::stoi(val);
        }
        catch(...){
            continue;
        }

        int idx = 0;
        while(std::getline(ss,val,',')){
            if(idx < 784)
                img->data[idx++] = std::stof(val) / 255.f;
        }

        return true;
    }

    return false;
}

// =====================================================
// MAIN
// =====================================================
int main(){

    std::cout << "==== MNIST TRAIN START ====\n";

    int TOTAL = 7500;
    int MINI_BATCH = 64;

    auto batch = load_mnist_batch(
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        TOTAL);

    TensorPtr images = batch.images;
    TensorPtr labels = batch.labels;

    // =================================================
    // MODEL
    // =================================================
    Conv2D c1(1,16,3,3,1,0);
    MaxPool2D p1(2,2);

    int flat_size = 16*13*13;
    Linear fc(flat_size,10);

    // =================================================
    // PARAMETERS
    // =================================================
    std::vector<TensorPtr> params;

    auto p_c1 = c1.parameters();
    params.insert(params.end(), p_c1.begin(), p_c1.end());

    auto p_fc = fc.parameters();
    params.insert(params.end(), p_fc.begin(), p_fc.end());

    float lr = 0.05f;

    // ⭐ NEW OPTIMIZER
    Adam Optimizer(params, lr);

    // =================================================
    // TRAIN LOOP
    // =================================================
    for(int epoch=0; epoch<20; epoch++){

        float epoch_loss = 0.f;
        int correct = 0;
        int seen = 0;

        for(int start=0; start<TOTAL; start+=MINI_BATCH){

            int bs = std::min(MINI_BATCH, TOTAL-start);

            TensorPtr x = slice_batch(images,start,bs);
            TensorPtr y = slice_batch(labels,start,bs);

            TensorPtr f1 = c1.forward(x);
            f1 = relu(f1);
            f1 = p1.forward(f1);

            TensorPtr flat = reshape(f1,{bs,flat_size});
            TensorPtr out  = fc.forward(flat);

            TensorPtr loss = cross_entropy_loss(out,y);
            epoch_loss += loss->data[0] * bs;

            for(int i=0;i<bs;i++){
                int pred = 0;
                float best = out->data[i*10];

                for(int j=1;j<10;j++){
                    float v = out->data[i*10+j];
                    if(v>best){ best=v; pred=j; }
                }

                if(pred==(int)y->data[i]) correct++;
            }

            seen += bs;

            // ⭐ NEW OPTIMIZER FLOW
            Optimizer.zero_grad();
            backward(loss);
            Optimizer.step();
        }

        std::cout
            << "Epoch " << epoch
            << " | Loss " << epoch_loss/seen
            << " | Acc "  << (float)correct/seen
            << "\n";
    }

    std::cout << "==== MNIST TRAIN END ====\n";

    // =====================================================
    // CSV REAL TESTING
    // =====================================================
    std::cout << "\n==== CSV REAL DIGIT TEST ====\n";

    std::ifstream file("mnist_test.csv");

    if(!file.is_open()){
        std::cout<<"Could not open mnist_test.csv\n";
        return 0;
    }

    int total_test = 0;
    int correct_test = 0;

    while(true){

        TensorPtr img;
        int label;

        if(!load_csv_row(file,img,label)) break;

        int pred = predict_digit(c1,p1,fc,img);

        std::cout
            << "Expected = " << label
            << " | Predicted = " << pred
            << "\n";

        if(pred==label) correct_test++;
        total_test++;

        if(total_test==20) break;
    }

    std::cout
        << "\nTest Accuracy = "
        << (float)correct_test/total_test
        << "\n";
}
