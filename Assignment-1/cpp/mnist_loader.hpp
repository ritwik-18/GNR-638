#pragma once
#include <fstream>
#include <string>
#include <algorithm> // for std::min
#include <iostream>  // for logging
#include "tensor.hpp"

struct MNISTBatch {
    TensorPtr images; // (B,1,28,28)
    TensorPtr labels; // (B)
};

// Helper: Read big-endian int
inline int read_int(std::ifstream& f) {
    unsigned char b[4];
    f.read((char*)b, 4);
    return (int)((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
}

inline MNISTBatch load_mnist_batch(
    const std::string& img_file,
    const std::string& lbl_file,
    int count) 
{
    std::ifstream fi(img_file, std::ios::binary);
    std::ifstream fl(lbl_file, std::ios::binary);

    if (!fi.is_open()) {
        std::cerr << "Error: Could not open " << img_file << "\n";
        exit(1);
    }
    if (!fl.is_open()) {
        std::cerr << "Error: Could not open " << lbl_file << "\n";
        exit(1);
    }

    // 1. Read Image Header
    int magic_img = read_int(fi);
    int num_imgs  = read_int(fi);
    int H         = read_int(fi);
    int W         = read_int(fi);

    // 2. Read Label Header
    int magic_lbl = read_int(fl);
    int num_lbls  = read_int(fl);

    // Safety: Ensure we don't read more than exists
    int actual_count = std::min(count, num_imgs);

    // 3. Create Tensors
    TensorPtr imgs = create_tensor({actual_count, 1, H, W}, false);
    TensorPtr lbls = create_tensor({actual_count}, false);

    // 4. Read Data
    for (int i = 0; i < actual_count; i++) {
        // Read image pixels
        for (int p = 0; p < H * W; p++) {
            unsigned char v;
            fi.read((char*)&v, 1);
            imgs->data[i * H * W + p] = v / 255.0f;
        }

        // Read label
        unsigned char l;
        fl.read((char*)&l, 1);
        lbls->data[i] = (float)l;
    }

    std::cout << "Loaded " << actual_count << " samples from MNIST.\n";
    return {imgs, lbls};
}