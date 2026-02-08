# 🔥 Custom Autograd + CNN Framework

## 📘 COMPLETE IMPLEMENTATION README

This README is the **single implementation contract** for the entire project.
It specifies **exactly what must be written in every file**, what responsibilities each file has, and how all components connect.

The framework is built in strict layers:

```
Tensor Storage → Autograd Engine → Tensor Ops → NN Layers → Python Bindings → Training Pipeline
```

⚠️ **Golden Rule:**
Never implement CNN layers before Tensor + Autograd + Basic Ops are working.

---

# 📁 PROJECT STRUCTURE

```
cpp/
 ├── tensor.hpp
 ├── tensor.cpp
 ├── autograd.hpp
 ├── autograd.cpp
 ├── ops.hpp
 ├── ops.cpp
 ├── nn.hpp
 ├── nn.cpp
 └── bindings.cpp

python/
 ├── framework.py
 ├── dataset.py
 ├── model.py
 ├── train.py
 └── evaluate.py
```

---

# 🧱 C++ IMPLEMENTATION

---

# ✅ tensor.hpp

## Purpose

Defines the **Tensor container**.
Only storage + interface declarations live here.

## MUST DEFINE

### Class

```
class Tensor
```

### Data Members

```
vector<float> data;
vector<float> grad;
vector<int> shape;

bool requires_grad;

vector<Tensor*> parents;
function<void()> backward_fn;
```

### Method Declarations

```
Tensor(vector<int> shape, bool requires_grad=false);
int numel() const;
void zero_grad();
```

## MUST NOT CONTAIN

* math operations
* backward traversal logic

---

# ✅ tensor.cpp

## Purpose

Implements tensor storage utilities.

## MUST IMPLEMENT

```
Tensor constructor → allocate data + grad
int numel()
void zero_grad()
reshape helper
indexing helper (flat indexing)
```

## Includes

```
#include "tensor.hpp"
```

---

# 🧠 autograd.hpp

## Purpose

Declares the gradient engine.

## MUST DECLARE

```
void backward(Tensor& loss);
void topo_sort(Tensor* node, vector<Tensor*>& graph);
```

No implementation here.

---

# 🧠 autograd.cpp

## Purpose

Executes backward propagation through computation graph.

## MUST IMPLEMENT

### topo_sort

* DFS through `parents`
* Build ordered graph list

### backward

```
loss.grad = 1
create topo order
iterate reversed order
if backward_fn exists → call it
```

## Includes

```
tensor.hpp
autograd.hpp
```

---

# ⚙️ ops.hpp

## Purpose

Declare all tensor operations.

## MUST DECLARE

### PHASE A — Core Ops

```
Tensor add(const Tensor&, const Tensor&);
Tensor relu(const Tensor&);
```

### PHASE B — Linear Algebra

```
Tensor matmul(const Tensor&, const Tensor&);
Tensor flatten(const Tensor&);
Tensor reshape(const Tensor&, vector<int>);
```

### PHASE C — CNN Ops

```
Tensor conv2d(...);
Tensor maxpool(...);
```

No implementations here.

---

# ⚙️ ops.cpp

## Purpose

Implements all mathematical operations + autograd behavior.

Every op MUST:

```
create output Tensor
assign parents
define backward_fn lambda
```

---

## PHASE A — CORE

Implement:

```
add forward
add backward_fn

relu forward
relu backward_fn
```

---

## PHASE B — LINEAR

Implement:

```
matmul forward/backward
flatten
reshape
```

---

## PHASE C — CNN (NAIVE)

Implement:

```
conv2d forward/backward
maxpool forward/backward
```

⚠️ Use simple loops. No optimizations required.

---

# 🧩 nn.hpp

## Purpose

High-level neural network abstraction.

## MUST DEFINE

### Base Class

```
class Module {
public:
    virtual Tensor forward(Tensor x)=0;
};
```

### Layers

```
class Linear;
class ReLU;
class Sequential;
class Conv2D;
class MaxPool;
```

### Training Components

```
class SGD;
Tensor cross_entropy(Tensor logits, Tensor targets);
```

### Metrics

```
size_t count_parameters(Module&);
size_t compute_flops(Module&);
```

---

# 🧩 nn.cpp

## Purpose

Implements neural network layers using ops.

---

## PHASE A — BASIC NN

Implement:

```
Linear::forward → matmul + add
ReLU::forward → relu
Sequential::forward → sequential execution
```

---

## PHASE B — CNN WRAPPERS

Implement:

```
Conv2D::forward → call conv2d op
MaxPool::forward → call maxpool op
```

Do NOT write math here.

---

## PHASE C — TRAINING LOGIC

Implement:

```
cross_entropy forward/backward
SGD optimizer step()
```

---

## PHASE D — METRICS

Implement:

```
count_parameters(Module&)
compute_flops(Module&)
```

---

# 🔗 bindings.cpp

## Purpose

Expose C++ API to Python using pybind11.

---

## FIRST VERSION EXPORTS

```
Tensor
Linear
Sequential
backward
```

---

## FINAL VERSION EXPORTS

```
Conv2D
MaxPool
SGD
cross_entropy
count_parameters
compute_flops
```

## Includes

```
tensor.hpp
nn.hpp
autograd.hpp
```

---

# 🐍 PYTHON IMPLEMENTATION

---

# 🐍 framework.py

## Purpose

Thin wrapper around compiled module.

## MUST CONTAIN

```
import deepframework_cpp
```

Optional aliases allowed.

No training logic here.

---

# 🐍 dataset.py

## Purpose

Load and preprocess images.

## MUST IMPLEMENT

```
load_dataset(folder_path)
infer_labels()
resize_to_32x32()
to_tensor()
batch_loader()
measure_loading_time()
```

Responsibilities:

* read images
* assign labels
* batching
* timing dataset loading

---

# 🧠 model.py

## Purpose

Define CNN architecture.

## MUST USE

```
Conv2D
ReLU
MaxPool
Flatten
Linear
```

Only define model structure.
No training loop.

---

# 🚀 train.py

## Purpose

Training pipeline.

## MUST PERFORM

```
load dataset
build model
print parameter count
print FLOPs
create SGD optimizer
training loop:
    forward
    loss
    backward
    optimizer.step()
save weights
```

---

# 📊 evaluate.py

## Purpose

Evaluation script.

## MUST PERFORM

```
load weights
forward pass
compute accuracy
print metrics
```

Script must run without modifying code.

---

# 🔗 STRICT DEPENDENCY FLOW

```
tensor → autograd → ops → nn → bindings → python
```

Never reverse this order.

---

# 👥 TEAM OWNERSHIP

## 👤 Engineer A — Core Engine

Creates:

```
tensor.hpp
tensor.cpp
autograd.hpp
autograd.cpp
ops.hpp
ops.cpp
```

---

## 👤 Engineer B — Neural Network System

Creates:

```
nn.hpp
nn.cpp
```

---

## 👤 Engineer C — Python Integration

Creates:

```
bindings.cpp
framework.py
dataset.py
model.py
train.py
evaluate.py
```

---

# 🧨 FINAL IMPLEMENTATION CHECKLIST

## C++

* Tensor container
* Backward engine
* add / relu / matmul ops
* conv2d / maxpool ops
* Module abstraction
* Linear / Sequential layers
* Conv2D / MaxPool layers
* cross_entropy loss
* SGD optimizer
* Metrics
* Python bindings

## Python

* Framework wrapper
* Dataset loader
* CNN model definition
* Training pipeline
* Evaluation script

---

# 🎯 FINAL GOAL

After implementing all files, the framework must support:

```
Custom Tensor Autograd
CNN Forward + Backward
Python Training Interface
Metrics Reporting
```

Follow this README strictly to prevent circular dependencies and architectural issues.

