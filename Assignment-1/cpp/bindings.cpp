#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h> // <--- THIS WAS MISSING
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include "tensor.hpp"
#include "autograd.hpp"
#include "ops.hpp"
#include "nn.hpp"
#include "cnn.hpp"
#include "loss.hpp"
#include "optim.hpp"

namespace py = pybind11;

// Helper to bind std::vector<float> as a Python object (avoids copying)
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);

PYBIND11_MODULE(deep_framework, m) {
    m.doc() = "Deep Learning Framework C++ Backend";

    // ---------------------------------------
    // 1. Bind std::vector for efficiency
    // ---------------------------------------
    py::bind_vector<std::vector<float>>(m, "FloatVector");
    py::bind_vector<std::vector<int>>(m, "IntVector");

    // ---------------------------------------
    // 2. Bind Tensor
    // ---------------------------------------
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, bool>(), 
             py::arg("shape"), py::arg("requires_grad") = false)
        // Expose fields
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        // Expose methods
        .def("size", &Tensor::size)
        .def("ndim", &Tensor::ndim)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", [](TensorPtr self) { backward(self); }) 
        .def("__repr__", [](const Tensor& t) {
            return "<Tensor shape=" + t.shape_str() + 
                   ", requires_grad=" + (t.requires_grad ? "True" : "False") + ">";
        });

    // Factory function
    m.def("create_tensor", &create_tensor, 
          py::arg("shape"), py::arg("requires_grad") = false);

    // ---------------------------------------
    // 3. Bind Autograd & Ops
    // ---------------------------------------
    m.def("backward", &backward, "Compute backpropagation");

    // Math Ops
    m.def("add", &add);
    m.def("relu", &relu);
    m.def("matmul", &matmul);
    m.def("flatten", &flatten);
    m.def("reshape", &reshape);
    m.def("sum", &sum);

    // ---------------------------------------
    // 4. Bind Layers (Modules)
    // ---------------------------------------
    
    // Base Module
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("parameters", &Module::parameters);

    // Linear
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"))
        .def_readwrite("weight", &Linear::weight)
        .def_readwrite("bias", &Linear::bias);

    // ReLU Layer
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());

    // Sequential
    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<const std::vector<Module*>&>());

    // Conv2D
    py::class_<Conv2D, Module, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<int, int, int, int, int, int>(),
             py::arg("in_c"), py::arg("out_c"), 
             py::arg("kH"), py::arg("kW"), 
             py::arg("stride") = 1, py::arg("padding") = 0)
        .def_readwrite("weight", &Conv2D::weight)
        .def_readwrite("bias", &Conv2D::bias);

    // MaxPool2D
    py::class_<MaxPool2D, Module, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<int, int>(), py::arg("k"), py::arg("stride"));

    // MeanPool2D
    py::class_<MeanPool2D, Module, std::shared_ptr<MeanPool2D>>(m, "MeanPool2D")
        .def(py::init<int, int>(), py::arg("k"), py::arg("stride"));

    // ---------------------------------------
    // 5. Bind Loss & Optimizer
    // ---------------------------------------
    m.def("cross_entropy_loss", &cross_entropy_loss);

    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad);

    py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
        .def(py::init<const std::vector<TensorPtr>&, float>(),
             py::arg("parameters"), py::arg("lr"));
}   