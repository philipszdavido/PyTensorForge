#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "kernels/MatMul.hpp"

namespace py = pybind11;

py::array_t<float> matmul_py(
    py::array_t<float> a,
    py::array_t<float> b)
{
    auto A = a.request();
    auto B = b.request();

    if (A.ndim != 2 || B.ndim != 2)
        throw std::runtime_error("Only 2D matrices supported.");

    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    if (B.shape[0] != K)
        throw std::runtime_error("Shape mismatch.");

    py::array_t<float> c({M, N});

    auto C = c.request();

    matmul(
        static_cast<float*>(A.ptr),
        static_cast<float*>(B.ptr),
        static_cast<float*>(C.ptr),
        M,
        N,
        K
    );

    return c;
}

PYBIND11_MODULE(_tensorforge_cpp, m)
{
    m.def("matmul", &matmul_py);
}