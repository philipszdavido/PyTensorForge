cmake_minimum_required(VERSION 3.18)

project(tensorforge)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    _tensorforge_cpp

    bindings.cpp
    kernels/MatMul.cpp
)