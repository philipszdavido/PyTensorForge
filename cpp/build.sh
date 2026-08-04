set -e

mkdir -p build
cd build

cmake .. \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

make