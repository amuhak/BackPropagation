# BackPropagation

A backpropagation implementation in C++ with CUDA support, and no external dependencies (except for the debugging build).

To run the debugging build, you will need to have the following dependencies:
```text
PkgConfig
GSL (GNU Scientific Library)
```

To build the project, you will need the basics CMake, gcc/g++ and a CUDA compiler (nvcc) if you want to use the GPU.

For Ubuntu, you can install the dependencies (for the cpu only builds) with the following commands:
```bash
sudo apt install cmake pkg-config build-essential libgsl-dev
```

## How to run

Debug mode:

```bash
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=<Debug or Release> -DGPU=<ON OR OFF> -DTESTING=<ON OR OFF> -DCMAKE_CUDA_COMPILER=<PATH/TO/NVCC> 
cmake --build . --target BackPropagation --config <Debug or Release>
```

For example, my nvcc is located at `/usr/local/cuda/bin/nvcc`.

So if I want a build with the GPU without optimizations (debug mode), I would run:

```bash
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Debug -DGPU=ON -DTESTING=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --target BackPropagation --config Debug
```

You can find test builds in releases.
