# Back propagation

A back propagation implementation in C++ with CUDA support,
and no external dependencies (except for the debugging build).

To run the debugging build, you will need to have the following dependencies:

- PkgConfig
- GSL (GNU Scientific Library)

To build the project, you will need the basics CMake, gcc/g++ and a CUDA compiler (nvcc) if you want to use the GPU.

For Ubuntu, you can install the dependencies (for the cpu only builds) with the following commands:

```bash
sudo apt install cmake pkg-config build-essential libgsl-dev git-lfs
```

## How to run

The pattern for the CMake build is as follows:

```bash
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=<Debug or Release> -DGPU=<ON OR OFF> -DTESTING=<ON OR OFF> -DCMAKE_CUDA_COMPILER=<PATH/TO/NVCC> 
cmake --build . --target BackPropagation --config <Debug or Release>
```

Note any omitted flags will default to OFF or Release.

GCC and Clang work out of the box, but the CMake file also has support for the
[Intel compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)
for CPU only builds.
The Intel compiler tends to be much faster than GCC and Clang for this project.
Due to the number of numerical calculations, the Intel compiler can apply great vectorization optimizations.

To use the Intel compiler, you can follow
the [following pattern](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/use-cmake-with-the-compiler.html)

For the GPU builds, you will need to have the CUDA compiler (nvcc) installed.

A popular location of nvcc is: `/usr/local/cuda/bin/nvcc`.

For example, to build the project with the GPU support in Release mode, you can use the following commands:

```bash
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Release -DGPU=ON -DTESTING=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --target BackPropagation --config Release
```

Pre-made test binaries are found on the release page on GitHub.  
