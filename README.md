# BackPropagation

## How to run

Debug mode:

```bash
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=<Debug or Release> -DGPU=<ON OR OFF> -DTESTING=<ON OR OFF> -DCMAKE_CUDA_COMPILER=<PATH/TO/NVCC> 
cmake --build . --target BackPropagation --config <Debug or Release>
```

For example, my nvcc is located at `/usr/local/cuda/bin/nvcc`.
So if I want a build with the GPU without optimizations (debug mode), I would run:

```bash
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Debug -DGPU=ON -DTESTING=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --target BackPropagation --config Debug
```

