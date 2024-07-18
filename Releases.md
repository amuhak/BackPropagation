These are test releases to make sure the code is working as expected.
They will multiply two 1025x1025 matrices and compare the results with the GNU Scientific Library (GSL) both on the CPU
and the GPU (The file name should make it clear which one is being used).

The debug builds have optimizations turned off, and the release builds have optimizations turned on.
The GPU build will not be run on the CI/CD pipeline because I am not rich.

You can download the test releases below: