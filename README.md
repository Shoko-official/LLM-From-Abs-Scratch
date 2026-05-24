# LLM From Absolute Scratch

C++17 learning project for implementing neural-network building blocks from scratch.

This repository is currently focused on the foundations needed for transformer-style models: matrix operations, activation layers, forward/backward passes, and numerical gradient checks.

## What is implemented

- CMake-based C++17 build.
- Static library for neural-network layers.
- Integration with a custom/learning linear algebra library under `external/laff`.
- Gradient checks for activation layers:
  - ReLU
  - Sigmoid
  - Tanh
  - GELU

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Project status

This is an early learning/research project, not yet a complete LLM implementation. The current value of the repository is in the low-level numerical work and tests.

## Next steps

- Add unit tests for every layer.
- Add examples showing forward and backward passes.
- Add a small transformer block once tensor and layer primitives are stable.
- Add CI for Linux builds.
- Document the `external/laff` dependency and setup process.
