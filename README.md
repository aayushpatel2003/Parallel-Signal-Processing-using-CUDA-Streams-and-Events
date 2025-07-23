# CUDA Streams Asynchronous Signal Processor

## 📝 Project Description

This project demonstrates the use of **CUDA streams** for asynchronous signal processing.
It simulates processing of a large dataset (255 float values × 3 runs) using multiple CUDA kernels
launched in two separate streams.

---

## 📂 Project Structure

- `streams.cu`      → Main CUDA logic with kernel definitions and async stream execution
- `streams.h`       → Header for memory management and helper functions
- `Makefile`        → Build configuration using `nvcc`
- `output.txt`      → Logs initial and final values (CSV format) for each of the 3 runs

---

## 🚀 Functionality

For each of the 3 runs:
1. Allocate and initialize 255 float values using a run-dependent seed.
2. Copy input to device memory.
3. Launch the following CUDA kernels asynchronously:
   - `kernelA1`: Adds random value `x1`
   - `kernelB1`: Multiplies each element by 2
   - `kernelA2`: Subtracts random value `x2`
   - `kernelB2`: Divides each element by 2
4. Copy result back to host memory.
5. Print both initial and final data to:
   - Console (stdout)
   - `output.txt` (in CSV format, alternating input/output per run)

---

## 🔧 Build Instructions

```bash
make clean build     # Compiles using nvcc
./streams.exe        # Run the executable
