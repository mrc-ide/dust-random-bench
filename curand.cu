// -*-c++-*-
// See the nvidia docs:
// https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example

#include <chrono>
#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "common.hpp"

__global__
void setup_kernel(curandState *state, const long nthreads) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    curand_init(1234, i, 0, &state[i]);
  }
}

__global__
void sample_uniform(curandState *state, float *draws,
                    const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    curandState localState = state[i];
    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = curand_uniform(&localState);
      draw += new_draw;
    }
    draws[i] = draw;
    state[i] = localState;
  }
}

__global__
void sample_normal(curandState *state, float *draws,
                   const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    curandState localState = state[i];
    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = curand_normal(&localState);
      draw += new_draw;
    }
    draws[i] = draw;
    state[i] = localState;
  }
}

__global__
void sample_poisson(curandState *state, float *draws,
                    const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    curandState localState = state[i];
    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = curand_poisson(&localState, 1);
      draw += new_draw;
    }
    draws[i] = draw;
    state[i] = localState;
  }
}

void run(const char * distribution_name, size_t nthreads, size_t ndraws) {
  auto distribution_type = check_distribution(distribution_name);

  curandState *devStates;
  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, nthreads * sizeof(float)));

  const size_t blockSize = 128;
  const size_t blockCount = (nthreads + blockSize - 1) / blockSize;

  auto t0_setup = std::chrono::high_resolution_clock::now();
  CUDA_CALL(cudaMalloc((void **)&devStates, nthreads * sizeof(curandState)));

  setup_kernel<<<blockCount, blockSize>>>(devStates, nthreads);
  CUDA_CALL(cudaDeviceSynchronize());
  auto t1_setup = std::chrono::high_resolution_clock::now();

  auto t0_sample = std::chrono::high_resolution_clock::now();
  switch (distribution_type) {
  case UNIFORM:
    sample_uniform<<<blockCount, blockSize>>>(devStates, draws, nthreads,
                                              ndraws);
    break;
  case NORMAL:
    sample_normal<<<blockCount, blockSize>>>(devStates, draws, nthreads,
                                             ndraws);
    break;
  case POISSON:
    sample_poisson<<<blockCount, blockSize>>>(devStates, draws, nthreads,
                                              ndraws);
    break;
  default:
    std::stringstream msg;
    msg << "Distribution not supported with curand: " << distribution_name;
    throw std::runtime_error(msg.str());
  }
  CUDA_CALL(cudaDeviceSynchronize());
  auto t1_sample = std::chrono::high_resolution_clock::now();

  CUDA_CALL(cudaFree(draws));
  CUDA_CALL(cudaFree(devStates));

  std::chrono::duration<double> t_setup = t1_setup - t0_setup;
  std::chrono::duration<double> t_sample = t1_sample - t0_sample;

  std::cout <<
    "distribution: " << distribution_name <<
    ", nthreads: " << nthreads <<
    ", ndraws: " << ndraws <<
    ", t_setup: " << t_setup.count() <<
    ", t_sample: " << t_sample.count() <<
    std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: curand <type> <nthreads> <ndraws>" << std::endl;
    return 1;
  }

  try {
    auto type_str = argv[1];
    const long nthreads = std::stoi(argv[2]);
    const int ndraws = std::stoi(argv[3]);
    run(type_str, nthreads, ndraws);
  } catch (const std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
