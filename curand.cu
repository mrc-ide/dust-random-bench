// -*-c++-*-
// See the nvidia docs:
// https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example

#include <chrono>
#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

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
      //__syncwarp();
    }
    draws[i] = draw;
    state[i] = localState;
  }
}

int main(int argc, char *argv[]) {
  using namespace std::chrono;
  if (argc != 3) {
    std::cout << "Usage: curand <nthreads> <ndraws>" << std::endl;
    return 1;
  }

  const long nthreads = std::stoi(argv[1]);
  const int ndraws = std::stoi(argv[2]);

  curandState *devStates;
  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, nthreads * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&devStates, nthreads *
              sizeof(curandState)));

  const size_t blockSize = 128;
  const size_t blockCount = (nthreads + blockSize - 1) / blockSize;

  auto t0_setup = high_resolution_clock::now();
  setup_kernel<<<blockCount, blockSize>>>(devStates, nthreads);
  CUDA_CALL(cudaDeviceSynchronize());
  auto t1_setup = high_resolution_clock::now();

  auto t0_sample = high_resolution_clock::now();
  sample_uniform<<<blockCount, blockSize>>>(devStates, draws, nthreads, ndraws);
  CUDA_CALL(cudaDeviceSynchronize());
  auto t1_sample = high_resolution_clock::now();

  CUDA_CALL(cudaFree(draws));
  CUDA_CALL(cudaFree(devStates));

  auto t_setup = duration_cast<duration<double>>(t1_setup - t0_setup);
  auto t_sample = duration_cast<duration<double>>(t1_sample - t0_sample);

  std::cout << "nthreads: " << nthreads <<
    ", ndraws: " << ndraws <<
    ", t_setup: " << t_setup.count() <<
    ", t_sample: " << t_sample.count() <<
    std::endl;
}
