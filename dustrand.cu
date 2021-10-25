// -*-c++-*-

#include <chrono>
#include <iostream>
#include <sstream>

// This needs to be put into numeric.hpp, I think
#include <cfloat>

#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__
#define ALIGN(n) __align__(n)

#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#define SYNCWARP __syncwarp();
#else
#define CONSTANT const
#define SYNCWARP
#endif

#include "common.hpp"
#include <dust/random/random.hpp>
#include "helper.hpp"

using rng_state_type = dust::random::xoshiro128plus_state;
using rng_int_type = rng_state_type::int_type;

__global__
void sample_uniform(rng_int_type * rng_state,
                    float *draws, const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    interleaved<rng_int_type> p_rng(rng_state, static_cast<size_t>(i), static_cast<size_t>(nthreads));

    rng_state_type rng_block;
    for (size_t j = 0; j < rng_block.size(); j++) {
      rng_block.state[j] = p_rng[j];
    }

    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = dust::random::random_real<float>(rng_block);
      draw += new_draw;
      // __syncwarp();
    }
    draws[i] = draw;

    // TODO: Tidy this up; could use put_rng_state in a bit I think?
    for (size_t j = 0; j < rng_block.size(); j++) {
      p_rng[j] = rng_block.state[j];
    }
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

  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, nthreads * sizeof(float)));

  const size_t blockSize = 128;
  const size_t blockCount = (nthreads + blockSize - 1) / blockSize;

  auto t0_setup = high_resolution_clock::now();

  // This is currently done in series on the cpu, and will be quite slow.

  // First, initialise all random number generators
  const int seed = 42;
  dust::random::prng<rng_state_type> rng(nthreads, seed);
  constexpr auto rng_len = rng_state_type::size();

  // Then create a vector of integers representing the underlying
  // random number state, interleaved.
  std::vector<rng_int_type> rng_interleaved(nthreads * rng_len);
  for (size_t i = 0; i < nthreads; ++i) {
    auto p = rng.state(i);
    for (size_t j = 0, at = i; j < rng_len; ++j, at += nthreads) {
      rng_interleaved[at] = p[j];
    }
  }

  device_array<rng_int_type> rng_state(nthreads * rng_len);
  rng_state.set_array(rng_interleaved);

  auto t1_setup = high_resolution_clock::now();

  auto t0_sample = high_resolution_clock::now();
  sample_uniform<<<blockCount, blockSize>>>(rng_state.data(), draws,
                                            nthreads, ndraws);
  CUDA_CALL(cudaDeviceSynchronize());
  auto t1_sample = high_resolution_clock::now();

  auto t_setup = duration_cast<duration<double>>(t1_setup - t0_setup);
  auto t_sample = duration_cast<duration<double>>(t1_sample - t0_sample);

  std::cout << "nthreads: " << nthreads <<
    ", ndraws: " << ndraws <<
    ", t_setup: " << t_setup.count() <<
    ", t_sample: " << t_sample.count() <<
    std::endl;
}
