// -*-c++-*-
#include <chrono>
#include <iostream>

#include "common.hpp"
#include "compatibility.hpp" // will be removed soonish
#include <dust/random/random.hpp>

using rng_state_type = dust::random::xoshiro128plus_state;
using rng_int_type = rng_state_type::int_type;

__device__
rng_state_type get_rng(const rng_int_type * data, size_t index, size_t n) {
  rng_state_type ret;
  for (size_t i = 0, j = i; i < ret.size(); ++i, j += n) {
    ret[i] = data[j];
  }
  return ret;
}

__device__
void set_rng(rng_state_type& rng, rng_int_type * data, size_t n) {
  for (size_t i = 0, j = i; i < rng.size(); ++i, j += n) {
    data[j] = rng[i];
  }
}

__global__
void sample_uniform(rng_int_type * rng_state_data,
                    float *draws, const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    auto rng_block = get_rng(rng_state_data, i, nthreads);

    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = dust::random::random_real<float>(rng_block);
      draw += new_draw;
    }
    draws[i] = draw;

    set_rng(rng_block, rng_state_data, nthreads);
  }
}

__global__
void sample_normal(rng_int_type * rng_state_data,
                   float *draws, const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    auto rng_block = get_rng(rng_state_data, i, nthreads);

    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = dust::random::normal<float>(rng_block, 0, 1);
      draw += new_draw;
    }
    draws[i] = draw;

    set_rng(rng_block, rng_state_data, nthreads);
  }
}

__global__
void sample_exponential(rng_int_type * rng_state_data,
                        float *draws, const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    auto rng_block = get_rng(rng_state_data, i, nthreads);

    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = dust::random::exponential<float>(rng_block, 1);
      draw += new_draw;
    }
    draws[i] = draw;

    set_rng(rng_block, rng_state_data, nthreads);
  }
}

__global__
void sample_poisson(rng_int_type * rng_state_data,
                        float *draws, const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    auto rng_block = get_rng(rng_state_data, i, nthreads);

    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = dust::random::poisson<float>(rng_block, 1);
      draw += new_draw;
    }
    draws[i] = draw;

    set_rng(rng_block, rng_state_data, nthreads);
  }
}

__global__
void sample_binomial(rng_int_type * rng_state_data,
                        float *draws, const long nthreads, const int ndraws) {
  const int dx = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += dx) {
    auto rng_block = get_rng(rng_state_data, i, nthreads);

    float draw = 0;
    for (int j = 0; j < ndraws; ++j) {
      float new_draw = dust::random::binomial<float>(rng_block, 10, 0.3);
      draw += new_draw;
    }
    draws[i] = draw;

    set_rng(rng_block, rng_state_data, nthreads);
  }
}

void run(const char * distribution_name, size_t nthreads, size_t ndraws) {
  auto distribution_type = check_distribution(distribution_name);
  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, nthreads * sizeof(float)));

  const size_t blockSize = 128;
  const size_t blockCount = (nthreads + blockSize - 1) / blockSize;

  auto t0_setup = std::chrono::high_resolution_clock::now();

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

  rng_int_type* rng_state;
  const size_t len = nthreads * rng_len * sizeof(rng_int_type);
  CUDA_CALL(cudaMalloc((void**)&rng_state, len));
  CUDA_CALL(cudaMemcpy(rng_state, rng_interleaved.data(), len,
                       cudaMemcpyDefault));
  auto t1_setup = std::chrono::high_resolution_clock::now();

  auto t0_sample = std::chrono::high_resolution_clock::now();
  switch(distribution_type) {
  case UNIFORM:
    sample_uniform<<<blockCount, blockSize>>>(rng_state, draws,
                                              nthreads, ndraws);
    break;
  case NORMAL:
    sample_normal<<<blockCount, blockSize>>>(rng_state, draws,
                                              nthreads, ndraws);
    break;
  case EXPONENTIAL:
    sample_exponential<<<blockCount, blockSize>>>(rng_state, draws,
                                                  nthreads, ndraws);
    break;
  case POISSON:
    sample_poisson<<<blockCount, blockSize>>>(rng_state, draws,
                                              nthreads, ndraws);
    break;
  case BINOMIAL:
    sample_binomial<<<blockCount, blockSize>>>(rng_state, draws,
                                               nthreads, ndraws);
    break;
  }
  CUDA_CALL(cudaDeviceSynchronize());
  auto t1_sample = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> t_setup = t1_setup - t0_setup;
  std::chrono::duration<double> t_sample = t1_sample - t0_sample;

  std::cout <<
    "distribution: " << distribution_name <<
    ", nthreads: " << nthreads <<
    ", ndraws: " << ndraws <<
    ", t_setup: " << t_setup.count() <<
    ", t_sample: " << t_sample.count() <<
    std::endl;

  CUDA_CALL(cudaFree(draws));
  CUDA_CALL(cudaFree(rng_state));
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: dustrand <type> <nthreads> <ndraws>" << std::endl;
    return 1;
  }

  try {
    auto type_str = argv[1];
    auto nthreads = std::stoi(argv[2]);
    auto ndraws = std::stoi(argv[3]);
    run(type_str, nthreads, ndraws);
  } catch (const std::exception& e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
