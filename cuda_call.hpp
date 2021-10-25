#pragma once

#include <sstream>

static void throw_cuda_error(const char *file, int line, cudaError_t status) {
  std::stringstream msg;
  if (status == cudaErrorUnknown) {
    msg << file << "(" << line << ") An Unknown CUDA Error Occurred :(";
  } else {
    msg << file << "(" << line << ") CUDA Error Occurred:\n" <<
      cudaGetErrorString(status);
  }
#ifdef DUST_ENABLE_CUDA_PROFILER
  cudaProfilerStop();
#endif
  throw std::runtime_error(msg.str());
}

static void handle_cuda_error(const char *file, int line,
                              cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

  if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess) {
    throw_cuda_error(file, line, status);
  }
}

#define CUDA_CALL( err ) (handle_cuda_error(__FILE__, __LINE__ , err))
