#pragma once

#include <sstream>
#include <string>

enum distribution { UNIFORM, NORMAL_BOX_MULLER, NORMAL_POLAR, NORMAL_ZIGGURAT,
                    EXPONENTIAL, POISSON, BINOMIAL, MULTINOMIAL };

distribution check_distribution(std::string name) {
  distribution ret;
  if (name == "uniform") {
    ret = UNIFORM;
  } else if (name == "normal_box_muller") {
    ret = NORMAL_BOX_MULLER;
  } else if (name == "normal_polar") {
    ret = NORMAL_POLAR;
  } else if (name == "normal_ziggurat") {
    ret = NORMAL_ZIGGURAT;
  } else if (name == "exponential") {
    ret = EXPONENTIAL;
  } else if (name == "poisson") {
    ret = POISSON;
  } else if (name == "binomial") {
    ret = BINOMIAL;
  } else if (name == "multinomial") {
    ret = MULTINOMIAL;
  } else {
    std::stringstream msg;
    msg << "Invalid distribution: " << name;
    throw std::runtime_error(msg.str());
  }
  return ret;
}

distribution check_distribution(const char * name) {
  return check_distribution(std::string(name));
}

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
