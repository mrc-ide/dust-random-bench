#pragma once

// This file will dissappear once we smooth out the remaining edges
// for using dust/random from standalone code.

// This needs to be put into numeric.hpp, I think, then the following
// bits need re-adding to the dust includes (note that ALIGN and
// KERNEL are missing here are they're not used in the library)
#include <cfloat>

#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__

#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#define SYNCWARP __syncwarp();
#else
#define CONSTANT const
#define SYNCWARP
#endif
