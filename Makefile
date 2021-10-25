CXXFLAGS=-O2 -std=c++11 --compiler-options="-Wall"
CUDAFLAGS=-Xptxas -dlcm=ca --cudart static -gencode arch=compute_75,code=sm_75

TARGETS=curand dustrand

all: $(TARGETS)

curand: curand.cu dust/random
	nvcc $(CXXFLAGS) $(CUDAFLAGS) $< -o $@

dustrand: dustrand.cu dust/random
	nvcc -I. $(CXXFLAGS) $(CUDAFLAGS) $< -o $@

dust/random:
	./download-dust-random

clean:
	$(RM) $(TARGETS)

.PHONEY: dust/random
