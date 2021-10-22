CXXFLAGS=-O2 -std=c++11
CUDAFLAGS=-Xptxas -dlcm=ca --cudart static -gencode arch=compute_75,code=sm_75

TARGETS=curand

all: $(TARGETS)

curand: curand.cu
	nvcc $(CXXFLAGS) $(CUDAFLAGS) curand.cu -o $@

clean:
	$(RM) $(TARGETS)
