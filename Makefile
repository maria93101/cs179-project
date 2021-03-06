CC = /usr/bin/g++

LD_FLAGS = -lrt


CXX = g++
CXXFLAGS = -std=c++0x -Wall -pedantic-errors -mcmodel=large -g
SRCS =  179_knn.cpp data.cpp
OBJS = ${SRCS:.cpp=.o}

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib
CUDA_LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -lcudnn -lcurand

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_61,code=compute_61
        
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS       := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
	CCFLAGS   	  := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS   := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32 
else
	NVCCFLAGS := -m64
endif

TARGETS = knn svd

all: $(TARGETS)

knn: merge_knn.cpp gpu_data.cpp knn.o
	$(CC) $^ -o $@ -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

knn.o: knn.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

cpu_knn: 
	${CXX} ${CXXFLAGS} 179_knn.cpp data.cpp -o cpu_knn

svd: svd_GPU.cpp
	$(CC) $^ -o $@ -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) $(CUDA_LIBS) 

svd.o: svd.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(CUDA_LIBS)  -o $@ -c $<

clean:
	rm -f *.o $(TARGETS) cpu_knn

again: clean $(TARGETS) cpu_knn
