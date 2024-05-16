all: gemm gemm_debug

gemm: cuda_gemm.cu
	nvcc -ccbin g++ -I../Common -m64 -maxrregcount=255 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o gemm cuda_gemm.cu

gemm_debug: cuda_gemm.cu
	nvcc -ccbin g++ -I../Common -m64 -maxrregcount=255 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o gemm_debug cuda_gemm.cu -DCPU_DEBUG

clean:
	rm -f gemm gemm_debug