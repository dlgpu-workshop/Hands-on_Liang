all: reduction_gpu

reduction_gpu: reduction_gpu.cu
	nvcc -I../Common -arch compute_50 -o reduction reduction_gpu.cu

clean:
	rm -f reduction