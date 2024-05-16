cmake_minimum_required(VERSION 3.10)

project(cuda-nn LANGUAGES CUDA CXX)

set(nn_src "src/layers/linear_layer.cu"
		   "src/layers/relu_activation.cu"
		   "src/layers/sigmoid_activation.cu"
		   "src/nn_utils/bce_cost.cu"
		   "src/nn_utils/matrix.cu"
		   "src/nn_utils/shape.cu"
		   "src/coordinates_dataset.cu"
		   "src/neural_network.cu")

add_library(nn ${nn_src})

add_executable(main src/main.cu)
target_link_libraries(main PRIVATE nn)

