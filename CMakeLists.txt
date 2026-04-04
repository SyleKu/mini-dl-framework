cmake_minimum_required(VERSION 3.16)
project(mini_dl_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(minidl
	src/tensor.cpp
	src/autograd.cpp
	src/module.cpp
	src/linear.cpp
	src/activations.cpp
	src/losses.cpp
	src/optimizer.cpp
	src/mnist.cpp
	src/utils.cpp
)

add_executable(xor_example examples/xor.cpp)
target_link_libraries(xor_example minidl)

add_executable(mnist_mlp examples/mnist_mlp.cpp)
target_link_libraries(mnist_mlp minidl)
target_compile_definitions(mnist_mlp PRIVATE MINI_DL_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
set_target_properties(mnist_mlp PROPERTIES 
	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor minidl)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd minidl)

add_executable(test_gradient_check tests/test_gradient_check.cpp)
target_link_libraries(test_gradient_check minidl)

add_executable(test_matmul_autograd tests/test_matmul_autograd.cpp)
target_link_libraries(test_matmul_autograd minidl)

add_executable(test_activations tests/test_activations.cpp)
target_link_libraries(test_activations minidl)

add_executable(test_bce_loss tests/test_bce_loss.cpp)
target_link_libraries(test_bce_loss minidl)

add_executable(test_mnist_loader tests/test_mnist_loader.cpp)
target_link_libraries(test_mnist_loader minidl)
target_compile_definitions(test_mnist_loader PRIVATE MINI_DL_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
set_target_properties(test_mnist_loader PROPERTIES 
	VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_executable(test_utils tests/test_utils.cpp)
target_link_libraries(test_utils minidl)

add_executable(test_softmax_cross_entropy tests/test_softmax_cross_entropy.cpp)
target_link_libraries(test_softmax_cross_entropy minidl)
