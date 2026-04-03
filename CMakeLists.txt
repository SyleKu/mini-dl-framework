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
)

add_executable(xor_example examples/xor.cpp)
target_link_libraries(xor_example minidl)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor minidl)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd minidl)

add_executable(test_gradient_check tests/test_gradient_check.cpp)
target_link_libraries(test_gradient_check minidl)
