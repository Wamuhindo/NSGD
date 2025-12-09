cmake_minimum_required(VERSION 4.0)
project(untitled2)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled2 main.cpp
        SimProcess.h
        FunctionInstance.h
        FunctionInstance.cpp
        ServerlessSimulator.h
        ServerlessSimulator.cpp
        AutoScalingAlgorithm.h
        AutoScalingAlgorithm.cpp
        input_test_tiny.json
)

target_include_directories(untitled2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
