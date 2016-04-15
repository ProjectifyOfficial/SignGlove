cmake_minimum_required(VERSION 2.8)
project( neural )
SET (CMAKE_CXX_FLAGS "-fPIC -Wall -std=c++11 -Wno-sign-compare")
add_library(NeuralNetwork SHARED NeuralNetwork.cpp)

