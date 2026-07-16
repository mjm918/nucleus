cmake_minimum_required(VERSION 4.2)
project(nucleus)

set(CMAKE_CXX_STANDARD 26)

add_executable(nucleus src/main.cpp
        src/nucleus/common.h
        src/nucleus/meminfo.h
        src/nucleus/sampler.h
)
