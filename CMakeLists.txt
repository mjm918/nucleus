cmake_minimum_required(VERSION 3.26)
project(nucleus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nucleus_core INTERFACE)
target_include_directories(nucleus_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(nucleus
    src/main.cpp
    src/nucleus/common.h
    src/nucleus/meminfo.h
    src/nucleus/sampler.h
    src/nucleus/kv.h
        src/nucleus/mmap.h
        src/nucleus/layout.h
        src/nucleus/page.h
        src/nucleus/threadpool.h
        src/nucleus/view.h
        src/nucleus/experts.h
        src/nucleus/kernel.cpp
        src/nucleus/kernel.h
        src/nucleus/model.cpp
        src/nucleus/model.h
        src/nucleus/tokenizer.cpp
        src/nucleus/tokenizer.h
        src/args.h
        src/reporter.h
)
target_link_libraries(nucleus PRIVATE nucleus_core)

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)

    find_package(GTest 1.14 CONFIG QUIET)
    if(NOT TARGET GTest::gtest_main)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(nucleus_tests
        tests/common_test.cpp
        tests/mmap_test.cpp
        tests/nuc_file_test.cpp
        tests/expert_store_test.cpp
        tests/kv_test.cpp
        tests/sampler_test.cpp
        tests/threadpool_test.cpp
    )
    target_link_libraries(nucleus_tests PRIVATE nucleus_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(nucleus_tests)
endif()
