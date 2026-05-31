cmake_minimum_required(VERSION 3.10)
project(LLM_From_Scratch VERSION 0.1.0 LANGUAGES NONE)

option(LLM_BUILD_TESTS "Build tests." ON)
option(LLM_ENABLE_WARNINGS "Enable warnings." ON)
option(LLM_WARNINGS_AS_ERRORS "Treat warnings as errors." OFF)

# Check that the LAFF submodule is really there.
set(LAFF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/laff")

if(NOT EXISTS "${LAFF_DIR}/include/laff/Laff.hpp"
    OR NOT EXISTS "${LAFF_DIR}/include/laff/Matrix.hpp"
    OR NOT IS_DIRECTORY "${LAFF_DIR}/src"
)
    message(FATAL_ERROR
        "LAFF is missing or incomplete in external/laff.\n"
        "Expected include/laff/Laff.hpp, include/laff/Matrix.hpp and src/*.cpp.\n"
        "Please run: git submodule update --init --recursive"
    )
endif()

file(GLOB_RECURSE LAFF_SOURCES "${LAFF_DIR}/src/*.cpp")

# A checked-out LAFF without sources is still unusable.
if(NOT LAFF_SOURCES)
    message(FATAL_ERROR
        "LAFF is checked out, but no sources were found in external/laff/src\n"
        "Run: git submodule update --init --recursive"
    )
endif()

# Ask CMake for C++ only after the dependency checks above.
include(CheckLanguage)
check_language(CXX)
if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR
        "No C++ compiler found. Install a C++17 compiler, or pass "
        "-DCMAKE_CXX_COMPILER=<path>."
    )
endif()

enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep warning flags on our code, not on vendored code.
function(enable_project_warnings target)
    if(NOT LLM_ENABLE_WARNINGS)
        return()
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(LLM_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(LLM_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# LAFF is built directly from the submodule sources.
add_library(laff STATIC ${LAFF_SOURCES})
target_include_directories(laff PUBLIC
    $<BUILD_INTERFACE:${LAFF_DIR}>
    $<BUILD_INTERFACE:${LAFF_DIR}/include>
    $<BUILD_INTERFACE:${LAFF_DIR}/include/laff>
    $<BUILD_INTERFACE:${LAFF_DIR}/include/laff/utility>
    $<BUILD_INTERFACE:${LAFF_DIR}/include/laff/blas>
    $<BUILD_INTERFACE:${LAFF_DIR}/include/laff/cholesky>
    $<BUILD_INTERFACE:${LAFF_DIR}/include/laff/linear_systems>
    $<BUILD_INTERFACE:${LAFF_DIR}/include/laff/vector_spaces>
)

# Neural-network layers.
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")
add_library(llm_layers STATIC ${PROJECT_SOURCES})
target_include_directories(llm_layers PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(llm_layers PUBLIC laff)
enable_project_warnings(llm_layers)

if(LLM_BUILD_TESTS)
    # Small numerical checks for the activation layers.
    enable_testing()
    add_executable(activation_grad_check tests/activation_grad_check.cpp)
    target_link_libraries(activation_grad_check PRIVATE llm_layers)
    enable_project_warnings(activation_grad_check)

    add_test(NAME activation_grad_check COMMAND activation_grad_check)
endif()
