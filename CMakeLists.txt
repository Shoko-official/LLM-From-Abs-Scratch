cmake_minimum_required(VERSION 3.10)
project(LLM_From_Scratch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# LAFF (Linear Algebra Foundational Library) 
# My own linear algebra math library as a submodule
# -----------------------------------------------------------------------------

# Collecting LAFF routines from its source folders
file(GLOB_RECURSE LAFF_SOURCES 
    "external/laff/src/*.cpp"
)

# Static library for the math engine
add_library(laff STATIC ${LAFF_SOURCES})

# Adding all include paths from LAFF to support my nested includes
target_include_directories(laff PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include/laff>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include/laff/utility>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include/laff/blas>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include/laff/cholesky>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include/laff/linear_systems>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/laff/include/laff/vector_spaces>
)

# -----------------------------------------------------------------------------
# LLM Project Source (My NN Layers)
# -----------------------------------------------------------------------------

# All my layers go here
file(GLOB_RECURSE PROJECT_SOURCES 
    "src/*.cpp"
)

# Static library for the layers
add_library(llm_layers STATIC ${PROJECT_SOURCES})
target_include_directories(llm_layers PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(llm_layers PRIVATE laff)

enable_testing()

# Activation gradient check
add_executable(activation_grad_check tests/activation_grad_check.cpp)
target_include_directories(activation_grad_check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(activation_grad_check PRIVATE llm_layers laff)

add_test(NAME activation_grad_check COMMAND activation_grad_check)
