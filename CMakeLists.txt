cmake_minimum_required(VERSION 3.18)  # Минимальная версия для поддержки CUDA (17)
project(CudaPlasticity LANGUAGES CXX CUDA) 	

# Cтандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB ALLFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh
  )

add_executable(${PROJECT_NAME} ${ALLFILES})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

#OpenMP
find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if(MSVC)
    add_compile_options(/openmp)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
endif()
set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY CUDA_HOST_COMPILER_OPTIONS " /openmp")
#list(APPEND CUDA_NVCC_FLAGS -Xcompiler "/openmp")
#target_compile_options(${PROJECT_NAME} PRIVATE "/openmp")

message (STATUS "OpenMP version: ${OpenMP_CXX_VERSION_MAJOR}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -extended-lambda")
#target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

#for CUDA 13.0
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "75")

#for CUDA 12.9
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "61")

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)