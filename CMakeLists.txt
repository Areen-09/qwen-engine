cmake_minimum_required(VERSION 3.26)

if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Support multi-architecture CUDA targets (T4, A100, RTX 30/40, H100)
if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;90-virtual")
endif()

project(qwen25_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(CUDAToolkit REQUIRED)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/Z7>
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/O2,/EHsc,/permissive-,/utf-8,/Z7,/arch:AVX2>
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
        $<$<COMPILE_LANGUAGE:CXX>:-mf16c>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2,-mf16c>
    )
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

option(ENABLE_NCCL "Build with NCCL backend on Linux" OFF)

if(ENABLE_NCCL AND NOT WIN32)
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS /usr/local/include /usr/include)
    find_library(NCCL_LIBRARY NAMES nccl PATHS /usr/local/lib /usr/lib)
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
        include_directories(${NCCL_INCLUDE_DIR})
        add_compile_definitions(QWEN_ENABLE_NCCL)
    else()
        message(WARNING "NCCL not found, falling back to zero-dependency P2P communicator")
        set(ENABLE_NCCL OFF)
    endif()
endif()

set(CORE_SOURCES
    src/kernels/rmsnorm_rope.cu
    src/kernels/gemv_fp16.cu
    src/kernels/gemv_w4a16.cu
    src/kernels/gemm_prefill.cu
    src/kernels/attention_gqa.cu
    src/kernels/sampling.cu
    src/kernels/deltanet_sm75.cu
    src/engine.cu
    src/tokenizer.cpp
)

if(ENABLE_NCCL)
    list(APPEND CORE_SOURCES src/communicator_nccl.cu)
else()
    list(APPEND CORE_SOURCES src/communicator_p2p.cu)
endif()

add_library(qwen_core STATIC ${CORE_SOURCES})
set_target_properties(qwen_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(qwen_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(qwen_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_link_libraries(qwen_core PUBLIC CUDA::cudart_static)

if(ENABLE_NCCL AND NCCL_LIBRARY)
    target_link_libraries(qwen_core PUBLIC ${NCCL_LIBRARY})
endif()

# Nanobind Python Bindings Target
option(BUILD_PYTHON_BINDINGS "Build nanobind Python bindings" ON)

if(SKBUILD OR BUILD_PYTHON_BINDINGS)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

    if(NOT nanobind_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
            RESULT_VARIABLE nanobind_STATUS
        )
        if(nanobind_STATUS EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
        endif()
    endif()

    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core NB_STATIC src/python_bindings.cpp)
    target_link_libraries(_core PRIVATE qwen_core CUDA::cudart_static)

    install(TARGETS _core LIBRARY DESTINATION qwen_engine RUNTIME DESTINATION qwen_engine)
endif()

# Gate CLI and standalone servers when packaging wheels
if(NOT SKBUILD)
    # CLI Benchmark executable
    add_executable(cli_bench src/cli_bench.cpp)
    target_link_libraries(cli_bench PRIVATE qwen_core CUDA::cudart_static)

    # Local HTTP / SSE Chat Server
    if(WIN32)
        add_executable(qwen_server src/main_server.cpp)
        target_link_libraries(qwen_server PRIVATE qwen_core CUDA::cudart_static ws2_32)

        add_executable(qwen_chat src/main_server.cpp)
        target_link_libraries(qwen_chat PRIVATE qwen_core CUDA::cudart_static ws2_32)
    else()
        add_executable(qwen_server src/main_server.cpp)
        target_link_libraries(qwen_server PRIVATE qwen_core CUDA::cudart_static)

        add_executable(qwen_chat src/main_server.cpp)
        target_link_libraries(qwen_chat PRIVATE qwen_core CUDA::cudart_static)
    endif()

    # Unit & Integration Kernel Tests
    add_executable(test_kernels tests/test_kernels.cu)
    target_link_libraries(test_kernels PRIVATE qwen_core CUDA::cudart_static)
endif()
