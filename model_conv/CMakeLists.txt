cmake_minimum_required(VERSION 3.0.0)
project(ModelConv VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(ModelConv 
	conv.cpp)

target_include_directories(ModelConv PUBLIC 
	${CMAKE_SOURCE_DIR}
	${NAPKIN_XC_SRC_DIR}
	${NAPKIN_XC_SRC_DIR}/liblinear/
	${CMAKE_SOURCE_DIR}/pecos/
	${CMAKE_SOURCE_DIR}/pecos/pecos/core/
)

target_compile_definitions(ModelConv PUBLIC
	DATA_DIR="${CMAKE_SOURCE_DIR}/data/"
)

target_link_libraries(ModelConv PUBLIC
	nxc-lib)

add_executable(ModelBenchmark
	benchmark.cpp)

target_include_directories(ModelBenchmark PUBLIC 
	${CMAKE_SOURCE_DIR}
	${NAPKIN_XC_SRC_DIR}
	${NAPKIN_XC_SRC_DIR}/liblinear/
	${CMAKE_SOURCE_DIR}/pecos/
	${CMAKE_SOURCE_DIR}/pecos/pecos/core/
)

target_compile_definitions(ModelBenchmark PUBLIC
	DATA_DIR="${CMAKE_SOURCE_DIR}/data/"
)

target_link_libraries(ModelBenchmark PUBLIC
	nxc-lib)
