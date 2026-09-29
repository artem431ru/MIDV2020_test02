cmake_minimum_required(VERSION 3.10)

project(MIDV2020_test02)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(MIDV2020_test02 MIDV2020_test02.cpp)

target_include_directories(MIDV2020_test02 PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(MIDV2020_test02 PRIVATE ${OpenCV_LIBS})

configure_file(config.yaml config.yaml COPYONLY)