cmake_minimum_required(VERSION 3.14)
project(UncertaintyQuantification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(/bigobj)

include(FetchContent)

find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(UncertaintyQuantification src/UncertaintyQuantification.cpp)

# Tell compiler where to find headers
target_include_directories(UncertaintyQuantification PRIVATE
    ${CMAKE_SOURCE_DIR}/libs
)

target_link_libraries(UncertaintyQuantification 
  PRIVATE
    nlohmann_json::nlohmann_json
    Ceres::ceres
)


