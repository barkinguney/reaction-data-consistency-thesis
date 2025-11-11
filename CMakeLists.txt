cmake_minimum_required(VERSION 3.14)
project(UncertaintyQuantification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(/bigobj)

include(FetchContent)

find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(Ceres REQUIRED)

add_library(alglib STATIC
    libs/alglib-cpp/src/alglibinternal.cpp
    libs/alglib-cpp/src/alglibmisc.cpp
    libs/alglib-cpp/src/ap.cpp
    libs/alglib-cpp/src/dataanalysis.cpp
    libs/alglib-cpp/src/diffequations.cpp
    libs/alglib-cpp/src/fasttransforms.cpp
    libs/alglib-cpp/src/integration.cpp
    libs/alglib-cpp/src/interpolation.cpp
    libs/alglib-cpp/src/linalg.cpp
    libs/alglib-cpp/src/optimization.cpp
    libs/alglib-cpp/src/solvers.cpp
    libs/alglib-cpp/src/specialfunctions.cpp
    libs/alglib-cpp/src/statistics.cpp
)

# Add your executable
add_executable(UncertaintyQuantification src/UncertaintyQuantification.cpp)

target_include_directories(alglib PUBLIC libs/alglib-cpp/src)

# Tell compiler where to find headers
target_include_directories(UncertaintyQuantification PRIVATE
    ${CMAKE_SOURCE_DIR}/libs
)

target_link_libraries(UncertaintyQuantification 
  PRIVATE
    alglib
    nlohmann_json::nlohmann_json
    Ceres::ceres
)


