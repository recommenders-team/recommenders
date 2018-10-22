project(sar)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(Boost ${Boost_VW_VERSION} REQUIRED COMPONENTS date_time iostreams system thread filesystem timer)


add_executable (sar main.cxx SAR.cxx)

find_package(Arrow)
# find_package(Parquet)
set(PARQUET_SHARED_LIB /usr/lib/x86_64-linux-gnu/libparquet.so.11.0.0)

include_directories(SYSTEM
    ${ARROW_INCLUDE_DIR}
    ${PARQUET_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS})

target_link_libraries(sar
    ${PARQUET_SHARED_LIB}
    ${ARROW_SHARED_LIB}
    ${Boost_LIBRARIES})
