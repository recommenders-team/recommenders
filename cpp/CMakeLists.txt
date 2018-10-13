project(sar)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")

find_package(Boost ${Boost_VW_VERSION} REQUIRED COMPONENTS date_time iostreams system thread filesystem)


add_executable (sar main.cxx)

find_package(Arrow)
find_package(Parquet)

include_directories(SYSTEM
    ${ARROW_INCLUDE_DIR}
    ${PARQUET_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS})

    # /mnt/c/work/arrow/cpp/src)
target_link_libraries(sar
    ${PARQUET_SHARED_LIB}
    ${ARROW_SHARED_LIB}
    ${Boost_LIBRARIES})