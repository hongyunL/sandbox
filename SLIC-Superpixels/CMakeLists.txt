cmake_minimum_required(VERSION 2.8)
project( slic )

# For building using OpenCV
find_package( OpenCV 2.4 REQUIRED )
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    set(OpenCV_FOUND TRUE)
else()
    set(OpenCV_FOUND FALSE)
endif()

add_library(${PROJECT_NAME} slic.cpp slic.h)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )


add_executable( slic_test test_slic.cpp)
target_link_libraries(slic_test slic)

