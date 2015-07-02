cmake_minimum_required(VERSION 2.8)
project( graph_saliency )

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

add_library(${PROJECT_NAME} slic/SLIC.cpp slic/SLIC.h PreGraph.h PreGraph.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )


add_executable( test main.cpp)
target_link_libraries(test ${PROJECT_NAME})

