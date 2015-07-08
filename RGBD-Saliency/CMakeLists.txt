cmake_minimum_required(VERSION 2.8)
project( rgbd_saliency )

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

# For building using PCL
find_package(PCL 1.8.0 REQUIRED )
if(PCL_FOUND)
    message(STATUS "Found PCL ${PCL_LIBRARY_DIRS}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    set(PCL_FOUND TRUE)
else()
    set(PCL_FOUND FALSE)
endif()


find_package(PkgConfig REQUIRED)


pkg_check_modules(V4R REQUIRED v4r)
set( V4R_LIBRARIES "-lv4rTomGine -lv4rSegmenterLight" )
message("== V4R LIBS REQUIRED ${V4R_LIBRARIES}" )
message("== V4R INCLUDE DIRS ${V4R_INCLUDE_DIRS}")
include_directories( ${V4R_INCLUDE_DIRS} )
message("== V4R LIB DIRS ${V4R_LIBRARY_DIRS}")
link_directories( ${V4R_LIBRARY_DIRS} )


find_package(Boost COMPONENTS filesystem regex program_options thread)
include_directories(${Boost_INCLUDE_DIRS})

set(DEPTH_LIB depth_saliency)
add_library( ${DEPTH_LIB} 
depth_saliency/segmentation.cpp 
depth_saliency/segmentation.h 
depth_saliency/saliency.cpp 
depth_saliency/saliency.h 
)
target_link_libraries(${DEPTH_LIB} ${Boost_LIBRARIES})
target_link_libraries(${DEPTH_LIB} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${V4R_LIBRARIES})

set(RGB_LIB rgb_saliency)
add_library(${RGB_LIB} 
rgb_saliency/slic/SLIC.cpp 
rgb_saliency/slic/SLIC.h 
rgb_saliency/PreGraph.h 
rgb_saliency/PreGraph.cpp
)
target_link_libraries( ${RGB_LIB} ${OpenCV_LIBS} )

add_executable( ${PROJECT_NAME} main.cpp fusion.h fusion.cpp )
target_link_libraries( ${PROJECT_NAME} ${RGB_LIB} ${DEPTH_LIB} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${V4R_LIBRARIES} )




