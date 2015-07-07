cmake_minimum_required(VERSION 2.8)
project( depth_saliency )

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

add_executable( depth_saliency main.cpp segmentation.cpp segmentation.h saliency.cpp saliency.h )
target_link_libraries(depth_saliency ${Boost_LIBRARIES})
target_link_libraries(depth_saliency ${OpenCV_LIBS} ${PCL_LIBRARIES} ${V4R_LIBRARIES})

