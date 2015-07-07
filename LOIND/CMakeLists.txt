cmake_minimum_required(VERSION 2.8)
project( loind )

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
find_package(PCL 1.8 REQUIRED )
if(PCL_FOUND)
	message(STATUS "Found PCL ${PCL_LIBRARY_DIRS}")
	include_directories(${PCL_INCLUDE_DIRS})
	link_directories(${PCL_LIBRARY_DIRS})
	set(PCL_FOUND TRUE)
else()
	set(PCL_FOUND FALSE)
endif()


find_package(Boost COMPONENTS filesystem regex program_options thread)
include_directories(${Boost_INCLUDE_DIRS})


add_library( ${PROJECT_NAME} loind.cpp loind.h )
target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})
