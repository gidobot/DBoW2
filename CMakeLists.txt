cmake_minimum_required(VERSION 3.0)
project(DBoW2)
include(ExternalProject)

option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Demo    "Build demo application" ON)
option(BUILD_SIFT    "Build sift application" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(HDRS
  include/DBoW2/BowVector.h           include/DBoW2/FBrief.h              include/DBoW2/FSIFT.h
  include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   include/DBoW2/FORB.h
  include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h)
set(SRCS 
  src/BowVector.cpp     src/FBrief.cpp        src/FORB.cpp            src/FSIFT.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

# find_package(OpenCV REQUIRED)
find_package(OpenCV 3.2 REQUIRED
  COMPONENTS
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_xfeatures2d
  PATHS /home/gidobot/workspace/lib/opencv/build
  NO_DEFAULT_PATH
  CONFIG
)

FIND_PACKAGE( Boost COMPONENTS filesystem REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc-7;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 

set(cuda_sources
  CudaSift/cudaImage.cu  
  CudaSift/cudaImage.h  
  CudaSift/cudaSiftH.cu 
  CudaSift/cudaSiftH.h  
  CudaSift/matching.cu  
  CudaSift/cudaSiftD.h  
  CudaSift/cudaSift.h  
  CudaSift/cudautils.h
)  

include_directories(${OpenCV_INCLUDE_DIRS} CudaSift)

if(BUILD_DBoW2)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  target_include_directories(${PROJECT_NAME} PUBLIC include/DBoW2/ include/)
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
endif(BUILD_DBoW2)

if(BUILD_Demo)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
  file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)

  add_executable(load_orb demo/load_orb.cpp)
  target_link_libraries(load_orb ${PROJECT_NAME})
  set_target_properties(load_orb PROPERTIES CXX_STANDARD 11)

  add_executable(make_orb_vocab demo/make_orb_vocabulary.cpp)
  target_link_libraries(make_orb_vocab ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
  set_target_properties(make_orb_vocab PROPERTIES CXX_STANDARD 11)
endif(BUILD_Demo)

if(BUILD_SIFT)
  cuda_add_executable(make_sift_vocab demo/make_sift_vocabulary.cpp
    ${cuda_sources}
    OPTIONS -arch=sm_35)
  target_link_libraries(make_sift_vocab ${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_cudadevrt_LIBRARY} ${Boost_LIBRARIES})
  set_target_properties(make_sift_vocab PROPERTIES CXX_STANDARD 11 COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")

  cuda_add_executable(make_sift_vocab_test demo/make_sift_vocabulary_test.cpp
    ${cuda_sources}
    OPTIONS -arch=sm_35)
  target_link_libraries(make_sift_vocab_test ${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_cudadevrt_LIBRARY} ${Boost_LIBRARIES})
  set_target_properties(make_sift_vocab_test PROPERTIES CXX_STANDARD 11 COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")

  add_executable(yaml_to_txt demo/yaml_to_txt.cpp)
  target_link_libraries(yaml_to_txt ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(yaml_to_txt PROPERTIES CXX_STANDARD 11)

  add_executable(yaml_to_txt_test demo/yaml_to_txt_test.cpp)
  target_link_libraries(yaml_to_txt_test ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(yaml_to_txt_test PROPERTIES CXX_STANDARD 11)

  add_executable(load_sift demo/load_sift.cpp)
  target_link_libraries(load_sift ${PROJECT_NAME})
  set_target_properties(load_sift PROPERTIES CXX_STANDARD 11)
endif(BUILD_SIFT)

configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
  install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)

