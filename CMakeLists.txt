cmake_minimum_required(VERSION 3.1)
project(VisionEngine)

Option(MIRROR_OPENMP "openmp support" ON)
Option(MIRROR_BUILD_OBJECT "build object test" ON)
Option(MIRROR_BUILD_CLASSIFIER "build classifier test" ON)
Option(MIRROR_BUILD_FACE "build face test" ON)
Option(MIRROR_BUILD_UNET "build unet test" OFF)

add_subdirectory(src)