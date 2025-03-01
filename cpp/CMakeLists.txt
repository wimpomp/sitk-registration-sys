cmake_minimum_required(VERSION 3.16.3)

project(sitk_adapter)

set(ENV{Elastix_DIR} "../sitk/build/Elastix-build" )
set(ENV{ITK_DIR} "~/code/c/SimpleITK/build/ITK-build" )
set(ENV{SimpleITK_DIR} "~/code/c/SimpleITK/build/SimpleITK-build" )
find_package(SimpleITK)
add_library(sitk_adapter SHARED sitk_adapter.cxx)
target_link_libraries (sitk_adapter ${SimpleITK_LIBRARIES})
install(TARGETS sitk_adapter DESTINATION .)
