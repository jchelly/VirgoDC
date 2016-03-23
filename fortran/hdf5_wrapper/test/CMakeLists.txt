cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_executable(test_read test_read.f90)
target_link_libraries(test_read hdfwrapper)

add_executable(test_write test_write.f90)
target_link_libraries(test_write hdfwrapper)

add_executable(test_extend test_extend.f90)
target_link_libraries(test_extend hdfwrapper)


