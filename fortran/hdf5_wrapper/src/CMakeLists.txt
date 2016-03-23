cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_library (hdfwrapper SHARED
	    hdf5_wrapper.F90 read_dataset_c.c write_dataset_c.c read_attribute_c.c write_attribute_c.c byte_order.c 
            read_attribute.inc write_attribute.inc read_attribute_preprocessor.inc write_attribute_preprocessor.inc
	    read_dataset.inc write_dataset.inc read_dataset_preprocessor.inc write_dataset_preprocessor.inc)

target_link_libraries(hdfwrapper ${HDF5_F90_LIBS} ${HDF5_C_LIBS})

install (TARGETS hdfwrapper DESTINATION lib)
install (DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/ DESTINATION include)
