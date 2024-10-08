add_definitions(-DGPGOMEA)

set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -Wextra -Wno-unused-parameter -Wno-redundant-move"
)

add_subdirectory(Evolution)
add_subdirectory(Fitness)
add_subdirectory(Genotype)
add_subdirectory(GOMEA)
add_subdirectory(Operators)
add_subdirectory(RunHandling)
add_subdirectory(Selection)
add_subdirectory(Semantics)
add_subdirectory(Utils)
add_subdirectory(Variation)

set(GPGOMEA_LINK_LIBS ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES} ${Python_LIBRARIES} ${OpenMP_CXX_FLAGS})
set(ALL_OBJECT_FILES ${ALL_OBJECT_FILES} PARENT_SCOPE)
set(GPGOMEA_LINK_LIBS ${GPGOMEA_LINK_LIBS} PARENT_SCOPE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Create C++ executable
add_executable(main main.cpp ${ALL_OBJECT_FILES})
target_link_libraries(main ${GPGOMEA_LINK_LIBS})
target_include_directories(
  main PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
