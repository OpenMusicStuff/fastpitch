cmake_minimum_required(VERSION 3.14)
project(fastpitch VERSION 1.0.0 LANGUAGES C CXX)

# ---- the library: a single header ------------------------------------------------------
add_library(fastpitch INTERFACE)
add_library(fastpitch::fastpitch ALIAS fastpitch)
target_include_directories(fastpitch INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(fastpitch INTERFACE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(fastpitch INTERFACE Threads::Threads)

include(CMakePackageConfigHelpers)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS fastpitch EXPORT fastpitchTargets)
install(EXPORT fastpitchTargets NAMESPACE fastpitch:: DESTINATION lib/cmake/fastpitch)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fastpitchConfig.cmake
  "include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\n"
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/fastpitchTargets.cmake\")\n")
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/fastpitchConfigVersion.cmake
  COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fastpitchConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/fastpitchConfigVersion.cmake DESTINATION lib/cmake/fastpitch)

# ---- tests, tools, benchmarks ----------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  option(FASTPITCH_NATIVE "Compile tests/tools for the host CPU (-march=native, /arch:AVX2)" ON)
  option(FASTPITCH_BUILD_COMPETITORS "Build benchmark harnesses for third-party pYIN libraries" OFF)

  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  set(FP_FLAGS "")
  if(MSVC)
    list(APPEND FP_FLAGS /W4 /permissive- /D_CRT_SECURE_NO_WARNINGS)
    if(FASTPITCH_NATIVE)
      list(APPEND FP_FLAGS /arch:AVX2)
    endif()
  else()
    list(APPEND FP_FLAGS -Wall -Wextra -Wshadow)
    if(FASTPITCH_NATIVE)
      list(APPEND FP_FLAGS -march=native)
    endif()
    if(WIN32)
      list(APPEND FP_FLAGS -D_CRT_SECURE_NO_WARNINGS)
    endif()
  endif()

  function(fp_exe name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE fastpitch)
    target_compile_options(${name} PRIVATE ${FP_FLAGS})
  endfunction()

  fp_exe(fastpitch_cli tools/fastpitch_cli.cpp)
  fp_exe(example_basic examples/basic.cpp)
  fp_exe(test_pyin tests/test_pyin.cpp)
  fp_exe(test_fft tests/test_fft.cpp)
  fp_exe(stage_profile bench/stage_profile.cpp)

  enable_testing()
  add_test(NAME test_pyin COMMAND test_pyin)
  add_test(NAME test_fft COMMAND test_fft)

  if(FASTPITCH_BUILD_COMPETITORS)
    set(TP ${CMAKE_CURRENT_SOURCE_DIR}/bench/third_party)
    # Original C++ pYIN (Mauch & Dixon), pitch-track path; GPL-2.0, benchmark only.
    add_executable(compete_c4dm bench/compete_c4dm.cpp
      ${TP}/pyin/Yin.cpp ${TP}/pyin/YinUtil.cpp ${TP}/pyin/MonoPitchHMM.cpp
      ${TP}/pyin/SparseHMM.cpp ${TP}/vamp-plugin-sdk/src/vamp-sdk/FFT.cpp)
    target_include_directories(compete_c4dm PRIVATE ${TP}/pyin ${TP}/vamp-plugin-sdk
      ${TP}/vamp-plugin-sdk/src ${CMAKE_CURRENT_SOURCE_DIR}/bench/stubs)
    target_compile_options(compete_c4dm PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native -w>)
    # Sleepwalking/libpyin (C, POSIX headers): build with GCC/Clang (MinGW on Windows).
    if(NOT MSVC)
      add_executable(compete_sw bench/compete_sw.c
        ${TP}/sw_libpyin/pyin.c ${TP}/sw_libpyin/yin.c ${TP}/sw_libpyin/math-funcs.c
        ${TP}/libgvps/gvps_full.c ${TP}/libgvps/gvps_obsrv.c ${TP}/libgvps/gvps_sampled.c
        ${TP}/libgvps/gvps_variable.c)
      target_include_directories(compete_sw PRIVATE ${TP} ${TP}/sw_libpyin)
      target_compile_definitions(compete_sw PRIVATE FP_TYPE=float)
      target_compile_options(compete_sw PRIVATE -Ofast -march=native -w -std=gnu99)
      target_link_libraries(compete_sw PRIVATE m)
    endif()
  endif()
endif()
