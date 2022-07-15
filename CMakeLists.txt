# CMakeList.txt : CMake project for Vulkan Simulator, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)

project("Vulkan_Simulator" VERSION 0.1 LANGUAGES CXX)

# Find includes in the build directories.
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC, UIC & RCC.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# This is both for 32 bit and 64 bit Windows.
if(WIN32)

  # This will create WinMain() entry point instead of main().
  set(CMAKE_WIN32_EXECUTABLE ON)

  # Executables fail to build with Qt 5 in the default configuration
  # without -fPIE. We add that here.
  set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

  # We need to add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
  add_definitions(${Qt5Widgets_DEFINITIONS})

  # Find the QtWidgets library.
  find_package(Qt5 REQUIRED COMPONENTS Widgets)

  # Find the Vulkan sdk library.
  find_package(Vulkan REQUIRED)

  # List source code files.
  set(VULKAN_SIMULATOR_SOURCES
        "definitions.h"
        "Vulkan_Simulator.h"
        "Vulkan_Simulator.cpp"
        "filereader.h"
        "filereader.cpp"
        "tiny_gltf.h"
        "tiny_gltf.cpp"
        "renderer.h"
        "renderer.cpp"
  )

  # Add source to Vulcan Simulator's executable.
  add_executable(Vulkan_Simulator ${VULKAN_SIMULATOR_SOURCES})

  # Copy shader files from source directory to the binary directory.
  add_custom_command(
        TARGET Vulkan_Simulator PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/vert.spv
                ${CMAKE_CURRENT_BINARY_DIR}/vert.spv)
  add_custom_command(
        TARGET Vulkan_Simulator PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/frag.spv
                ${CMAKE_CURRENT_BINARY_DIR}/frag.spv)

  # Specify to use C++ version 17 in sources of Vulkan_Simulator
  target_compile_features(Vulkan_Simulator PUBLIC cxx_std_17)
  set_target_properties(Vulkan_Simulator PROPERTIES CXX_EXTENSIONS OFF)

  # Add the Qt5 Widgets and other needed libraries for linking.
  target_link_libraries(Vulkan_Simulator Qt5::Core Qt5::Widgets)
  target_link_libraries(Vulkan_Simulator Vulkan::Vulkan)

  # Add needed header directory paths.
  target_include_directories(Vulkan_Simulator PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})
  target_include_directories(Vulkan_Simulator PUBLIC ${STB_INCLUDE_DIRS})
  target_include_directories(Vulkan_Simulator PUBLIC ${GLM_INCLUDE_DIRS})
  target_include_directories(Vulkan_Simulator PUBLIC "${VULKAN_DIRS}\\Include")
  link_directories("${VULKAN_DIRS}\\Lib" "${VULKAN_DIRS}\\Bin")

else()

  # Ensure we are building against Linux.
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")

    # find the name of library containing dlopen and dlclose.
    set(DL ${CMAKE_DL_LIBS})

    # Set C++ version 17 settings
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    # set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
    # set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std=gnu++17")

    # Executables fail to build with Qt 5 in the default configuration
    # without -fPIE. We add that here.
    set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

    # We need to add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
    add_definitions(${Qt5Widgets_DEFINITIONS})

    # Find the QtWidgets library.
    find_package(Qt5 REQUIRED COMPONENTS Widgets)

    # Find the Vulkan sdk library.
    find_package(Vulkan REQUIRED)

    # List source code files.
    set(VULKAN_SIMULATOR_SOURCES
        "definitions.h"
        "Vulkan_Simulator.h"
        "Vulkan_Simulator.cpp"
        "filereader.h"
        "filereader.cpp"
        "tiny_gltf.h"
        "tiny_gltf.cpp"
        "renderer.h"
        "renderer.cpp"
    )

    # Tell CMake to create the Vulkan Simulator executable.
    add_executable(Vulkan_Simulator ${VULKAN_SIMULATOR_SOURCES})

    # Copy shader files from source directory to the binary directory.
    add_custom_command(
        TARGET Vulkan_Simulator PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/vert.spv
                ${CMAKE_CURRENT_BINARY_DIR}/vert.spv)
    add_custom_command(
        TARGET Vulkan_Simulator PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/frag.spv
                ${CMAKE_CURRENT_BINARY_DIR}/frag.spv)

    # Specify to use C++ version 17 in sources of Vulkan_Simulator
    # target_compile_features(Vulkan_Simulator PUBLIC cxx_std_17)
    # set_target_properties(Vulkan_Simulator PROPERTIES CXX_EXTENSIONS OFF)

    # Add the Qt5 Widgets and other needed libraries for linking.
    target_link_libraries(Vulkan_Simulator Qt5::Core Qt5::Widgets)
    target_link_libraries(Vulkan_Simulator Vulkan::Vulkan)

    # Add needed header directory paths.
    target_include_directories(Vulkan_Simulator PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})
    target_include_directories(Vulkan_Simulator PUBLIC ${STB_INCLUDE_DIRS})
    target_include_directories(Vulkan_Simulator PUBLIC ${GLM_INCLUDE_DIRS})
    target_include_directories(Vulkan_Simulator PUBLIC ${VULKAN_INCLUDE_DIRS})

  endif()

endif()
