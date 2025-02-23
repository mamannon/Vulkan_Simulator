# CMakeList.txt : CMake project for Vulkan Simulator, include source and define
# project specific logic here.
cmake_minimum_required(VERSION 3.8)

project("Vulkan_Simulator" VERSION 0.1 LANGUAGES CXX)

# Find includes in the build directories.
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Turn on automatic invocation of the MOC, UIC & RCC.
# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTOUIC ON)
# set(CMAKE_AUTORCC ON)

# This is both for 32 bit and 64 bit Windows.
if(WIN32)

  # These paths don't work if they reside in the CMakeSettings.json file.
  set(CMAKE_PREFIX_PATH "C:/Qt/6.8.1/msvc2022_64/lib/cmake;C:/VulkanSDK/1.3.296.0/cmake")

  # Specify to use C++ version 20 in sources of Vulkan_Simulator
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  # This will create WinMain() entry point instead of main().
  set(CMAKE_WIN32_EXECUTABLE ON)

  # Executables fail to build with Qt 6 in the default configuration
  # without -fPIE. We add that here.
  set(CMAKE_CXX_FLAGS "${Qt6Widgets_EXECUTABLE_COMPILE_FLAGS}")

  # We need to add -DQT_WIDGETS_LIB when using QtWidgets in Qt 6.
  add_definitions(${Qt6Widgets_DEFINITIONS})

  # Find the QtWidgets library.
  find_package(Qt6 REQUIRED COMPONENTS Widgets)

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
                ${CMAKE_SOURCE_DIR}/shaders/vert.spv
                ${CMAKE_CURRENT_BINARY_DIR}/vert.spv)
  add_custom_command(
        TARGET Vulkan_Simulator PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/shaders/frag.spv
                ${CMAKE_CURRENT_BINARY_DIR}/frag.spv)

  # Add the Qt6 Widgets and other needed libraries for linking.
  target_link_libraries(Vulkan_Simulator Qt6::Core Qt6::Widgets)
  target_link_libraries(Vulkan_Simulator Vulkan::Vulkan)

  # Add needed header directory paths.
  target_include_directories(Vulkan_Simulator PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})
  target_include_directories(Vulkan_Simulator PUBLIC ${STB_INCLUDE_DIRS})
  target_include_directories(Vulkan_Simulator PUBLIC ${GLM_INCLUDE_DIRS})
  target_include_directories(Vulkan_Simulator PUBLIC "${VULKAN_SDK}/Include")
  
  # Copy necessary DLLs to the binary directory.
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Windows/SysWOW64/vulkan-1.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Cored.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Cored.pdb"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Guid.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Guid.pdb"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Widgetsd.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Widgetsd.pdb"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/plugins/platforms/qwindowsd.dll"
                "$<TARGET_FILE_DIR:Vulkan_Simulator>/platforms/qwindowsd.dll")
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/plugins/platforms/qwindowsd.pdb"
                "$<TARGET_FILE_DIR:Vulkan_Simulator>/platforms/qwindowsd.pdb")
  else()
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Windows/SysWOW64/vulkan-1.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Core.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Gui.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/bin/Qt6Widgets.dll"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "C:/Qt/6.8.1/msvc2022_64/plugins/platforms/qwindows.dll"
                "$<TARGET_FILE_DIR:Vulkan_Simulator>/platforms/qwindows.dll")
  endif()

else()

  # Ensure we are building against Linux.
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")

    # find the name of library containing dlopen and dlclose.
    set(DL ${CMAKE_DL_LIBS})

    # Set C++ version 20 settings
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    # set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std=gnu++17")

    # Executables fail to build with Qt 6 in the default configuration
    # without -fPIE. We add that here.
    set(CMAKE_CXX_FLAGS "${Qt6Widgets_EXECUTABLE_COMPILE_FLAGS}")

    # We need to add -DQT_WIDGETS_LIB when using QtWidgets in Qt 6.
    add_definitions(${Qt6Widgets_DEFINITIONS})

    # Find the QtWidgets library.
    find_package(Qt6 REQUIRED COMPONENTS Widgets)

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
                ${CMAKE_SOURCE_DIR}/shaders/vert.spv
                ${CMAKE_CURRENT_BINARY_DIR}/vert.spv)
    add_custom_command(
        TARGET Vulkan_Simulator PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/shaders/frag.spv
                ${CMAKE_CURRENT_BINARY_DIR}/frag.spv)

    # Add the Qt6 Widgets and other needed libraries for linking.
    target_link_libraries(Vulkan_Simulator Qt6::Core Qt6::Widgets)
    target_link_libraries(Vulkan_Simulator Vulkan::Vulkan)

    # Add needed header directory paths.
    target_include_directories(Vulkan_Simulator PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})
    target_include_directories(Vulkan_Simulator PUBLIC ${STB_INCLUDE_DIRS})
    target_include_directories(Vulkan_Simulator PUBLIC ${GLM_INCLUDE_DIRS})
    target_include_directories(Vulkan_Simulator PUBLIC ${VULKAN_INCLUDE_DIRS})

    # Copy necessary DLLs to the binary directory.
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "/home/mikael/Qt/6.8.2/gcc_64/lib/libQt6Core.so.6.8.2"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "/home/mikael/Qt/6.8.2/gcc_64/lib/libQt6Gui.so.6.8.2"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)
    add_custom_command(TARGET Vulkan_Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "/home/mikael/Qt/6.8.2/gcc_64/lib/libQt6Widgets.so.6.8.2"
                $<TARGET_FILE_DIR:Vulkan_Simulator>)

  endif()

endif()
