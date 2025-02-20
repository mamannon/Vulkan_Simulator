# Vulkan Simulator

Multi platform application development, modern C++ and the new graphics API Vulkan (new in the sense that OpenGL is the old API...) create a mixture of challenges which are to be tackled in this project.

## Preface

Chosen development tool Microsoft Visual Studio offers us a possibility to develop an application simultaneously for several different platforms using the same code base. This time chosen platforms are Windows and Linux, and I use CMake and Qt to enable multi platform development. Vulkan graphics API is used for 3D visualizing of glTF file contents.

Currently the code renders correctly, but data reading from 3D file is not implemented yet.

## Requirements

To fully run and develop this project, you need Microsoft Visual Studio 2022 or newer installed on a Windows computer. Then, if you like to develop for Linux, you need another computer with a Linux operating system and an ethernet connection between the two computers. Graphics drivers of both computers must be Vulkan 1.0 capable.


On Windows machine you need Vulkan SDK and Qt 6.8.1 library installed on your computer, among other libraries. The installation should go like:

- Install Qt libaray from https://www.qt.io/download-qt-installer-oss . Select Qt version 6.8.1.

- Install vcpkg library management tool using command line tool like Windows PowerShell. First download vcpkg into your computer: git clone https://github.com/microsoft/vcpkg.git. Second go to the vcpkg directory: cd vcpkg. Third install vcpkg: .\bootstrap-vcpkg.bat.

- Then use vcpkg to install other needed libraries, nlohmann, stb and glm, to your computer: .\vcpkg install nlohmann-json, .\vcpkg install glm, .\vcpkg install stb.

- Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home#windows .

Also on Linux machine you need Vulkan headers and Qt 6.8.1 library installed, among other libraries:

- Install Qt library using Qt installation software https://www.qt.io/download-qt-installer-oss . After downloading yu need to give a permit to install in terminal with a command 'chmod +x qt-online-installer-linux-x64-4.8.1.run'. After that, double-click the qt-online-installer-linux-x64-4.8.1.run file in downloads folder. Select Qt version 6.8.2.

- Install other needed libraries, nlohmann, stb and glm: sudo apt install nlohmann-json-dev, sudo apt install libstdc++6, sudo apt install libglm-dev.

- Vulkan may be already in your Linux machine or not. There are different installation procedures to install Vulkan into Linux, but you can try this: apt install libvulkan1 mesa-vulkan-drivers vulkan-utils You need Vulkan header files in your Linux machine, of which graphics card need to support Vulkan.

- To install CMake on your Linux machine, go to https://cmake.org/download/ .

- Install GCC compiler, GDB debug manager, openssh-server etc: sudo apt-get install openssh-server g++ gdb make ninja-build rsync zip.


CMake is the script language to orchestrate project build and you need to set correct absolute file paths to get things working.

## License

This project is licensed under the MIT License.

