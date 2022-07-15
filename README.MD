# Vulkan Simulator

Multi platform application development, modern C++ and the new graphics API Vulkan (new in the sense that OpenGL is the old API...) create a mixture of challenges which are to be tackled in this project.

## Preface

Chosen development tool Microsoft Visual Studio offers us a possibility to develop an application simultaneously for several different platforms using the same code base. This time chosen platforms are Windows and Linux, and I use CMake and Qt to enable multi platform development. Vulkan graphics API is used for 3D visualizing of glTF file contents.

Currently the code has several bugs and deficiencies, but 3D rendering already works...

## Requirements

To fully run and develop this project, you need Microsoft Visual Studio 2019 or newer installed on a Windows computer. Then you need another computer with a Linux operating system and an ethernet connection between the two computers. Both computers need Vulkan SDK and Qt 5 installed among many other libraries I cannot remember now. And of course the graphics drivers of both computers must be Vulkan 1.0 capable.

CMake is the script language to orchestrate project build and you need to set correct absolute file paths of your both computers to get things working.

## License

This project is licensed under the MIT License.

