#ifndef __definitions_h__
#define __definitions_h__

#include <iostream>
#include <memory>
#include <filesystem>
#include <regex>
#include <QApplication>
#include <QVulkanInstance>
#include <QLoggingCategory>
#include <QVulkanFunctions>
#include <QVulkanWindowRenderer>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>
#include <QThread>
#include <QMutex>
#include <QtWidgets/qfiledialog.h>
#include <QtWidgets>
#include <QProcessEnvironment>

#include <algorithm>


#ifdef __linux__
    #include <glm.hpp>
#elif _WIN32
    #include <glm/glm.hpp>
    #define VK_USE_PLATFORM_WIN32_KHR
#else

#endif

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

#include "tiny_gltf.h"

const std::vector<const char*> ValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
//    "VK_LAYER_LUNARG_standard_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

/// <summary>
/// Vertex shader. Normally one should not compute model- view- and projection matrix
/// multiplications in a shader, because it is inefficient due to repetition in every vertex.
/// </summary>
static const std::string VERTEX_GLSL =
"#version 450\n"
"#extension GL_KHR_vulkan_glsl : enable\n"

"layout(binding = 0) uniform UniformBufferObject {\n"
"mat4 model;\n"
"mat4 view;\n"
"mat4 proj;\n"
"} ubo;\n"

"layout(location = 0) in vec3 inPosition;\n"

"void main() {\n"
"gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);\n"
"}\n\n";

/// <summary>
/// Fragment shader. All objects drawn are shown flat yellow in the screen.
/// WE DON'T USE THIS SHADER, BECAUSE IT'S FORMAT IS GLSL, AND VULKAN ACCEPT SPIR-V ONLY.
/// HOWEVER, THIS IS THE SOURCE CODE USED TO COMPILE SPIR-V.
/// </summary>
static const std::string FRAGMENT_GLSL =
"#version 450\n"
"#extension GL_KHR_vulkan_glsl : enable\n"

"layout(location = 0) out vec4 outColor;\n"

"void main() {\n"
"outColor = vec4(0.9, 0.9, 0.0, 1.0);\n"
"}\n";

/// <summary>
/// This is the name of the file containing vertex shader in spir-v.
/// </summary>
static const std::string VERTEX_SPIR = "vert.spv";

/// <summary>
/// This is the name of the file containing fragment shader in spir-v.
/// </summary>
static const char* FRAGMENT_SPIR = "frag.spv";

class VulkanRenderer;
class VulkanWindow;

struct VulkanPointers {
 //   VulkanRenderer* pVulkanRenderer = nullptr;
    VulkanWindow* pVulkanWindow = nullptr;
    QVulkanDeviceFunctions* pDeviceFunctions = nullptr;
    QVulkanFunctions* pVulkanFunctions = nullptr;
    QVulkanInstance* pInstance = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::string path = "";
};

#endif