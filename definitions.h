
#ifndef __definitions_h__
#define __definitions_h__

#include <iostream>
#include <memory>
#include <filesystem>
#include <regex>
#include <set>
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
#include <gtc/matrix_transform.hpp>
#elif _WIN32
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define VK_USE_PLATFORM_WIN32_KHR
#elif __APPLE__
#include "TargetConditionals.h"
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

#include "tiny_gltf.h"

#ifdef NDEBUG
static const bool enableValidationLayers = false;
#else
static const bool enableValidationLayers = true;
#define ENABLEVALIDATIONLAYERS
#endif

static const int MAX_FRAMES_IN_FLIGHT = 2;
static const VkClearValue clearValues[] = { { 0.67f, 0.84f, 0.9f, 1.0f }, { 1.0f, 0.0f } };

/// <summary>
/// ValidationLayers is a vector you can add the validation layers you want to use.
/// </summary>
static const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

/// <summary>
/// This is the name of the file containing vertex shader in spir-v.
/// </summary>
static const std::string VERTEX_SPIR = "vert.spv";

/// <summary>
/// This is the name of the file containing fragment shader in spir-v.
/// </summary>
static const char* FRAGMENT_SPIR = "frag.spv";

/// <summary>
/// DeviceExtensions contains all needed logical device extensions.
/// </summary>
static const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// Vulkan surface types:
//VK_KHR_win32_surface
//VK_KHR_wayland_surface
//VK_KHR_xcb_surface
//VK_KHR_xlib_surface
//VK_KHR_android_surface
//VK_MVK_macos_surface
//VK_MVK_ios_surface

/// <summary>
/// InstanceExtensions does not include all needed extensions, because some need to be defined at run time.
/// </summary>
static const std::vector<const char*> instanceExtensions = {
    VK_KHR_SURFACE_EXTENSION_NAME,

#ifdef _WIN32
    "VK_KHR_win32_surface",
#elif __APPLE__ 
    #if TARGET_OS_IPHONE

    // iOS device
    "VK_MVK_ios_surface",
#elif TARGET_OS_MAC

    // Other kinds of Mac OS
    "VK_MVK_macos_surface",
#endif
#elif __ANDROID__
    "VK_KHR_android_surface",
#elif __linux__
    "__linux__",
#endif

#ifdef ENABLEVALIDATIONLAYERS
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif

};

/// <summary>
/// DebugCallback is custom callback function to route validation layers debug messages to specified output,
/// in this case the output is "qInfo".
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::string message = "Vulkan validation layer : ";
    message.append(pCallbackData->pMessage);
    qInfo(message.c_str());

    return VK_FALSE;
}

/// <summary>
/// DebugCreateInfo is the message struct of debugCallback fuction.
/// </summary>
static const VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = nullptr,
            .flags = 0,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData = nullptr };

class VulkanRenderer;
class VulkanWindow;
class FileReader;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
};

struct SwapChainSupportDetails {
    std::optional<VkSurfaceCapabilitiesKHR> capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct VulkanPointers {
    bool resourcesDeleted = true;
    std::weak_ptr<VulkanWindow> vulkanWindow;
    std::weak_ptr<FileReader> fileReader;
    QVulkanDeviceFunctions* pDeviceFunctions = nullptr;
    QVulkanFunctions* pVulkanFunctions = nullptr;
    QVulkanInstance* pInstance = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::string path = "";
    QueueFamilyIndices queueFamilyIndices = {};
    SwapChainSupportDetails swapChainSupportDetails = {};
    VkQueue graphicsQueue = 0;
    VkQueue presentQueue = 0;

    // Logical device functions, which need setting up.
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = nullptr;
    PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR = nullptr;
    PFN_vkQueuePresentKHR vkQueuePresentKHR = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;

    // Physical device functions, which need setting up.
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR
        vkGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR
        vkGetPhysicalDeviceSurfacePresentModesKHR = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties
        vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR
        vkGetPhysicalDeviceSurfaceSupportKHR = nullptr;
    PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties
        vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;
};

#endif