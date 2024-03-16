// Vulkan Simulator.h : Include file for standard system include files,
// or project specific include files.

#ifndef __Vulkan_Simulator_h__
#define __Vulkan_Simulator_h__

#include <vulkan/vulkan.h>

#include "definitions.h"
#include "renderer.h"
#include "filereader.h"

enum class LinuxDisplayType : uint8_t {
	Wayland,
	X11,
	None
};

class VulkanWindow : public QWindow
{

public:

	VulkanWindow();
	virtual ~VulkanWindow();
	VkPhysicalDevice physicalDevice() { return mPhysDevice; }
	VkDevice device() { return mDevice; }
	QMatrix4x4 clipCorrectionMatrix();
	QSize swapChainImageSize() { return QSize(this->width(), this->height()); }
	int depthStencilFormat() { return mDepthStencilFormat; }
	int colorFormat() { return mColorFormat; }
	void setupVulkanInstance(QVulkanInstance& instance);
	VkQueue graphicsQueue() { return mGraphicsQueue; }
	VkQueue presentQueue() { return mPresentQueue; }
	QVector<const char*> getRequiredExtensions();
	VkInstance createInstance();

private:

	void exposeEvent(QExposeEvent*) override;
	void resizeEvent(QResizeEvent*) override;
	bool event(QEvent*) override;
	void refresh();
	void init();
	void initResources();
	void initDeviceFunctions();
	void initPhysDeviceFunctions();
	void initSwapChainResources();
	void release();
	void releaseResources();
	void releaseSwapChainResources();
	LinuxDisplayType getLinuxDisplayType();
	const char* pickLinuxSurfaceExtension();


	PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
	PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
	PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = nullptr;
	PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR = nullptr;
	PFN_vkQueuePresentKHR vkQueuePresentKHR = nullptr;


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


	bool mInitialized = true;
	bool mStart = true;
	VkSurfaceKHR mSurface = 0;
	VkPhysicalDevice mPhysDevice = 0;
	VkDevice mDevice = 0;
	VkInstance mInstance = 0;
	VkQueue mGraphicsQueue = 0, mPresentQueue = 0;
	VkCommandPool mCommandPool = 0;
	PFN_vkCreateInstance vkCreateInstance_ = nullptr;
	VkFormat mColorFormat = VK_FORMAT_B8G8R8_UNORM;
	VkFormat mDepthStencilFormat = VK_FORMAT_D24_UNORM_S8_UINT;
	VkSwapchainKHR mSwapChain = 0;
	uint32_t mSwapchainBufferCount = 0;
	QMatrix4x4 mClipCorrect = QMatrix4x4();
	VkRenderPass mDefaultRenderPass = 0;
	QVulkanInstance* mQInstance = VK_NULL_HANDLE;
	VulkanPointers mVulkanPointers;
	std::unique_ptr<Renderer> mRenderer = nullptr;
	std::unique_ptr<FileReader> mFileReader = nullptr;
	QTimer* mTimer = nullptr;
};

#endif