
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

class VulkanWindow : public std::enable_shared_from_this<VulkanWindow>, public QWindow {

public:

	VulkanWindow();
	virtual ~VulkanWindow();
	VkPhysicalDevice physicalDevice() { return mVulkanPointers.physicalDevice; }
	VkDevice device() { return mVulkanPointers.device; }
	QMatrix4x4 clipCorrectionMatrix();
	QSize swapChainImageSize() { return QSize(this->width(), this->height()); }
	void setupVulkanInstance(QVulkanInstance& instance);
	VkInstance createInstance();
	std::vector<const char*> getRequiredInstanceExtensions();
	const char* getLinuxDisplayType();

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
	bool isDeviceSuitable(const VkPhysicalDevice& device,
		const VkSurfaceKHR& surface, QueueFamilyIndices& qfi, SwapChainSupportDetails& scsd);
	void setupDebugMessenger();
	void releaseDebugMesseger();

	bool mInitialized = true;
	bool mStart = true;
	PFN_vkCreateInstance vkCreateInstance_ = nullptr;
	QMatrix4x4 mClipCorrect = QMatrix4x4();
	VulkanPointers mVulkanPointers;
	std::unique_ptr<Renderer> mRenderer = nullptr;
	std::shared_ptr<FileReader> mFileReader = nullptr;
	QTimer* mTimer = nullptr;
	VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
};

#endif