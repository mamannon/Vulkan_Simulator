#ifndef __renderer_h__
#define __renderer_h__

#include "definitions.h"

/// <summary>
/// PipelineBuilder is a helper struct to organize various Vulkan objects, which are
/// needed to establish rendering pipeline.
/// </summary>
struct PipelineBuilder {

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	VkPipelineMultisampleStateCreateInfo multisampling{};
	VkPipelineLayoutCreateInfo ppipelineLayout{};
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	VkPipelineDynamicStateCreateInfo dynamic{};

	VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
	VkVertexInputAttributeDescription vertexAttrDesc{};
	VkVertexInputBindingDescription vertexBindingDesc{};
	std::vector<VkDynamicState> dynamicStates;

	VkPipeline buildPipeline(VulkanPointers& vp, VkRenderPass pass);
};

/// <summary>
/// Renderer class handles actual rendering. It uses Vulkan, glTF and glm stuff and 
/// as little as possible Qt. Notice that Vulkan function pointers are delivered
/// by Qt.
/// </summary>
class Renderer {

public:

	Renderer(VulkanPointers& vulkanPointers);
	~Renderer() {}

	void createVertexBuffer(tinygltf::Model& model);
	void deleteVertexBuffer();
	void setProjectionMatrix(float* proj);
	void render();
	void setViewMatrix(float* view = nullptr);
	void setModelMatrix(float* model = nullptr);
	void createUniformBuffers();
	void deleteUniformBuffers();
	void createGraphicsPipeline();
	void deleteGraphicsPipeline();
	void createSwapChain(int* colorFormat = nullptr,
		int* depthFormat = nullptr,
		VkSwapchainKHR oldSwapChain = VK_NULL_HANDLE);
	void deleteSwapChain();
	VkSwapchainKHR getSwapChain() { return mSwapChain; };
	void createSyncObjects();
	void deleteSyncObjects();

private:

	void traverse(int& nodeIndex, tinygltf::Model& model,
		std::vector<glm::vec3>& vertices,
		glm::mat4x4 transformation = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
	void getPositions(std::vector<glm::vec3>& vertices,
		tinygltf::Model& model, int& posAcc, int indAcc = -1,
		glm::mat4x4 transformation = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
	void setMinMax(glm::vec4& vec);
	VkShaderModule createShaderModule(std::string file);
	void updateUniformBuffer();
	bool createTransientImage(VkFormat format,
		VkImageUsageFlags usage,
		VkImageAspectFlags aspectMask,
		VkImage* images,
		VkDeviceMemory* mem,
		VkImageView* views,
		int count);
	uint32_t chooseTransientImageMemType(VkImage img, uint32_t startIndex);

	VulkanPointers mVulkanPointers;
	VkDeviceMemory mVertexBufferMemory = nullptr;
	VkBuffer mVertexBuffer = nullptr;
	int mVertexBufferSize = 0;
	glm::vec3 mMin = { -1, -1, -1 };
	glm::vec3 mMax = { 1, 1, 1 };
	std::vector<VkBuffer> mUniformBuffers;
	std::vector<VkDeviceMemory> mUniformBuffersMemory;
	std::vector<void*> mUniformBuffersMapped;
	const std::chrono::system_clock::time_point mStartTime = std::chrono::system_clock::now();
	PipelineBuilder mPipelineBuilder;
	VkPipeline mPipeline = VK_NULL_HANDLE;
	VkSwapchainKHR mSwapChain = VK_NULL_HANDLE;
	QSize mSwapChainImageSize = QSize(0, 0);
	VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
	uint32_t mActualSwapChainBufferCount = 0;
	uint32_t mCurrentFrame = 0;
	std::vector<VkDescriptorSet> mDescriptorSets;
	VkShaderModule mVertShaderModule = VK_NULL_HANDLE;
	VkShaderModule mFragShaderModule = VK_NULL_HANDLE;
	VkSemaphore mPresentCompleteSemaphore = VK_NULL_HANDLE;
	VkSemaphore mRenderingCompleteSemaphore = VK_NULL_HANDLE;
	VkFence mRenderFence = VK_NULL_HANDLE;

	PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
	PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
	PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = nullptr;
	PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR = nullptr;
	PFN_vkQueuePresentKHR vkQueuePresentKHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR = nullptr;

	//	PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT = nullptr;

	struct UniformBufferObject {
		glm::mat4 projectionMatrix = glm::mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0);
		glm::mat4 modelMatrix = glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
		glm::mat4 viewMatrix = glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
	} mUboMVPMatrices;

	struct SwapChainRes {
		VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
		VkImage depthImage = VK_NULL_HANDLE;
		VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
		VkImage images[MAX_FRAMES_IN_FLIGHT] = {};
		VkImageView depthImageView = VK_NULL_HANDLE;
		VkImageView imageViews[MAX_FRAMES_IN_FLIGHT] = {};
		VkRenderPass renderPass = VK_NULL_HANDLE;
		VkFramebuffer frameBuffers[MAX_FRAMES_IN_FLIGHT] = {};
		VkCommandPool commandPool = VK_NULL_HANDLE;
	} mSwapChainRes;

};

#endif

