﻿
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
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	VkPipelineMultisampleStateCreateInfo multisampling{};
	VkPipelineViewportStateCreateInfo viewportState{};
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	VkPipelineDynamicStateCreateInfo dynamic{};

	VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
	VkVertexInputAttributeDescription vertexAttrDesc[2];
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
	void createSwapChain(const SwapChainSupportDetails& details, int* colorFormat = nullptr,
		int* depthFormat = nullptr, VkSwapchainKHR oldSwapChain = VK_NULL_HANDLE);
	void deleteSwapChain();
	void createCommandPool();
	void deleteCommandPool();
	void createSyncObjects();
	void deleteSyncObjects();
	VkSwapchainKHR getSwapChain() { return mSwapChainRes.swapChain; };

private:

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	VkShaderModule createShaderModule(std::string file);
	void updateUniformBuffer();
	VkFormat findSupportedDepthFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	VulkanPointers mVulkanPointers;
	VkDeviceMemory mVertexBufferMemory = nullptr;
	VkBuffer mVertexBuffer = nullptr;
	VkDeviceSize mVertexBufferSize = 0;
	std::vector<VkBuffer> mUniformBuffers;
	std::vector<VkDeviceMemory> mUniformBuffersMemory;
	std::vector<void*> mUniformBuffersMapped;
	//const std::chrono::system_clock::time_point mStartTime = std::chrono::system_clock::now();
	const std::chrono::high_resolution_clock::time_point mStartTime = std::chrono::high_resolution_clock::now();
	PipelineBuilder mPipelineBuilder;
	VkPipeline mPipeline = VK_NULL_HANDLE;
	VkCommandPool mCommandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> mCommandBuffers;
	QSize mSwapChainImageSize = QSize(0, 0);
	VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
	uint32_t mCurrentFrame = 0;
	std::vector<VkDescriptorSet> mDescriptorSets;
	std::vector<VkSemaphore> mImageAvailableSemaphores;
	std::vector<VkSemaphore> mRenderingCompleteSemaphores;
	std::vector<VkFence> mRenderFences;
	glm::mat4 mModelMatrix = glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
	glm::mat4 mViewMatrix = glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
	glm::mat4 mProjectionMatrix = glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

	struct UniformBufferObject {
		glm::mat4 modelViewProjectionMatrix = glm::mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
	} mUboMVPMatrices;

	struct SwapChainRes {
		VkImage depthImage = VK_NULL_HANDLE;
		VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
		uint32_t swapChainImageCount = 0;
		std::vector<VkImage> swapChainImages = {};
		std::vector<VkImageView> swapChainImageViews = {};
		VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
		VkFormat swapChainDepthFormat = VK_FORMAT_UNDEFINED;
		VkExtent2D swapChainImageSize = {};
		VkImageView depthImageView = VK_NULL_HANDLE;
		VkRenderPass renderPass = VK_NULL_HANDLE;
		std::vector<VkFramebuffer> swapChainFrameBuffers = {};
		VkSwapchainKHR swapChain = VK_NULL_HANDLE;
	} mSwapChainRes;

};

#endif

