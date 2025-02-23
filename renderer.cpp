
#include "renderer.h"
#include "Vulkan_Simulator.h"

/// <summary>
/// This struct member function requires all PipelineBuilder member structs are 
/// defined before using this.
/// </summary>
/// <param name="vp">Reference to the VulkanPointers struct.</param>
/// <param name="pass">Reference to Vulkan renderpass object.</param>
/// <returns></returns>
VkPipeline PipelineBuilder::buildPipeline(VulkanPointers& vp, VkRenderPass pass) {

    VkPipelineLayoutCreateInfo ppipelineLayout{};
    ppipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ppipelineLayout.pNext = nullptr;
    ppipelineLayout.flags = 0;
    ppipelineLayout.setLayoutCount = 1;
    ppipelineLayout.pSetLayouts = &dsLayout;
    ppipelineLayout.pushConstantRangeCount = 0;
    ppipelineLayout.pPushConstantRanges = nullptr;

    // Create pipelinelayout according to the VkPipelineLayoutCreateInfo struct.
    if (vp.pDeviceFunctions->vkCreatePipelineLayout(vp.device, &ppipelineLayout,
        nullptr, &vp.pipelineLayout) != VK_SUCCESS)
        qFatal("Failed to create pipelinelayout.");

    // We now use all of the structs we have been writing into to create the pipeline.
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = VK_NULL_HANDLE;
    pipelineInfo.flags = 0;
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pTessellationState = VK_NULL_HANDLE;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = vp.pipelineLayout;
    pipelineInfo.renderPass = pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = 0;

    // Finally create graphics pipeline.
    VkPipeline newPipeline;
    if (vp.pDeviceFunctions->vkCreateGraphicsPipelines(vp.device,
        VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
        qFatal("Failed to create pipeline.");
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    }
    else
    {
        return newPipeline;
    }
}

Renderer::Renderer(VulkanPointers& vulkanPointers) {
    mVulkanPointers = vulkanPointers;
}

/// <summary>
/// SetProjectionMatrix function sets the projection matrix.
/// </summary>
/// <param name="proj">4x4 matrix according to the row order.</param>
void Renderer::setProjectionMatrix(float* proj) { 
    mProjectionMatrix = glm::mat4x4(
           proj[0], proj[1], proj[2], proj[3],
           proj[4], proj[5], proj[6], proj[7],
           proj[8], proj[9], proj[10], proj[11],
           proj[12], proj[13], proj[14], proj[15]);
}

/// <summary>
/// SetViewMatrix function sets the view matrix.
/// </summary>
/// <param name="view">Optional 4x4 matrix to set according to the row order.</param>
void Renderer::setViewMatrix(float* view) {
    
    if (view != nullptr) {

        // If we are given a matrix.
        mViewMatrix = glm::mat4x4(
            view[0], view[1], view[2], view[3],
            view[4], view[5], view[6], view[7],
            view[8], view[9], view[10], view[11],
            view[12], view[13], view[14], view[15]);
    }
    else {
        
        // Define suitable view matrix according to the scene.
        if (std::shared_ptr ptr = mVulkanPointers.fileReader.lock()) {
            glm::vec3 midPoint = ptr->getMin() + (ptr->getMax() - ptr->getMin()) / 2.0f;
            float length = glm::length(ptr->getMax() - ptr->getMin());
            glm::vec3 eye = midPoint + glm::vec3(length, length, length);
            mViewMatrix = glm::lookAt(eye, midPoint, glm::vec3(0.0f, 0.0f, 1.0f));
        }
        else {
            qFatal("FileReader object is empty.");
        }
    }
}

/// <summary>
/// SetModelMatrix function sets the model matrix.
/// </summary>
/// <param name="model">Optional 4x4 matrix according to the row order.</param>
void Renderer::setModelMatrix(float* model) {
    
    if (model != nullptr) {

        // If we are given a matrix.
        mModelMatrix = glm::mat4x4(
            model[0], model[1], model[2], model[3],
            model[4], model[5], model[6], model[7],
            model[8], model[9], model[10], model[11],
            model[12], model[13], model[14], model[15]);
    }
    else {
        
        // Get decent time step.
        //static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - mStartTime).count();

        // Rotation center should be in the middle of the object, not necessarily the origo of 3D space.
        glm::vec3 rotationCenter(0.0f, 0.0f, 0.0f);
        if (std::shared_ptr ptr = mVulkanPointers.fileReader.lock()) {
            rotationCenter = ptr->getMin() + (ptr->getMax() - ptr->getMin()) / 2.0f;
        }
        else {
            qFatal("FileReader object is empty.");
        }

        // Translate the object to the origin.
        glm::mat4 T_neg = glm::translate(glm::mat4(1.0f), -rotationCenter);

        // Rotate around the origin.
        glm::mat4 R = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        // Translate the object back to its original position.
        glm::mat4 T_pos = glm::translate(glm::mat4(1.0f), rotationCenter);
        mModelMatrix = T_pos * R * T_neg;
    }
}

/// <summary>
/// FindMemoryType function is used to find suitable memory type from device memory types available before
/// allocating memory for a Vulkan buffer or image.
/// </summary>
/// <param name="typeFilter">Typefilter bitset tells requirements given by created buffer or image.</param>
/// <param name="properties">Properties bitset tells requirements given by a developer.</param>
/// <returns></returns>
uint32_t Renderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

    // Ask memory type requirements given by hardware.
    VkPhysicalDeviceMemoryProperties memProperties;
    mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceMemoryProperties(mVulkanPointers.physicalDevice, &memProperties);

    // Try to find suitable memory type.
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

/// <summary>
/// This function creates vertex buffer for Vulkan. The vertex buffer includes
/// all the vertex data in the default glTF scene.
/// </summary>
/// <param name="model">The content of single gltf file read by tinygltf.</param>
void Renderer::createVertexBuffer(tinygltf::Model& model) {

    std::vector<Vertex> vertices;

    // First we need to extract 3D vertex data from glTF format. glTF data may consist several
    // scenes, but we consider the defaultscene only. Let's enum all nodes in defaultScene:
    for (int i = 0; i < model.scenes[model.defaultScene].nodes.size(); i++) {

        // We investigate every node in defaultScene and their sub-nodes recursively.
        int nodeIndex = model.scenes[model.defaultScene].nodes[i];
        if (std::shared_ptr ptr = mVulkanPointers.fileReader.lock()) {
            ptr->traverse(nodeIndex, vertices, &model);
        }
        else {
            qFatal("FileReader object is empty.");
        }
    }

    mVertexBufferSize = sizeof(vertices[0]) * vertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    // Need to create a temporary CPU buffer to hold the vertex data.
    // Vertex array creation step 1.
    VkBufferCreateInfo bufferInfo1{};
    bufferInfo1.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo1.size = mVertexBufferSize;
    bufferInfo1.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo1.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo1.queueFamilyIndexCount = 0;
    bufferInfo1.pQueueFamilyIndices = nullptr;
    if (mVulkanPointers.pDeviceFunctions->vkCreateBuffer(
        mVulkanPointers.device, &bufferInfo1, nullptr, &stagingBuffer) != VK_SUCCESS) {
        qFatal("Failed to create vertex buffer!");
    }

    VkMemoryRequirements memRequirements1;
    mVulkanPointers.pDeviceFunctions->vkGetBufferMemoryRequirements(mVulkanPointers.device, stagingBuffer, &memRequirements1);
    VkMemoryAllocateInfo allocInfo1{};
    allocInfo1.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo1.allocationSize = memRequirements1.size;
    allocInfo1.memoryTypeIndex = findMemoryType(memRequirements1.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mVulkanPointers.pDeviceFunctions->vkAllocateMemory(mVulkanPointers.device, &allocInfo1, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate vertex buffer memory!");
    }
    mVulkanPointers.pDeviceFunctions->vkBindBufferMemory(mVulkanPointers.device, stagingBuffer, stagingBufferMemory, 0);

    // Then copy the vertex data to the temporary buffer.
    // Vertex array creation step 2.
    void* data;
    mVulkanPointers.pDeviceFunctions->vkMapMemory(mVulkanPointers.device, stagingBufferMemory, 0, mVertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)mVertexBufferSize);
    mVulkanPointers.pDeviceFunctions->vkUnmapMemory(mVulkanPointers.device, stagingBufferMemory);

    // Create a permanent GPU buffer for the vertex data.
    // Vertex array creation step 3.
    VkBufferCreateInfo bufferInfo2{};
    bufferInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo2.size = mVertexBufferSize;
    bufferInfo2.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo2.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo2.queueFamilyIndexCount = 0;
    bufferInfo2.pQueueFamilyIndices = nullptr;
    if (mVulkanPointers.pDeviceFunctions->vkCreateBuffer(
        mVulkanPointers.device, &bufferInfo2, nullptr, &mVertexBuffer) != VK_SUCCESS) {
        qFatal("Failed to create vertex buffer!");
    }

    VkMemoryRequirements memRequirements2;
    mVulkanPointers.pDeviceFunctions->vkGetBufferMemoryRequirements(mVulkanPointers.device, mVertexBuffer, &memRequirements2);
    VkMemoryAllocateInfo allocInfo2{};
    allocInfo2.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo2.allocationSize = memRequirements2.size;
    allocInfo2.memoryTypeIndex = findMemoryType(memRequirements2.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mVulkanPointers.pDeviceFunctions->vkAllocateMemory(mVulkanPointers.device, &allocInfo2, nullptr, &mVertexBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate vertex buffer memory!");
    }
    mVulkanPointers.pDeviceFunctions->vkBindBufferMemory(mVulkanPointers.device, mVertexBuffer, mVertexBufferMemory, 0);

    // Finally copy data from temporary buffer to permanent buffer.
    // Vertex array creation step 4.
    this->copyBuffer(stagingBuffer, mVertexBuffer, mVertexBufferSize);

    // Delete the temporary CPU buffer.
    mVulkanPointers.pDeviceFunctions->vkDestroyBuffer(mVulkanPointers.device, stagingBuffer, nullptr);
    mVulkanPointers.pDeviceFunctions->vkFreeMemory(mVulkanPointers.device, stagingBufferMemory, nullptr);
}

/// <summary>
/// This function destroys the vertex buffer and releases its memory.
/// </summary>
void Renderer::deleteVertexBuffer() {

    // Vertex array creation step 9.
    if (mVertexBuffer) {
        mVulkanPointers.pDeviceFunctions->vkDestroyBuffer(mVulkanPointers.device, mVertexBuffer, nullptr);
        mVulkanPointers.pDeviceFunctions->vkFreeMemory(mVulkanPointers.device, mVertexBufferMemory, nullptr);
        mVertexBuffer = VK_NULL_HANDLE;
        mVertexBufferMemory = VK_NULL_HANDLE;
    }
}

/// <summary>
/// Uniform buffers are formatted in createUniformBuffers function. We have mUboMVPMatrices struct only as a 
/// buffer, but we need MAX_FRAMES_IN_FLIGHT number of buffers for it.
/// </summary>
void Renderer::createUniformBuffers() {

    VkDeviceSize bufferSize = sizeof(mUboMVPMatrices);
    mUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    mUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    mUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

        // Create a uniform buffer for Vulkan...
        // Uniform buffer creation step 2.
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (mVulkanPointers.pDeviceFunctions->vkCreateBuffer(
            mVulkanPointers.device, &bufferInfo, nullptr, &mUniformBuffers[i]) != VK_SUCCESS) {
            std::cout << "Failed to create uniform buffer!\n";
            throw std::runtime_error("Failed to create uniform buffer!");
        }

        // ...tell what kind of memory is suitable for our buffer...
        // Uniform buffer creation step 3.
        VkMemoryRequirements memRequirements{};
        mVulkanPointers.pDeviceFunctions->vkGetBufferMemoryRequirements(
            mVulkanPointers.device, mUniformBuffers[i], &memRequirements);
        uint32_t properties = findMemoryType(memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // ...create that buffer memory...
        // Uniform buffer creation step 4.
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = properties;
        if (mVulkanPointers.pDeviceFunctions->vkAllocateMemory(
            mVulkanPointers.device, &allocInfo, nullptr, &mUniformBuffersMemory[i]) != VK_SUCCESS) {
            qFatal("Failed to allocate uniform buffer memory!");
        }

        // ...bind that buffer...
        // Uniform buffer creation step 5.
        if (mVulkanPointers.pDeviceFunctions->vkBindBufferMemory(mVulkanPointers.device,
            mUniformBuffers[i], mUniformBuffersMemory[i], 0) != VK_SUCCESS) {
            qFatal("Failed to bind uniform buffer memory.");
        }

        // ...and finally map the buffer so we get a memory pointer to that buffer.
        // Uniform buffer creation step 6.
        if (mVulkanPointers.pDeviceFunctions->vkMapMemory(
            mVulkanPointers.device, mUniformBuffersMemory[i], 0,
            (size_t)sizeof(mUboMVPMatrices), 0, &mUniformBuffersMapped[i]) != VK_SUCCESS) {
            qFatal("Failed to map uniform buffer memory.");
        }
    }

    // ...next create descriptor pool...
    // Uniform buffer creation step 7.
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = (uint32_t)10;   // We can make a bigger pool we actually need.

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.flags = 0;
    poolCreateInfo.maxSets = 10;
    poolCreateInfo.pNext = nullptr;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;

    VkResult err = mVulkanPointers.pDeviceFunctions->vkCreateDescriptorPool(
        mVulkanPointers.device,
        &poolCreateInfo,
        nullptr,
        &mDescriptorPool);
    if (err != VK_SUCCESS)
        qFatal("vkCreateDescriptorPool failed (%d)", (uint32_t)err);

    // ...then allocate the descriptor sets. We need MAX_FRAMES_IN_FLIGHT number of them...
    // Uniform buffer creation step 8.
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, mPipelineBuilder.dsLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = mDescriptorPool;
    allocInfo.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    mDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (mVulkanPointers.pDeviceFunctions->vkAllocateDescriptorSets(mVulkanPointers.device, &allocInfo, mDescriptorSets.data()) != VK_SUCCESS) {
        qFatal("Failed to allocate descriptor sets.");
    }

    // ...and finally assing a uniform buffer for each descriptor set.
    // Uniform buffer creation step 9.
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bInfo{};
        bInfo.buffer = mUniformBuffers[i];
        bInfo.offset = 0;
        bInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.pNext = nullptr;
        descriptorWrite.dstSet = mDescriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.pBufferInfo = &bInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;

        mVulkanPointers.pDeviceFunctions->vkUpdateDescriptorSets(
            mVulkanPointers.device, 1, &descriptorWrite, 0, nullptr);
    }
}

/// <summary>
/// DeleteUniformBuffers function destroys the uniform buffer and releases its memory.
/// </summary>
void Renderer::deleteUniformBuffers() {

    // Uniform buffer creation step 11.
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (mUniformBuffers[i]) {
            mVulkanPointers.pDeviceFunctions->vkUnmapMemory(mVulkanPointers.device, mUniformBuffersMemory[i]);
            mVulkanPointers.pDeviceFunctions->vkDestroyBuffer(mVulkanPointers.device, mUniformBuffers[i], nullptr);
            mVulkanPointers.pDeviceFunctions->vkFreeMemory(mVulkanPointers.device, mUniformBuffersMemory[i], nullptr);
            mUniformBuffers[i] = VK_NULL_HANDLE;
            mUniformBuffersMemory[i] = VK_NULL_HANDLE;
        }
    }

    // Descriptor sets are freed automatically when deleting descriptor pool, so no need to delete them separately.
    mVulkanPointers.pDeviceFunctions->vkDestroyDescriptorPool(mVulkanPointers.device, mDescriptorPool, nullptr);
    mDescriptorPool = VK_NULL_HANDLE;
}

/// <summary>
/// UpdateUniformBuffer function updates the given uniform buffer of mUniformBuffers.
/// </summary>
void Renderer::updateUniformBuffer() {


    /*
    // These are temporary testing.
    glm::mat4 modelMatrix = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f),
        mSwapChainRes.swapChainImageSize.width / (float)mSwapChainRes.swapChainImageSize.height, 0.1f, 10.0f);
    projectionMatrix[1][1] *= -1;
    */
    mUboMVPMatrices.modelViewProjectionMatrix = mProjectionMatrix * mViewMatrix * mModelMatrix;

    // Update uniform buffer with the mUboMVPMatrices struct.
    // Uniform buffer creation step 10.
    if (mUniformBuffers[mCurrentFrame] != VK_NULL_HANDLE) {
        std::memcpy(mUniformBuffersMapped[mCurrentFrame],
            &mUboMVPMatrices, sizeof(UniformBufferObject));
    }
}

/// <summary>
/// Render function does the drawing of a single frame.
/// </summary>
void Renderer::render() {

    // First ensure we have swapchain.
    if (mSwapChainRes.swapChain == VK_NULL_HANDLE) return;

    // Set model matrix to rotate the object and after that update uniforms.
    this->setModelMatrix();
    this->updateUniformBuffer();

    // Wait until GPU has done previous rendering.
    mVulkanPointers.pDeviceFunctions->vkWaitForFences(
        mVulkanPointers.device, 1, &mRenderFences[mCurrentFrame], VK_TRUE, UINT64_MAX);
    mVulkanPointers.pDeviceFunctions->vkResetFences(
        mVulkanPointers.device, 1, &mRenderFences[mCurrentFrame]);

    // Then we need to request image index from the swapchain.
    uint32_t imageIndex;
    mVulkanPointers.vkAcquireNextImageKHR(mVulkanPointers.device, mSwapChainRes.swapChain, UINT64_MAX,
        mImageAvailableSemaphores[mCurrentFrame], VK_NULL_HANDLE, &imageIndex);

    // Time to begin the render command buffer.
    mVulkanPointers.pDeviceFunctions->vkResetCommandBuffer(mCommandBuffers[mCurrentFrame], 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    mVulkanPointers.pDeviceFunctions->vkBeginCommandBuffer(
        mCommandBuffers[mCurrentFrame], &beginInfo);

    // Begin render pass.
    QSize swapChainImageSize;
    if (std::shared_ptr ptr = mVulkanPointers.vulkanWindow.lock()) {
        swapChainImageSize = ptr->size() * ptr->devicePixelRatio();
    }
    else {
        qFatal("VulkanWindow object is empty.");
    }
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = mSwapChainRes.renderPass;
    renderPassBeginInfo.framebuffer = mSwapChainRes.swapChainFrameBuffers[imageIndex];
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = swapChainImageSize.width();
    renderPassBeginInfo.renderArea.extent.height = swapChainImageSize.height();
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;
    mVulkanPointers.pDeviceFunctions->vkCmdBeginRenderPass(
        mCommandBuffers[mCurrentFrame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind the graphics pipeline to the command buffer.
    mVulkanPointers.pDeviceFunctions->vkCmdBindPipeline(
        mCommandBuffers[mCurrentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline);

    // Take care of dynamic state.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)mSwapChainImageSize.width();
    viewport.height = (float)mSwapChainImageSize.height();
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    mVulkanPointers.pDeviceFunctions->vkCmdSetViewport(
        mCommandBuffers[mCurrentFrame], 0, 1, &viewport);
    VkRect2D scissor = { 0, 0, mSwapChainImageSize.width(), mSwapChainImageSize.height() };
    mVulkanPointers.pDeviceFunctions->vkCmdSetScissor(
        mCommandBuffers[mCurrentFrame], 0, 1, &scissor);

    // Bind shader parameters.
    VkBuffer vertexBuffers[] = { mVertexBuffer };
    VkDeviceSize vbOffsets[] = { 0 };
    mVulkanPointers.pDeviceFunctions->vkCmdBindVertexBuffers(
        mCommandBuffers[mCurrentFrame], 0, 1, vertexBuffers, vbOffsets);
    mVulkanPointers.pDeviceFunctions->vkCmdBindDescriptorSets(mCommandBuffers[mCurrentFrame],
        VK_PIPELINE_BIND_POINT_GRAPHICS, mVulkanPointers.pipelineLayout, 0, 1,
        &mDescriptorSets[mCurrentFrame], 0, nullptr);

    // Render triangles.
    mVulkanPointers.pDeviceFunctions->vkCmdDraw(
        mCommandBuffers[mCurrentFrame], mVertexBufferSize, 1, 0, 0);

    mVulkanPointers.pDeviceFunctions->vkCmdEndRenderPass(mCommandBuffers[mCurrentFrame]);
    mVulkanPointers.pDeviceFunctions->vkEndCommandBuffer(mCommandBuffers[mCurrentFrame]);

    // Send command buffer to the queue in GPU to process it.
    VkPipelineStageFlags waitStageMash = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &mImageAvailableSemaphores[mCurrentFrame];
    submitInfo.pWaitDstStageMask = &waitStageMash;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &mCommandBuffers[mCurrentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &mRenderingCompleteSemaphores[mCurrentFrame];
    mVulkanPointers.pDeviceFunctions->vkQueueSubmit(
        mVulkanPointers.graphicsQueue, 1, &submitInfo, mRenderFences[mCurrentFrame]);

    // Ask GPU to present a new frame.
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &mRenderingCompleteSemaphores[mCurrentFrame];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &mSwapChainRes.swapChain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    mVulkanPointers.vkQueuePresentKHR(mVulkanPointers.graphicsQueue, &presentInfo);

    // Finally change currentFrame.
    mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

/// <summary>
/// CreateGraphicsPipeline function establishes rendering pipeline.
/// </summary>
void Renderer::createGraphicsPipeline() {

    // Create descriptor set layout. All bindings declared in the shaders need a descriptor set.
    // We have one uniform buffer binding in vertex shader and it's name is UniformBufferObject.
    // Uniform buffer creation step 1.
    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
    uniformBufferLayoutBinding.binding = 0;
    uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uniformBufferLayoutBinding.descriptorCount = 1;
    uniformBufferLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo dsLayoutInfo{};
    dsLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsLayoutInfo.flags = 0;
    dsLayoutInfo.pNext = nullptr;
    dsLayoutInfo.bindingCount = 1;
    dsLayoutInfo.pBindings = &uniformBufferLayoutBinding;

    VkResult err = mVulkanPointers.pDeviceFunctions->vkCreateDescriptorSetLayout(
        mVulkanPointers.device, &dsLayoutInfo, nullptr, &mPipelineBuilder.dsLayout);
    if (err != VK_SUCCESS)
        qFatal("vkCreateDescriptorSetLayout failed (%d)", (uint32_t)err);

    // Create shaders.
    VkShaderModule vertShaderModule = this->createShaderModule(VERTEX_SPIR);
    VkShaderModule fragShaderModule = this->createShaderModule(FRAGMENT_SPIR);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    vertShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    fragShaderStageInfo.pSpecializationInfo = nullptr;

    mPipelineBuilder.shaderStages.clear();
    mPipelineBuilder.shaderStages.push_back(vertShaderStageInfo);
    mPipelineBuilder.shaderStages.push_back(fragShaderStageInfo);

    // Create vertex buffer binding.
    // Vertex array creation step 5.
    mPipelineBuilder.vertexBindingDesc.binding = 0;
    mPipelineBuilder.vertexBindingDesc.stride = sizeof(Vertex);
    mPipelineBuilder.vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Vertex array creation step 6. 
    // Position... 
    mPipelineBuilder.vertexAttrDesc[0].binding = 0;
    mPipelineBuilder.vertexAttrDesc[0].location = 0;
    mPipelineBuilder.vertexAttrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    mPipelineBuilder.vertexAttrDesc[0].offset = offsetof(Vertex, pos);

    // ...and color.
    mPipelineBuilder.vertexAttrDesc[1].binding = 0;
    mPipelineBuilder.vertexAttrDesc[1].location = 1;
    mPipelineBuilder.vertexAttrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    mPipelineBuilder.vertexAttrDesc[1].offset = offsetof(Vertex, color);

    // Vertex array creation step 7.
    mPipelineBuilder.vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    mPipelineBuilder.vertexInputInfo.pNext = nullptr;
    mPipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = 1;
    mPipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = &mPipelineBuilder.vertexBindingDesc;
    mPipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = 2;
    mPipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = mPipelineBuilder.vertexAttrDesc;

    // Create input assembly.
    // Vertex array creation step 8.  
    mPipelineBuilder.inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    mPipelineBuilder.inputAssembly.pNext = nullptr;
    mPipelineBuilder.inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    mPipelineBuilder.inputAssembly.primitiveRestartEnable = VK_FALSE;

    // At the moment we use dynamic viewport and scissor and won't support multiple 
    // viewports or scissors.
    mPipelineBuilder.viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    mPipelineBuilder.viewportState.pNext = nullptr;
    mPipelineBuilder.viewportState.viewportCount = 1;
    mPipelineBuilder.viewportState.pViewports = nullptr;
    mPipelineBuilder.viewportState.scissorCount = 1;
    mPipelineBuilder.viewportState.pScissors = nullptr;

    // Create rasterization state.
    mPipelineBuilder.rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    mPipelineBuilder.rasterizer.pNext = nullptr;
    mPipelineBuilder.rasterizer.depthClampEnable = VK_FALSE;

    // Discards all primitives before the rasterization stage if enabled which we don't want.
    mPipelineBuilder.rasterizer.rasterizerDiscardEnable = VK_FALSE;
    mPipelineBuilder.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    mPipelineBuilder.rasterizer.lineWidth = 1.0f;

    // No backface culling.
    //mPipelineBuilder.rasterizer.cullMode = VK_CULL_MODE_NONE;
    //mPipelineBuilder.rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    mPipelineBuilder.rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    mPipelineBuilder.rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // No depth bias.
    mPipelineBuilder.rasterizer.depthBiasEnable = VK_FALSE;
    mPipelineBuilder.rasterizer.depthBiasConstantFactor = 0.0f;
    mPipelineBuilder.rasterizer.depthBiasClamp = 0.0f;
    mPipelineBuilder.rasterizer.depthBiasSlopeFactor = 0.0f;

    // Create multisampling state. Multisampling defaulted to no multisampling (1 sample per pixel).
    mPipelineBuilder.multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    mPipelineBuilder.multisampling.pNext = nullptr;
    mPipelineBuilder.multisampling.sampleShadingEnable = VK_FALSE;
    mPipelineBuilder.multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    mPipelineBuilder.multisampling.minSampleShading = 0.0f;
    mPipelineBuilder.multisampling.pSampleMask = nullptr;
    mPipelineBuilder.multisampling.alphaToCoverageEnable = VK_FALSE;
    mPipelineBuilder.multisampling.alphaToOneEnable = VK_FALSE;

    // Create depth stencil state.
    mPipelineBuilder.depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    mPipelineBuilder.depthStencil.depthTestEnable = VK_TRUE;
    mPipelineBuilder.depthStencil.depthWriteEnable = VK_TRUE;
    mPipelineBuilder.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    mPipelineBuilder.depthStencil.depthBoundsTestEnable = VK_FALSE;
    mPipelineBuilder.depthStencil.stencilTestEnable = VK_FALSE;

    // Create color blend.
    mPipelineBuilder.colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    mPipelineBuilder.colorBlendAttachment.blendEnable = VK_FALSE;

    // Setup dummy color blending. We aren't using transparent objects.
    // The blending is just "no blend", but we do write to the color attachment.
    mPipelineBuilder.colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    mPipelineBuilder.colorBlending.pNext = nullptr;
    mPipelineBuilder.colorBlending.logicOpEnable = VK_FALSE;
    mPipelineBuilder.colorBlending.logicOp = VK_LOGIC_OP_COPY;
    mPipelineBuilder.colorBlending.attachmentCount = 1;
    mPipelineBuilder.colorBlending.pAttachments = &mPipelineBuilder.colorBlendAttachment;
    mPipelineBuilder.colorBlending.blendConstants[0] = 0.0f;
    mPipelineBuilder.colorBlending.blendConstants[1] = 0.0f;
    mPipelineBuilder.colorBlending.blendConstants[2] = 0.0f;
    mPipelineBuilder.colorBlending.blendConstants[3] = 0.0f;

    // Create dynamic state info.
    mPipelineBuilder.dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    mPipelineBuilder.dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    mPipelineBuilder.dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    mPipelineBuilder.dynamic.dynamicStateCount = 2;
    mPipelineBuilder.dynamic.pDynamicStates = mPipelineBuilder.dynamicStates.data();

    // Now necessary structs are defined and we're ready to build the pipeline when we
    // have a renderpass to build with. We don't use Qt's default renderpass but we have 
    // our own renderpass created. Now build a pipeline which uses that renderpass.
    if (mPipeline != VK_NULL_HANDLE) deleteGraphicsPipeline();
    mPipeline = mPipelineBuilder.buildPipeline(mVulkanPointers, mSwapChainRes.renderPass);

    // Dont't need shader modules anymore, so destroy them.
    mVulkanPointers.pDeviceFunctions->vkDestroyShaderModule(mVulkanPointers.device, fragShaderModule, nullptr);
    mVulkanPointers.pDeviceFunctions->vkDestroyShaderModule(mVulkanPointers.device, vertShaderModule, nullptr);
}

/// <summary>
/// This function destroys Pipeline and its PipelineLayout and releases their memory.
/// </summary>
void Renderer::deleteGraphicsPipeline() {
    if (mPipeline == VK_NULL_HANDLE) return;

    mVulkanPointers.pDeviceFunctions->vkDestroyPipeline(mVulkanPointers.device,
        mPipeline, nullptr);
    mPipeline = VK_NULL_HANDLE;
    mVulkanPointers.pDeviceFunctions->vkDestroyPipelineLayout(mVulkanPointers.device,
        mVulkanPointers.pipelineLayout, nullptr);
    mVulkanPointers.pipelineLayout = VK_NULL_HANDLE;
    mVulkanPointers.pDeviceFunctions->vkDestroyDescriptorSetLayout(
        mVulkanPointers.device, mPipelineBuilder.dsLayout, nullptr);
    mPipelineBuilder.dsLayout = VK_NULL_HANDLE;
    mVulkanPointers.pDeviceFunctions->vkDestroyDescriptorPool(
        mVulkanPointers.device,
        mDescriptorPool,
        nullptr
    );
    mDescriptorPool = VK_NULL_HANDLE;
}

/// <summary>
/// This function creates semaphores needed to synchronize asynchronous GPU function calls.
/// A fence is also created to syncronize GPU and CPU.
/// </summary>
void Renderer::createSyncObjects() {

    mImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    mRenderingCompleteSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    mRenderFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = 0;
    semaphoreInfo.flags = 0;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    fenceInfo.pNext = 0;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (mVulkanPointers.pDeviceFunctions->vkCreateSemaphore(
            mVulkanPointers.device, &semaphoreInfo, nullptr, &mImageAvailableSemaphores[i]) != VK_SUCCESS ||
            mVulkanPointers.pDeviceFunctions->vkCreateSemaphore(
                mVulkanPointers.device, &semaphoreInfo, nullptr, &mRenderingCompleteSemaphores[i]) != VK_SUCCESS ||
            mVulkanPointers.pDeviceFunctions->vkCreateFence(
                mVulkanPointers.device, &fenceInfo, nullptr, &mRenderFences[i]) != VK_SUCCESS) {
            qFatal("failed to create semaphores!");
        }
    }
}

/// <summary>
/// This function deletes semaphores and fences.
/// </summary>
void Renderer::deleteSyncObjects() {

    // Wait until GPU has done previous rendering.
    mVulkanPointers.pDeviceFunctions->vkDeviceWaitIdle(mVulkanPointers.device);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (mRenderingCompleteSemaphores[i] != VK_NULL_HANDLE && mImageAvailableSemaphores[i] != VK_NULL_HANDLE && mRenderFences[i] != VK_NULL_HANDLE) {
            mVulkanPointers.pDeviceFunctions->vkDestroySemaphore(mVulkanPointers.device, mRenderingCompleteSemaphores[i], nullptr);
            mRenderingCompleteSemaphores[i] = VK_NULL_HANDLE;
            mVulkanPointers.pDeviceFunctions->vkDestroySemaphore(mVulkanPointers.device, mImageAvailableSemaphores[i], nullptr);
            mImageAvailableSemaphores[i] = VK_NULL_HANDLE;
            mVulkanPointers.pDeviceFunctions->vkDestroyFence(mVulkanPointers.device, mRenderFences[i], nullptr);
            mRenderFences[i] = VK_NULL_HANDLE;
        }
    }
}

/// <summary>
/// This helper function creates a shader module. Shader can be any type, vertex, fragment etc...
/// </summary>
/// <param name="file">Spv shader file name with format, ex. vertex.spv</param>
/// <returns>Vulkan shader module</returns>
VkShaderModule Renderer::createShaderModule(std::string file) {

    std::unique_ptr<uint32_t[]> data32;
    uint32_t length32 = 0;
    try {

        // Let's see do we find the file from this path.
        std::filesystem::path path = std::filesystem::absolute(file);
        int length = 0;
        if (std::filesystem::is_regular_file(path)) {

            // Read the contents of a file as char.
            std::unique_ptr<char[]> data;
            std::fstream reader;
            reader.open(path.c_str(), std::ios_base::binary | std::ifstream::in | std::ios_base::ate);
            if (!reader.is_open()) {
                qFatal("Failed to open file: %s", file.c_str());
                return nullptr;
            }
            length = reader.tellg();
            reader.seekg(0, reader.beg);
            data = std::make_unique<char[]>(length);
            reader.read(data.get(), length);
            reader.close();

            /*
            // Transform binary char data to uint32_t array using little endian.
            // Use this option if your SPIR-V is pure binary file.
            data32 = std::make_unique<uint32_t[]>(length / 4 + 1);
            for (int i=0, j=0; i < length; i=i+4, j++) {
                data32[j] = data[i] << 24 | data[i + 1] << 16 | data[i + 2] << 8 | data[i + 3];
            }
            */

            // Suppose data is a hexadecimal string.
            // Transform hexadecimal char array to uint32_t array.  
            // Use this option if your SPIR-V is an array of hexadecimal numbers.
            // First separate every hexadecimal number to a single token.
            try {
                data32 = std::make_unique<uint32_t[]>(length / 11 + 1);
                const std::string str = data.get();
                const std::regex re(",", std::regex_constants::basic);
                std::sregex_token_iterator it{ str.begin(), str.end(), re, -1 };
                std::vector<std::string> tokenized{ it, {} };

                // Then remove all unwanted chars away from tokens.
                const char removed[] = "{}\\n ";
                int index = 0;
                for (std::string token : tokenized) {
                    if (token.size() == 0) continue;
                    for (int i = 0; i < strlen(removed); ++i)
                    {
                        token.erase(std::remove(token.begin(), token.end(), removed[i]), token.end());
                    }

                    // Finally transform each hexadecimal number to unsigned int.
                    data32[index] = std::stoul(token, nullptr, 16);
                    index++;
                }
                data32[index] = 0;
                length32 = (uint32_t)index * 4;
            }
            catch (...) {
                qInfo("Exception catched, continuing normal processing...");

                // If we get here, probably data is not in a hexadecimal format. We just need to cast it
                // to uint32_t and deep copy it to a new uint32_t array.
                length32 = (uint32_t)length;
                data32 = std::make_unique<uint32_t[]>(length);
                memcpy(data32.get(), data.get(), length);
            }
        }
        else {
            qFatal("Coldn't find a shader file: %s", file.c_str());
            return nullptr;
        }
    }
    catch (...) {
        qFatal("Failed to read a shader file: %s", file.c_str());
        return nullptr;
    }

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.flags = 0;
    shaderInfo.pNext = VK_NULL_HANDLE;
    shaderInfo.codeSize = length32;
    shaderInfo.pCode = data32.get();
    VkResult err = mVulkanPointers.pDeviceFunctions->vkCreateShaderModule(
        mVulkanPointers.device, &shaderInfo, nullptr, &shaderModule);
    if (err != VK_SUCCESS) {
        qFatal("Failed to create shader module: %d", err);
    }
    return shaderModule;
}

/// <summary>
/// CreateSwapChain function creates Vulkan swapchain, imageviews, renderpass and framebuffers. If there is 
/// an old swap chain you want to replace, define oldSwapChain parameter.
/// </summary>
/// <param name="details">SwapChainSupportDetails struct containing information about graphics cards' properties.</param>
/// <param name="colorFormat">Optional VkFormat colorformat.</param>
/// <param name="depthFormat">Optional VkFormat depthformat</param>
/// <param name="oldSwapChain">Optional existing swapchain, which will be deleted.</param>
void Renderer::createSwapChain(const SwapChainSupportDetails& details, int* colorFormat,
    int* depthFormat, VkSwapchainKHR oldSwapChain) {

    // Test do we have a window.
    if (std::shared_ptr ptr = mVulkanPointers.vulkanWindow.lock()) {
        mSwapChainImageSize = ptr->size() * ptr->devicePixelRatio();
    }
    else {
        qFatal("VulkanWindow object is empty.");
    }
    if (mSwapChainImageSize.isEmpty()) return;

    // Test do we have valid SwapChainSupportDetails struct.
    if (!details.capabilities.has_value() || details.formats.size() < 1 || details.presentModes.size() < 1) {
        qFatal("Cannot create swap chain. Missing swap chain support details.");
    }

    // Wait until possible pending work done.
    if (mVulkanPointers.vkDeviceWaitIdle != nullptr) {
        mVulkanPointers.vkDeviceWaitIdle(mVulkanPointers.device);
    }

    // Select suitable surface image and depth image formats and colorspace.
    VkFormat dFormat = VK_FORMAT_UNDEFINED;
    VkFormat cFormat = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colorSpace;
    if (depthFormat == nullptr) {
        dFormat = this->findSupportedDepthFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }
    else {
        dFormat = (VkFormat)*depthFormat;
    }
    if (colorFormat == nullptr) {

        // If the format list includes just one entry of VK_FORMAT_UNDEFINED, the surface has no 
        // preferred format. Otherwise, at least one supported format will be returned.
        if (details.formats.size() == 1 && details.formats[0].format == VK_FORMAT_UNDEFINED) {
            cFormat = VK_FORMAT_B8G8R8_UNORM;
            colorSpace = details.formats[0].colorSpace;
        }
        else {

            // Try find a preferred format.
            for (const auto& availableFormat : details.formats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    cFormat = availableFormat.format;
                    colorSpace = availableFormat.colorSpace;
                    break;
                }
            }

            // If preferred not available, just take first format.
            if (cFormat == VK_FORMAT_UNDEFINED) {
                cFormat = details.formats[0].format;
                colorSpace = details.formats[0].colorSpace;
            }
        }
    }
    else {
        cFormat = (VkFormat)*colorFormat;
        colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }

    // Set number of images in swap chain.
    uint32_t reqBufferCount = details.capabilities.value().minImageCount + 1;
    if (details.capabilities.value().maxImageCount > 0 && reqBufferCount > details.capabilities.value().maxImageCount) {
        reqBufferCount = details.capabilities.value().maxImageCount;
    }

    // Set image size.
    VkExtent2D bufferSize;
    bufferSize.width = (uint32_t)mSwapChainImageSize.width();
    bufferSize.height = (uint32_t)mSwapChainImageSize.height();
    bufferSize.width = std::clamp(bufferSize.width,
        details.capabilities.value().minImageExtent.width, details.capabilities.value().maxImageExtent.width);
    bufferSize.height = std::clamp(bufferSize.height,
        details.capabilities.value().minImageExtent.height, details.capabilities.value().maxImageExtent.height);

    // Set another properties.
    VkSurfaceTransformFlagBitsKHR preTransform =
        (details.capabilities.value().supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        : details.capabilities.value().currentTransform;

    VkCompositeAlphaFlagBitsKHR compositeAlpha =
        (details.capabilities.value().supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
        ? VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
        : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    // Create a new swapchain for the given window surface.
    if (oldSwapChain == VK_NULL_HANDLE) oldSwapChain = mSwapChainRes.swapChain;
    VkSwapchainCreateInfoKHR swapChainInfo{};
    swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainInfo.flags = 0;
    swapChainInfo.pNext = VK_NULL_HANDLE;
    swapChainInfo.surface = mVulkanPointers.surface;
    swapChainInfo.minImageCount = reqBufferCount;
    swapChainInfo.imageFormat = cFormat;
    swapChainInfo.imageColorSpace = colorSpace;
    swapChainInfo.imageExtent = bufferSize;
    swapChainInfo.imageArrayLayers = 1;
    swapChainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapChainInfo.preTransform = preTransform;
    swapChainInfo.compositeAlpha = compositeAlpha;
    swapChainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapChainInfo.clipped = VK_TRUE;
    swapChainInfo.oldSwapchain = oldSwapChain;

    VkSwapchainKHR newSwapChain;
    VkResult err = mVulkanPointers.vkCreateSwapchainKHR(mVulkanPointers.device, &swapChainInfo, nullptr,
        &newSwapChain);
    if (err != VK_SUCCESS) {
        qFatal("QVulkanWindow: Failed to create swap chain: %d", err);
        return;
    }

    // Delete old swapchain and free its memory before assinging a new one.
    if (oldSwapChain)
        deleteSwapChain();
    mSwapChainRes.swapChain = newSwapChain;

    // Need to save those those swapchain image buffers, surfaceformats and image sizes.
    mVulkanPointers.vkGetSwapchainImagesKHR(
        mVulkanPointers.device, mSwapChainRes.swapChain, &mSwapChainRes.swapChainImageCount, nullptr);
    mSwapChainRes.swapChainImages.resize(mSwapChainRes.swapChainImageCount);
    mVulkanPointers.vkGetSwapchainImagesKHR(mVulkanPointers.device, mSwapChainRes.swapChain,
        &mSwapChainRes.swapChainImageCount, mSwapChainRes.swapChainImages.data());
    mSwapChainRes.swapChainImageFormat = cFormat;
    mSwapChainRes.swapChainImageSize = bufferSize;
    mSwapChainRes.swapChainDepthFormat = dFormat;

    // Create VkImageViews for our swapchain image buffers.
    mSwapChainRes.swapChainImageViews.resize(mSwapChainRes.swapChainImageCount);
    for (int i = 0; i < mSwapChainRes.swapChainImageCount; i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = mSwapChainRes.swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = mSwapChainRes.swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (mVulkanPointers.pDeviceFunctions->vkCreateImageView(
            mVulkanPointers.device, &createInfo, nullptr, &mSwapChainRes.swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    // Create render pass attachments for color and depth image views.
    VkAttachmentDescription passAttachments[2]{};
    passAttachments[0].format = mSwapChainRes.swapChainImageFormat;
    passAttachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    passAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    passAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    passAttachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    passAttachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    passAttachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    passAttachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    passAttachments[1].format = mSwapChainRes.swapChainDepthFormat;
    passAttachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    passAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    passAttachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    passAttachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    passAttachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    //passAttachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    passAttachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    passAttachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentReference{};
    colorAttachmentReference.attachment = 0;
    colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentReference{};
    depthAttachmentReference.attachment = 1;
    depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Create one main subpass of our renderpass.
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentReference;
    subpass.pDepthStencilAttachment = &depthAttachmentReference;

    // Set subpass dependency.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Create the main renderpass.
    VkRenderPassCreateInfo renderPassCreateInfo{};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = 2;
    renderPassCreateInfo.pAttachments = passAttachments;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpass;
    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &dependency;
    if (mVulkanPointers.pDeviceFunctions->vkCreateRenderPass(
        mVulkanPointers.device, &renderPassCreateInfo, nullptr, &mSwapChainRes.renderPass) != VK_SUCCESS) {
        qFatal("Failed to create render pass.");
    }

    // Create one depth image which is used by all swap chain image buffers.
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = dFormat;
    imageCreateInfo.extent.width = mSwapChainImageSize.width();
    imageCreateInfo.extent.height = mSwapChainImageSize.height();
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = nullptr;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    err = mVulkanPointers.pDeviceFunctions->vkCreateImage(
        mVulkanPointers.device, &imageCreateInfo, nullptr, &mSwapChainRes.depthImage);
    if (err != VK_SUCCESS)
        qFatal("Failed to create depth image.");

    VkMemoryRequirements memoryRequirements{};
    mVulkanPointers.pDeviceFunctions->vkGetImageMemoryRequirements(
        mVulkanPointers.device, mSwapChainRes.depthImage, &memoryRequirements);

    VkMemoryAllocateInfo imageAllocateInfo{};
    imageAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imageAllocateInfo.allocationSize = memoryRequirements.size;
    imageAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    err = mVulkanPointers.pDeviceFunctions->vkAllocateMemory(
        mVulkanPointers.device, &imageAllocateInfo, nullptr, &mSwapChainRes.depthImageMemory);
    if (err != VK_SUCCESS)
        qFatal("Failed to allocate device memory for depth image.");

    err = mVulkanPointers.pDeviceFunctions->vkBindImageMemory(
        mVulkanPointers.device, mSwapChainRes.depthImage, mSwapChainRes.depthImageMemory, 0);
    if (err != VK_SUCCESS)
        qFatal("Failed to bind depth image memory.");

    // Create the depth image view and wrap depth image.
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = mSwapChainRes.depthImage;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = imageCreateInfo.format;
    imageViewCreateInfo.components = { VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY };
    imageViewCreateInfo.subresourceRange.aspectMask = aspectMask;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;
    err = mVulkanPointers.pDeviceFunctions->vkCreateImageView(
        mVulkanPointers.device, &imageViewCreateInfo, nullptr, &mSwapChainRes.depthImageView);
    if (err != VK_SUCCESS)
        qFatal("Failed to create depth image view.");

    // Create frame buffer for each index in swap chain with two attachments, color and depth.
    VkFramebufferCreateInfo frameBufferCreateInfo{};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.renderPass = mSwapChainRes.renderPass;
    frameBufferCreateInfo.attachmentCount = 2;
    frameBufferCreateInfo.width = mSwapChainImageSize.width();
    frameBufferCreateInfo.height = mSwapChainImageSize.height();
    frameBufferCreateInfo.layers = 1;

    mSwapChainRes.swapChainFrameBuffers.resize(mSwapChainRes.swapChainImageCount);
    for (int i = 0; i < mSwapChainRes.swapChainImageCount; ++i) {
        VkImageView frameBufferAttachments[] = {
            mSwapChainRes.swapChainImageViews[i],
            mSwapChainRes.depthImageView
        };
        frameBufferCreateInfo.pAttachments = frameBufferAttachments;

        if (mVulkanPointers.pDeviceFunctions->vkCreateFramebuffer(mVulkanPointers.device,
            &frameBufferCreateInfo, nullptr, &mSwapChainRes.swapChainFrameBuffers[i]) != VK_SUCCESS) {
            qFatal("Failed to create swap chain framebuffer no %d.", i);
        }
    }

    mCurrentFrame = 0;

}

/// <summary>
/// DeleteSwapChain deletes objects related to swap chain.
/// </summary>
void Renderer::deleteSwapChain() {

    if (mSwapChainRes.swapChain != VK_NULL_HANDLE) {
        mVulkanPointers.vkDestroySwapchainKHR(mVulkanPointers.device, mSwapChainRes.swapChain, nullptr);
        mSwapChainRes.swapChain = VK_NULL_HANDLE;
    }
    else {
        return;
    }
    for (int i = 0; i < mSwapChainRes.swapChainImageCount; ++i) {
        if (mSwapChainRes.swapChainImageViews[i]) {
            mVulkanPointers.pDeviceFunctions->vkDestroyImageView(mVulkanPointers.device,
                mSwapChainRes.swapChainImageViews[i], nullptr);
            mSwapChainRes.swapChainImageViews[i] = VK_NULL_HANDLE;
            mVulkanPointers.pDeviceFunctions->vkDestroyFramebuffer(mVulkanPointers.device,
                mSwapChainRes.swapChainFrameBuffers[i], nullptr);
            mSwapChainRes.swapChainFrameBuffers[i] = VK_NULL_HANDLE;
        }
    }
    if (mSwapChainRes.depthImageView) {
        mVulkanPointers.pDeviceFunctions->vkDestroyImageView(mVulkanPointers.device,
            mSwapChainRes.depthImageView, nullptr);
        mSwapChainRes.depthImageView = VK_NULL_HANDLE;
    }
    if (mSwapChainRes.depthImage) {
        mVulkanPointers.pDeviceFunctions->vkDestroyImage(mVulkanPointers.device,
            mSwapChainRes.depthImage, nullptr);
        mSwapChainRes.depthImage = VK_NULL_HANDLE;
    }
    if (mSwapChainRes.depthImageMemory) {
        mVulkanPointers.pDeviceFunctions->vkFreeMemory(
            mVulkanPointers.device, mSwapChainRes.depthImageMemory, nullptr);
        mSwapChainRes.depthImageMemory = VK_NULL_HANDLE;
    }
    if (mSwapChainRes.renderPass) {
        mVulkanPointers.pDeviceFunctions->vkDestroyRenderPass(mVulkanPointers.device,
            mSwapChainRes.renderPass, nullptr);
        mSwapChainRes.renderPass = VK_NULL_HANDLE;
    }

}

/// <summary>
/// createCommandPool function creates a command pool and command buffers.
/// </summary>
void Renderer::createCommandPool() {

    // Create commandpool.
    QueueFamilyIndices queueFamilyIndices = mVulkanPointers.queueFamilyIndices;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.pNext = nullptr;
    if (mVulkanPointers.pDeviceFunctions->vkCreateCommandPool(mVulkanPointers.device, &poolInfo,
        nullptr, &mCommandPool) != VK_SUCCESS) {
        qFatal("failed to create command pool!");
    }

    // Create command buffers.
    mCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = mCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)mCommandBuffers.size();
    if (mVulkanPointers.pDeviceFunctions->vkAllocateCommandBuffers(
        mVulkanPointers.device, &allocInfo, mCommandBuffers.data()) != VK_SUCCESS) {
        qFatal("failed to allocate command buffers!");
    }

}

void Renderer::deleteCommandPool() {

    // Command buffers are freed automatically when deleting command pool, so no need to delete them separately.
    mVulkanPointers.pDeviceFunctions->vkDestroyCommandPool(mVulkanPointers.device, mCommandPool, nullptr);
    mCommandPool = VK_NULL_HANDLE;
}

VkFormat Renderer::findSupportedDepthFormat(
    const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceFormatProperties(mVulkanPointers.physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    qFatal("Failed to find supported depth format!");
}

/// <summary>
/// CopyBuffer is a common function to copy data from one Vulkan buffer to another.
/// </summary>
/// <param name="srcBuffer">VkBuffer source.</param>
/// <param name="dstBuffer">VkBuffer destination.</param>
/// <param name="size">Copied data size in bytes.</param>
void Renderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = mCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    mVulkanPointers.pDeviceFunctions->vkAllocateCommandBuffers(
        mVulkanPointers.device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    mVulkanPointers.pDeviceFunctions->vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    mVulkanPointers.pDeviceFunctions->vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    mVulkanPointers.pDeviceFunctions->vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    mVulkanPointers.pDeviceFunctions->vkQueueSubmit(mVulkanPointers.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    mVulkanPointers.pDeviceFunctions->vkQueueWaitIdle(mVulkanPointers.graphicsQueue);

    mVulkanPointers.pDeviceFunctions->vkFreeCommandBuffers(
        mVulkanPointers.device, mCommandPool, 1, &commandBuffer);
}
