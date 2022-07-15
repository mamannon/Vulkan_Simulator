
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

    // At the moment we use dynamic viewport and scissor and won't support multiple 
    // viewports or scissors.
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;
    viewportState.viewportCount = 1;
    viewportState.pViewports = nullptr;
    viewportState.scissorCount = 1;
    viewportState.pScissors = nullptr;

    // Setup dummy color blending. We aren't using transparent objects.
    // The blending is just "no blend", but we do write to the color attachment.
    VkPipelineColorBlendStateCreateInfo colorBlending {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // Create pipelinelayout according to the VkPipelineLayoutCreateInfo struct.
    if (vp.pDeviceFunctions->vkCreatePipelineLayout(vp.device, &pipelineLayout, 
        nullptr, &vp.pipelineLayout) != VK_SUCCESS)
        qFatal("Failed to create pipelinelayout.");

    // We now use all of the structs we have been writing into to create the pipeline.
    VkGraphicsPipelineCreateInfo pipelineInfo {};
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
 //   mUboMVPMatrices.modelMatrix = glm::mat4();
 //   mUboMVPMatrices.projectionMatrix = glm::mat4();
 //   mUboMVPMatrices.viewMatrix = glm::mat4();

    // Some Vulkan function pointers we need require initializing, so we have to define some here.
    // These are already defined in VulkanWindow, but let's get Renderer it's own pointers.
    vkAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(mVulkanPointers.pVulkanFunctions->
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkAcquireNextImageKHR"));
    vkCreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(mVulkanPointers.pVulkanFunctions->
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkCreateSwapchainKHR"));
    vkDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(mVulkanPointers.pVulkanFunctions->
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkDestroySwapchainKHR"));
    vkGetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(mVulkanPointers.pVulkanFunctions->
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkGetSwapchainImagesKHR"));
    vkQueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(mVulkanPointers.pVulkanFunctions->
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkQueuePresentKHR"));
}

/// <summary>
/// This function sets the projection matrix.
/// </summary>
/// <param name="proj">4x4 matrix according to the column order.</param>
void Renderer::setProjectionMatrix(float* proj) {
 
    mUboMVPMatrices.projectionMatrix = glm::mat4x4(
        proj[0], proj[1], proj[2], proj[3],
        proj[4], proj[5], proj[6], proj[7], 
        proj[8], proj[9], proj[10], proj[11], 
        proj[12], proj[13], proj[14], proj[15]);
        
}

/// <summary>
/// This function sets the view matrix.
/// </summary>
/// <param name="view">Optional 4x4 matrix to set.</param>
void Renderer::setViewMatrix(float* view) {
    
    if (view != nullptr) {

        // If we are given a matrix.
        mUboMVPMatrices.viewMatrix = glm::mat4x4(
            view[0], view[1], view[2], view[3],
            view[4], view[5], view[6], view[7],
            view[8], view[9], view[10], view[11],
            view[12], view[13], view[14], view[15]);
    }
    else {

        // Define suitable view matrix according to the scene.
        float z = std::max(std::abs(mMin.x), std::max(std::abs(mMin.y), std::abs(mMin.z)));
        mUboMVPMatrices.viewMatrix = glm::mat4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, z, z*3, 1);
    }
    
}

/// <summary>
/// This function sets the model matrix.
/// </summary>
/// <param name="model">Optional 4x4 matrix to set.</param>
void Renderer::setModelMatrix(float* model) {
    
    if (model != nullptr) {

        // If we are given a matrix.
        mUboMVPMatrices.modelMatrix = glm::mat4x4(
            model[0], model[1], model[2], model[3],
            model[4], model[5], model[6], model[7],
            model[8], model[9], model[10], model[11],
            model[12], model[13], model[14], model[15]);
    }
    else {

        // Create a cumulative rotation around the y-axis with 5.729 degrees ie 0.1 
        // radians per 0.1 seconds. This should be constant rotation between different
        // computers with regard to speed.
        auto currentTime = std::chrono::system_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>
            (currentTime - mStartTime).count();
        float rotation = 0.1 * time;
        glm::mat4x4 temp = glm::mat4x4(
            std::cos(rotation), 0, std::sin(rotation), 0, 0, 1, 0, 0,
            -std::sin(rotation), 0, std::cos(rotation), 0, 0, 0, 0, 1);
        mUboMVPMatrices.modelMatrix = temp * mUboMVPMatrices.modelMatrix;
    }
    
}

/// <summary>
/// This function creates vertex buffer for Vulkan. The vertex buffer includes
/// all the vertex data in the default glTF scene.
/// </summary>
/// <param name="model">The content of single gltf file read by tinygltf.</param>
void Renderer::createVertexBuffer(tinygltf::Model& model) {

    std::vector<glm::vec3> vertices;

    // First we need to extract 3D vertex data from glTF format. glTF data may consist several 
    // scenes, but we consider the defaultscene only. Let's enum all nodes in defaultScene:
    for (int i = 0; i < model.scenes[model.defaultScene].nodes.size(); i++) {

        // We investigate every node in defaultScene and their sub-nodes recursively.
        int nodeIndex = model.scenes[model.defaultScene].nodes[i];
        this->traverse(nodeIndex, model, vertices);
    }

    // Save the number of vertices.
    //mVertexBufferSize = vertices.size();  // !!!USE THIS LINE IF YOU WANT TO USE GLTF FILES!!!
    mVertexBufferSize = 3;  // !!!USE THIS IF YOU USE TEST TRIANGLE!!!

    // Then create a vertex buffer for Vulkan...
    // Vertex array creation step 1.
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(vertices[0]) * mVertexBufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.queueFamilyIndexCount = 0;
    bufferInfo.pQueueFamilyIndices = nullptr;
    if (mVulkanPointers.pDeviceFunctions->vkCreateBuffer(
        mVulkanPointers.device, &bufferInfo, nullptr, &mVertexBuffer) != VK_SUCCESS) {
        qFatal("Failed to create vertex buffer!");
    }

    // ...tell what kind of memory is suitable for our buffer...
    // Vertex array creation step 2.
    VkMemoryRequirements memRequirements{};
    mVulkanPointers.pDeviceFunctions->vkGetBufferMemoryRequirements(
        mVulkanPointers.device, mVertexBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = NULL;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = 0;

    VkPhysicalDeviceMemoryProperties memProperties{};
    mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceMemoryProperties(
        mVulkanPointers.physicalDevice, &memProperties);

    uint32_t vertexMemoryTypeBits = memRequirements.memoryTypeBits;
    VkMemoryPropertyFlags vertexDesiredMemoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    for (uint32_t i = 0; i < 32; ++i) {
        VkMemoryType memoryType = memProperties.memoryTypes[i];
        if (vertexMemoryTypeBits & 1) {
            if ((memoryType.propertyFlags & vertexDesiredMemoryFlags) == vertexDesiredMemoryFlags) {
                allocInfo.memoryTypeIndex = i;
                break;
            }
        }
        vertexMemoryTypeBits = vertexMemoryTypeBits >> 1;
    }

    // ...create that buffer memory...
    // Vertex array creation step 3.
    VkDeviceMemory vertexBufferMemory;
    if (mVulkanPointers.pDeviceFunctions->vkAllocateMemory(
        mVulkanPointers.device, &allocInfo, nullptr, &mVertexBufferMemory) != VK_SUCCESS) {
        qFatal("Failed to allocate vertex buffer memory!");
    }

    // ...fill memory with the extracted vertex data...
    // Vertex array creation step 4.
    void* data;
    if (!mVulkanPointers.pDeviceFunctions->vkMapMemory(
        mVulkanPointers.device, mVertexBufferMemory, 0, bufferInfo.size, 0, &data) == VK_SUCCESS) {
        qFatal("Failed to map vertex buffer memory.");
    }
    
    // !!!TEST TRIANGLE BELOW. USE IT IF YOU DON'T WANT TO VIEW GLTF FILES!!!
    float* triangle = (float*)data;
    triangle[0] = -1.0;
    triangle[1] = -1.0;
    triangle[2] = 0.9;
    triangle[3] = 1.0;
    triangle[4] = -1.0;
    triangle[5] = 0.9;
    triangle[6] = 0.0;
    triangle[7] = 1.0;
    triangle[8] = 0.9;
    // !!!TEST TRIANGLE ABOVE!!!
    
    // !!!USE THIS LINE IF YOU WANT TO USE GLTF FILES!!!
    //memcpy(data, vertices.data(), (size_t)bufferInfo.size);

    mVulkanPointers.pDeviceFunctions->vkUnmapMemory(mVulkanPointers.device, mVertexBufferMemory);

    // ...and finally bind the vertex buffer memory.
    // Vertex array creation step 5.
    if (mVulkanPointers.pDeviceFunctions->vkBindBufferMemory(mVulkanPointers.device,
        mVertexBuffer, mVertexBufferMemory, 0) != VK_SUCCESS) {
        qFatal("Failed to bind vertex buffer memory!");
    }
}

/// <summary>
/// This function destroys the vertex buffer and releases its memory.
/// </summary>
void Renderer::deleteVertexBuffer() {
    if (mVertexBuffer) {
        mVulkanPointers.pDeviceFunctions->vkDestroyBuffer(mVulkanPointers.device, mVertexBuffer, nullptr);
        mVulkanPointers.pDeviceFunctions->vkFreeMemory(mVulkanPointers.device, mVertexBufferMemory, nullptr);
        mVertexBuffer = VK_NULL_HANDLE;
        mVertexBufferMemory = VK_NULL_HANDLE;
    }
}


void Renderer::traverse(int& nodeIndex, tinygltf::Model& model, 
    std::vector<glm::vec3>& vertices, glm::mat4x4 transformation) {

    // Each node in glTF does or doesn't have a set of meshes. However, tinygltf can have only
    // single mesh in a node. Let's investigate it.
    int meshIndex = -1;
    if (model.nodes.size() > nodeIndex) {
        meshIndex = model.nodes[nodeIndex].mesh;

        // If meshIndex is invalid, skip it.
        if (meshIndex < 0) goto down;

        std::vector<double> vec = model.nodes[nodeIndex].matrix;
        if (vec.size() == 16) {
            glm::mat4x4 mat = { vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7],
                vec[8], vec[9], vec[10], vec[11], vec[12], vec[13], vec[14], vec[15] };
            transformation = transformation * mat;
        }

        // Let's enum every primitive in a mesh.
        for (int j = 0; j < model.meshes[meshIndex].primitives.size(); j++) {

            // Each primitive has it's part of the vertex data, ie. indexes of accessors where
            // vertex position, normal and texture coordinate data as well other attributes can be 
            // acqured. CreateVertexBuffer creates a vertex buffer, so we are only interested to get 
            // vertex positions. They can be served as indexed or non indexed data.
            int indexAccessor = -1;
            int positionAccessor = -1;
            try {
                auto temp = model.meshes[meshIndex].primitives[j].attributes.find("POSITION");
                positionAccessor = temp->second;
                indexAccessor = model.meshes[meshIndex].primitives[j].indices;
            }
            catch (...) {

                // Three dots means we catch any exception and continue.
                continue;
            }
            getPositions(vertices, model, positionAccessor, indexAccessor, transformation);
        }
    }

    down:

    // Meshes are done for this node. Now check are there any child nodes.
    std::vector<int> children = model.nodes[nodeIndex].children;
    for (int j = 0; j < children.size(); j++) {
        traverse(children[j], model, vertices, transformation);
    }
}

/// <summary>
/// This helper function reads vertexes from tinygltf model into a std::vector. 
/// </summary>
/// <param name="vertices">A vector where vertices are added into.</param>
/// <param name="model">The source of vertices.</param>
/// <param name="posAcc">A glTF accessor index where to find vertex positions.</param>
/// <param name="indAcc">A glTF accessor index where to find vertex indices. If not available, ingnore.</param>
void Renderer::getPositions(std::vector<glm::vec3>& vertices,
    tinygltf::Model& model, int& posAcc, int indAcc, glm::mat4x4 transformation) {

    // Always need position accessor.
    if (posAcc < 0) return;

    // Index accessor is not needed necessarily.
    if (indAcc < 0) {

        // We have only positions. Ensure they are float 3D vectors.
        if (model.accessors[posAcc].componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
            model.accessors[posAcc].type == TINYGLTF_TYPE_VEC3) {

            // Get the index and offset of the bufferView to get the buffer.
            int bufferView = model.accessors[posAcc].bufferView;
            int byteOffsetBufferView = model.accessors[posAcc].byteOffset;

            // Get the index, offset of the stride, lengt and stride of the buffer 
            // to get the vectors.
            int bufferIndex = model.bufferViews[bufferView].buffer;
            int byteOffsetStride = model.bufferViews[bufferView].byteOffset;
            int byteLength = model.bufferViews[bufferView].byteLength;
            int byteStride = std::max(model.bufferViews[bufferView].byteStride, size_t(12));

            // Store each position into the vertices vector. Suppose big endian
            // byte order.
            std::vector<unsigned char> buffer = model.buffers[bufferIndex].data;
            for (int i = 0; i < (byteLength / byteStride); i++) {

                uint8_t c1 = buffer[3 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                uint8_t c2 = buffer[2 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                uint8_t c3 = buffer[1 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                uint8_t c4 = buffer[0 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                uint32_t four = ((c1 << 24) | (c2 << 16) | (c3 << 8) | (c4 << 0));
                float pos1;
                std::memcpy(&pos1, &four, sizeof(float));

                c1 = buffer[7 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                c2 = buffer[6 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                c3 = buffer[5 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                c4 = buffer[4 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                four = ((c1 << 24) | (c2 << 16) | (c3 << 8) | (c4 << 0));
                float pos2;
                std::memcpy(&pos2, &four, sizeof(float));

                c1 = buffer[11 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                c2 = buffer[10 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                c3 = buffer[9 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                c4 = buffer[8 + i * byteStride + byteOffsetBufferView + byteOffsetStride];
                four = ((c1 << 24) | (c2 << 16) | (c3 << 8) | (c4 << 0));
                float pos3;
                std::memcpy(&pos3, &four, sizeof(float));

                // Need to transform vertex position into its absolute position in space.
                // This makes the Model matrix computation in common ModelViewProjection
                // matrix computation chain.
                glm::vec4 position(pos1, pos2, pos3, 1.0);
                position = transformation * position;

                // Need to update minmax.
                this->setMinMax(position);

                vertices.push_back(position);
            }
        }
    }
    else {

        // We have indexers too. Ensure they are unsigned short scalars and positions
        // are float 3D vectors.
        if (model.accessors[indAcc].componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT &&
            model.accessors[indAcc].type == TINYGLTF_TYPE_SCALAR &&
            model.accessors[posAcc].componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
            model.accessors[posAcc].type == TINYGLTF_TYPE_VEC3) {
        
            // Get the indexes and offsets of the bufferViews to get the data of
            // positions and indexes.
            int bufferViewPos = model.accessors[posAcc].bufferView;
            int byteOffsetBufferViewPos = model.accessors[posAcc].byteOffset;
            int bufferViewInd = model.accessors[indAcc].bufferView;
            int byteOffsetBufferViewInd = model.accessors[indAcc].byteOffset;

            // Get the index, offset of the stride, lengt and stride of the buffer 
            // to get the vectors and their indexes.
            int bufferIndexPos = model.bufferViews[bufferViewPos].buffer;
            int byteOffsetStridePos = model.bufferViews[bufferViewPos].byteOffset;
            int byteLengthPos = model.bufferViews[bufferViewPos].byteLength;
            int byteStridePos = std::max(model.bufferViews[bufferViewPos].byteStride, size_t(12));
            int bufferIndexInd = model.bufferViews[bufferViewInd].buffer;
            int byteOffsetStrideInd = model.bufferViews[bufferViewInd].byteOffset;
            int byteLengthInd = model.bufferViews[bufferViewInd].byteLength;
            int byteStrideInd = std::max(model.bufferViews[bufferViewInd].byteStride, size_t(2));

            // Store each position into the vertices vector. Suppose big endian
            // byte order. Now each scalar index represent the location of one position 
            // whose vector can be found in that index.
            std::vector<unsigned char> bufferInd = model.buffers[bufferIndexInd].data;
            std::vector<unsigned char> bufferPos = model.buffers[bufferIndexPos].data;
            for (int i = 0; i < (byteLengthInd / byteStrideInd); i++) {

                uint16_t index = ((uint16_t)((bufferInd[1 + i * byteStrideInd + byteOffsetBufferViewInd + byteOffsetStrideInd] << 8) |
                    bufferInd[0 + i * byteStrideInd + byteOffsetBufferViewInd + byteOffsetStrideInd]));
               
                uint8_t c1 = bufferPos[3 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                uint8_t c2 = bufferPos[2 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                uint8_t c3 = bufferPos[1 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                uint8_t c4 = bufferPos[0 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                uint32_t four = ((c1 << 24) | (c2 << 16) | (c3 << 8) | (c4 << 0));
                float pos1;
                std::memcpy(&pos1, &four, sizeof(float));

                c1 = bufferPos[7 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                c2 = bufferPos[6 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                c3 = bufferPos[5 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                c4 = bufferPos[4 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                four = ((c1 << 24) | (c2 << 16) | (c3 << 8) | (c4 << 0));
                float pos2;
                std::memcpy(&pos2, &four, sizeof(float));

                c1 = bufferPos[11 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                c2 = bufferPos[10 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                c3 = bufferPos[9 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                c4 = bufferPos[8 + index * byteStridePos + byteOffsetBufferViewPos + byteOffsetStridePos];
                four = ((c1 << 24) | (c2 << 16) | (c3 << 8) | (c4 << 0));
                float pos3;
                std::memcpy(&pos3, &four, sizeof(float));
                
                // Need to transform vertex position into its absolute position in space.
                // This makes the Model matrix computation in common ModelViewProjection
                // matrix computation chain.
                glm::vec4 position(pos1, pos2, pos3, 1.0);
                position = transformation * position;

                // Need to update minmax.
                this->setMinMax(position);

                vertices.push_back(position);
            }
        }
    }
}

/// <summary>
/// This function is used to find out the absolute bounding box consisting the whole scene.
/// </summary>
/// <param name="vec"></param>
void Renderer::setMinMax(glm::vec4& vec) {
    mMax.x = std::max(mMax.x, vec.x);
    mMax.y = std::max(mMax.y, vec.y);
    mMax.z = std::max(mMax.z, vec.z);
    mMin.x = std::min(mMin.x, vec.x);
    mMin.y = std::min(mMin.y, vec.y);
    mMin.z = std::min(mMin.z, vec.z);
}

/// <summary>
/// Uniform buffers are formatted here. We have mUboMVPMatrices struct only as a buffer,
/// but we need MAX_FRAMES_IN_FLIGHT number of buffers for it.
/// </summary>
void Renderer::createUniformBuffers() {

    VkDeviceSize bufferSize = sizeof(mUboMVPMatrices);

        // Create a uniform buffer for Vulkan...
        // Uniform buffer creation step 1.
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (mVulkanPointers.pDeviceFunctions->vkCreateBuffer(
            mVulkanPointers.device, &bufferInfo, nullptr, &mUniformBuffer) != VK_SUCCESS) {
            std::cout << "Failed to create uniform buffer!\n";
            throw std::runtime_error("Failed to create uniform buffer!");
        }

        // ...tell what kind of memory is suitable for our buffer...
        // Uniform buffer creation step 2.
        VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        VkMemoryRequirements memRequirements{};
        mVulkanPointers.pDeviceFunctions->vkGetBufferMemoryRequirements(
            mVulkanPointers.device, mUniformBuffer, &memRequirements);
        VkPhysicalDeviceMemoryProperties memProperties{};
        mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceMemoryProperties(
            mVulkanPointers.physicalDevice, &memProperties);
        /*
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((memProperties.memoryTypes[i].propertyFlags & properties) == properties &&
                (memRequirements.memoryTypeBits & (1 << i))) {
                properties = i;
                break;
            }
            if (i == memProperties.memoryTypeCount - 1) {
                qFatal("Failed to find suitable memory type for uniform buffer!");
            }
        }
        */
        for (uint32_t i = 0; i < 32; ++i) {
            VkMemoryType memoryType = memProperties.memoryTypes[i];
            if (memRequirements.memoryTypeBits & 1) {
                if ((memoryType.propertyFlags & properties) == properties) {
                    properties = i;
                    break;
                }
            }
            memRequirements.memoryTypeBits = memRequirements.memoryTypeBits >> 1;
        }

        // ...create that buffer memory...
        // Uniform buffer creation step 3.
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = properties;
        if (mVulkanPointers.pDeviceFunctions->vkAllocateMemory(
            mVulkanPointers.device, &allocInfo, nullptr, &mUniformBufferMemory) != VK_SUCCESS) {
            qFatal("Failed to allocate uniform buffer memory!");
        }

        // ...and finally bind that buffer.
        // Uniform buffer creation step 4.
        if (mVulkanPointers.pDeviceFunctions->vkBindBufferMemory(mVulkanPointers.device, 
            mUniformBuffer, mUniformBufferMemory, 0) != VK_SUCCESS) {
            qFatal("Failed to bind uniform buffer memory.");
        }
    
}
/// <summary>
/// This function destroys the uniform buffer and releases its memory.
/// </summary>
void Renderer::deleteUniformBuffers() {
    if(mUniformBuffer) {
        mVulkanPointers.pDeviceFunctions->vkDestroyBuffer(mVulkanPointers.device, mUniformBuffer, nullptr);
        mVulkanPointers.pDeviceFunctions->vkFreeMemory(mVulkanPointers.device, mUniformBufferMemory, nullptr);
        mUniformBuffer = VK_NULL_HANDLE;
        mUniformBufferMemory = VK_NULL_HANDLE;
    }
}

/// <summary>
/// This helper function updates the given uniform buffer of mUniformBuffers.
/// </summary>
void Renderer::updateUniformBuffer(uint32_t currentImage) {

    // Fill uniform buffer with the mUboMVPMatrices struct.
    // Uniform buffer creation step 9. 
    void* data;
    if (mVulkanPointers.pDeviceFunctions->vkMapMemory(
        mVulkanPointers.device, mUniformBufferMemory, 0,
        (size_t)sizeof(mUboMVPMatrices), 0, &data) != VK_SUCCESS) {
        qFatal("Failed to map uniform buffer memory.");
    }

    memcpy(data, &mUboMVPMatrices, (size_t)sizeof(mUboMVPMatrices));
            
    VkMappedMemoryRange memoryRange {};
    memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    memoryRange.memory = mUniformBufferMemory;
    memoryRange.offset = 0;
    memoryRange.size = VK_WHOLE_SIZE;

    if (mVulkanPointers.pDeviceFunctions->vkFlushMappedMemoryRanges(
        mVulkanPointers.device, 1, &memoryRange) != VK_SUCCESS) {
        qFatal("Failed to flush mapped memory.");
    }
            
    mVulkanPointers.pDeviceFunctions->vkUnmapMemory(mVulkanPointers.device,
        mUniformBufferMemory);   
}

/// <summary>
/// This function does the drawing of a single frame.
/// </summary>
void Renderer::render() {

    // First synchonization.
    VkSemaphore presentCompleteSemaphore, renderingCompleteSemaphore;
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = 0;
    semaphoreCreateInfo.flags = 0;
    mVulkanPointers.pDeviceFunctions->vkCreateSemaphore(
        mVulkanPointers.device, &semaphoreCreateInfo, nullptr, &presentCompleteSemaphore);
    mVulkanPointers.pDeviceFunctions->vkCreateSemaphore(
        mVulkanPointers.device, &semaphoreCreateInfo, nullptr, &renderingCompleteSemaphore);
 
    // Need to set model matrix to rotate the scene.
    setModelMatrix();
    
    // Then we need to request image index from the swapchain.
    uint32_t imageIndex;
    vkAcquireNextImageKHR(mVulkanPointers.device, mSwapChain, UINT64_MAX,
        presentCompleteSemaphore, VK_NULL_HANDLE, &imageIndex);

    // Update mUboMVPMatrices struct data for the right frame in flight.
    if (imageIndex < MAX_FRAMES_IN_FLIGHT) updateUniformBuffer(imageIndex);

    // Time to begin the render command buffer.
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkCommandBuffer cmdBuf = mSwapChainRes.renderCmdBuffer;
    mVulkanPointers.pDeviceFunctions->vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Barrier for reading from uniform buffer after all writing is done.
    VkMemoryBarrier uniformMemoryBarrier {};
    uniformMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    uniformMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    uniformMemoryBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
    mVulkanPointers.pDeviceFunctions->vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0,
        1, &uniformMemoryBarrier,
        0, nullptr,
        0, nullptr);

    // Change image layout from VK_IMAGE_LAYOUT_PRESENT_SRC_KHR to 
    // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
    VkImageMemoryBarrier layoutTransitionBarrier {};
    layoutTransitionBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    layoutTransitionBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    layoutTransitionBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    layoutTransitionBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    layoutTransitionBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    layoutTransitionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    layoutTransitionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    layoutTransitionBarrier.image = mSwapChainRes.images[imageIndex];
    VkImageSubresourceRange resourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    layoutTransitionBarrier.subresourceRange = resourceRange;
    mVulkanPointers.pDeviceFunctions->vkCmdPipelineBarrier(
        cmdBuf,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &layoutTransitionBarrier);

    // Activate render pass.
    QSize swapChainImageSize = mVulkanPointers.pVulkanWindow->size() *
        mVulkanPointers.pVulkanWindow->devicePixelRatio();
    VkClearValue clearValue[] = { { 0.67f, 0.84f, 0.9f, 1.0f }, { 1.0, 0.0 } };
    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = mSwapChainRes.renderPass;
    renderPassBeginInfo.framebuffer = mSwapChainRes.frameBuffers[imageIndex];
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = swapChainImageSize.width();
    renderPassBeginInfo.renderArea.extent.height = swapChainImageSize.height();
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValue;
    mVulkanPointers.pDeviceFunctions->vkCmdBeginRenderPass(
        cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind the graphics pipeline to the command buffer.
    mVulkanPointers.pDeviceFunctions->vkCmdBindPipeline(
        cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline);

    // Take care of dynamic state.
    VkViewport viewport = { 
        0, 0, (float)mSwapChainImageSize.width(), (float)mSwapChainImageSize.height(), 0, 1 };
    mVulkanPointers.pDeviceFunctions->vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
    VkRect2D scissor = { 0, 0, mSwapChainImageSize.width(), mSwapChainImageSize.height() };
    mVulkanPointers.pDeviceFunctions->vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

    // Bind shader parameters.
    VkBuffer vertexBuffers[] = { mVertexBuffer };
    VkDeviceSize vbOffsets { };
    mVulkanPointers.pDeviceFunctions->vkCmdBindVertexBuffers(cmdBuf, 0, 1, vertexBuffers, &vbOffsets);
    mVulkanPointers.pDeviceFunctions->vkCmdBindDescriptorSets(cmdBuf,
        VK_PIPELINE_BIND_POINT_GRAPHICS, mVulkanPointers.pipelineLayout, 0, 1,
        &mDescriptorSet, 0, nullptr);

    // Render triangles.
    mVulkanPointers.pDeviceFunctions->vkCmdDraw(cmdBuf, mVertexBufferSize, 1, 0, 0);
    mVulkanPointers.pDeviceFunctions->vkCmdEndRenderPass(cmdBuf);

    // Change layout back to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
    VkImageMemoryBarrier prePresentBarrier {};
    prePresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    prePresentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    prePresentBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    prePresentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    prePresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    prePresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prePresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prePresentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    prePresentBarrier.image = mSwapChainRes.images[imageIndex];
    mVulkanPointers.pDeviceFunctions->vkCmdPipelineBarrier(
        cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
        0, nullptr, 0, nullptr, 1, &prePresentBarrier);
    mVulkanPointers.pDeviceFunctions->vkEndCommandBuffer(cmdBuf);

    // present:
    VkFence renderFence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    mVulkanPointers.pDeviceFunctions->vkCreateFence(
        mVulkanPointers.device, &fenceCreateInfo, nullptr, &renderFence);

    VkPipelineStageFlags waitStageMash = { VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &presentCompleteSemaphore;
    submitInfo.pWaitDstStageMask = &waitStageMash;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderingCompleteSemaphore;
    mVulkanPointers.pDeviceFunctions->vkQueueSubmit(
        mVulkanPointers.pVulkanWindow->graphicsQueue(), 1, &submitInfo, renderFence);

    mVulkanPointers.pDeviceFunctions->vkWaitForFences(
        mVulkanPointers.device, 1, &renderFence, VK_TRUE, UINT64_MAX);
    mVulkanPointers.pDeviceFunctions->vkDestroyFence(
        mVulkanPointers.device, renderFence, nullptr);

    VkPresentInfoKHR presentInfo {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderingCompleteSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &mSwapChain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(mVulkanPointers.pVulkanWindow->graphicsQueue(), &presentInfo);

    mVulkanPointers.pDeviceFunctions->vkDestroySemaphore(
        mVulkanPointers.device, presentCompleteSemaphore, nullptr);
    mVulkanPointers.pDeviceFunctions->vkDestroySemaphore(
        mVulkanPointers.device, renderingCompleteSemaphore, nullptr);
        
    // Finally change currentFrame.
    mCurrentFrame = (mCurrentFrame + 1) % mActualSwapChainBufferCount;
    
}

/// <summary>
/// This function establishes rendering pipeline.
/// </summary>
void Renderer::createGraphicsPipeline() {

    // Create shaders.
    mVertShaderModule = createShaderModule(VERTEX_SPIR);
    mFragShaderModule = createShaderModule(FRAGMENT_SPIR);

    // Vertex array creation step 6.
    VkPipelineShaderStageCreateInfo vertShaderStageInfo {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = mVertShaderModule;
    vertShaderStageInfo.pName = "main";
    vertShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo fragShaderStageInfo {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = mFragShaderModule;
    fragShaderStageInfo.pName = "main";
    fragShaderStageInfo.pSpecializationInfo = nullptr;

    mPipelineBuilder.shaderStages.clear();
    mPipelineBuilder.shaderStages.push_back(vertShaderStageInfo);
    mPipelineBuilder.shaderStages.push_back(fragShaderStageInfo);

    // Create vertex buffer binding.
    // Vertex array creation step 7.
    mPipelineBuilder.vertexBindingDesc.binding = 0;
    mPipelineBuilder.vertexBindingDesc.stride = 3 * sizeof(float);
    mPipelineBuilder.vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Vertex array creation step 8.   
    mPipelineBuilder.vertexAttrDesc.location = 0;
    mPipelineBuilder.vertexAttrDesc.binding = 0;
    mPipelineBuilder.vertexAttrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
    mPipelineBuilder.vertexAttrDesc.offset = 0;

    // Vertex array creation step 9.
    mPipelineBuilder.vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    mPipelineBuilder.vertexInputInfo.pNext = nullptr;
    mPipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = 1;
    mPipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = &mPipelineBuilder.vertexBindingDesc;
    mPipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = 1;
    mPipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = &mPipelineBuilder.vertexAttrDesc;

    // Create input assembly.
    // Vertex array creation step 10.
    mPipelineBuilder.inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    mPipelineBuilder.inputAssembly.pNext = nullptr;
    mPipelineBuilder.inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    mPipelineBuilder.inputAssembly.primitiveRestartEnable = VK_FALSE;
    /*
    //build viewport and scissor from the swapchain extents
    pipelineBuilder._viewport.x = 0.0f;
    pipelineBuilder._viewport.y = 0.0f;
    pipelineBuilder._viewport.width = (float)_windowExtent.width;
    pipelineBuilder._viewport.height = (float)_windowExtent.height;
    pipelineBuilder._viewport.minDepth = 0.0f;
    pipelineBuilder._viewport.maxDepth = 1.0f;

    pipelineBuilder._scissor.offset = { 0, 0 };
    pipelineBuilder._scissor.extent = _windowExtent;
    */
    // Create rasterization state.
    mPipelineBuilder.rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    mPipelineBuilder.rasterizer.pNext = nullptr;
    mPipelineBuilder.rasterizer.depthClampEnable = VK_FALSE;

    // Discards all primitives before the rasterization stage if enabled which we don't want.
    mPipelineBuilder.rasterizer.rasterizerDiscardEnable = VK_FALSE;
    mPipelineBuilder.rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    mPipelineBuilder.rasterizer.lineWidth = 1.0f;

    // No backface culling.
    mPipelineBuilder.rasterizer.cullMode = VK_CULL_MODE_NONE;
    mPipelineBuilder.rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

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
    mPipelineBuilder.multisampling.minSampleShading = 1.0f;
    mPipelineBuilder.multisampling.pSampleMask = nullptr;
    mPipelineBuilder.multisampling.alphaToCoverageEnable = VK_FALSE;
    mPipelineBuilder.multisampling.alphaToOneEnable = VK_FALSE;

    // Create depth stencil state.
    mPipelineBuilder.depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    mPipelineBuilder.depthStencil.depthTestEnable = VK_TRUE;
    mPipelineBuilder.depthStencil.depthWriteEnable = VK_TRUE;
    mPipelineBuilder.depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Create color blend.
    mPipelineBuilder.colorBlendAttachment.colorWriteMask = 
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    mPipelineBuilder.colorBlendAttachment.blendEnable = VK_FALSE;

    // Create dynamic state info.
    mPipelineBuilder.dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    mPipelineBuilder.dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    mPipelineBuilder.dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    mPipelineBuilder.dynamic.dynamicStateCount = 2;
    mPipelineBuilder.dynamic.pDynamicStates = mPipelineBuilder.dynamicStates.data();

    // Create descriptor set bindings. All bindings declared in the shaders need a descriptor set.
    // We have one uniform buffer binding in vertex shader and it's name is UniformBufferObject.
    // First create descriptor set layout(s)...
    // Uniform buffer creation step 5.
    VkDescriptorSetLayoutBinding uniformBufferObjectBinding{};
    uniformBufferObjectBinding.binding = 0;
    uniformBufferObjectBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferObjectBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uniformBufferObjectBinding.descriptorCount = 1;
    uniformBufferObjectBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo dsLayoutInfo{};
    dsLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsLayoutInfo.flags = 0;
    dsLayoutInfo.pNext = nullptr;
    dsLayoutInfo.bindingCount = 1;
    dsLayoutInfo.pBindings = &uniformBufferObjectBinding;

    VkResult err = mVulkanPointers.pDeviceFunctions->vkCreateDescriptorSetLayout(
            mVulkanPointers.device, &dsLayoutInfo, nullptr, &mPipelineBuilder.dsLayout);
    if (err != VK_SUCCESS)
            qFatal("vkCreateDescriptorSetLayout failed (%d)", (uint32_t)err);

    // ...next create descriptor pool...
    // Uniform buffer creation step 6.
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    //    poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCreateInfo.flags = 0;
    poolCreateInfo.maxSets = 1;
    poolCreateInfo.pNext = nullptr;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;

    err = mVulkanPointers.pDeviceFunctions->vkCreateDescriptorPool(
        mVulkanPointers.device,
        &poolCreateInfo,
        nullptr,
        &mDescriptorPool);
    if (err != VK_SUCCESS)
        qFatal("vkCreateDescriptorPool failed (%d)", (uint32_t)err);

    // ...then allocate the descriptor set layout(s)...
    // Uniform buffer creation step 7.
    VkDescriptorSetAllocateInfo dsAllocateInfo{};
    dsAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocateInfo.pNext = nullptr;
    dsAllocateInfo.descriptorPool = mDescriptorPool;
    dsAllocateInfo.descriptorSetCount = 1;
    dsAllocateInfo.pSetLayouts = &mPipelineBuilder.dsLayout;
    err = mVulkanPointers.pDeviceFunctions->vkAllocateDescriptorSets(
        mVulkanPointers.device,
        &dsAllocateInfo, &mDescriptorSet);
    if (err != VK_SUCCESS)
        qFatal("vkAllocateDescriptorSets failed (%d)", (uint32_t)err);

    // ...and finally bind and assing an uniform buffer for each descriptor set.
    // Uniform buffer creation step 8.
    VkDescriptorBufferInfo bInfo{};
    bInfo.buffer = mUniformBuffer;
    bInfo.offset = 0;
    bInfo.range = sizeof(mUboMVPMatrices);

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.pNext = nullptr;
    descriptorWrite.dstSet = mDescriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.pBufferInfo = &bInfo;
    descriptorWrite.pImageInfo = nullptr;
    descriptorWrite.pTexelBufferView = nullptr;

    mVulkanPointers.pDeviceFunctions->vkUpdateDescriptorSets(
            mVulkanPointers.device, 1, &descriptorWrite, 0, nullptr);

    // Create pipeline layout. 
    mPipelineBuilder.pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    mPipelineBuilder.pipelineLayout.pNext = nullptr;
    mPipelineBuilder.pipelineLayout.flags = 0;
    mPipelineBuilder.pipelineLayout.setLayoutCount = 1;
    mPipelineBuilder.pipelineLayout.pSetLayouts = &mPipelineBuilder.dsLayout;
    mPipelineBuilder.pipelineLayout.pushConstantRangeCount = 0;
    mPipelineBuilder.pipelineLayout.pPushConstantRanges = nullptr;

    // Now necessary structs are defined and we're ready to build the pipeline when we
    // have a renderpass to build with. We don't use Qt's default renderpass but we create 
    // our own.
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
    mVulkanPointers.pDeviceFunctions->vkDestroyDescriptorPool(
        mVulkanPointers.device,
        mDescriptorPool,
        nullptr
    );
    mDescriptorPool = VK_NULL_HANDLE;  
    mVulkanPointers.pDeviceFunctions->vkDestroyShaderModule(
        mVulkanPointers.device, mFragShaderModule, nullptr);
    mFragShaderModule = VK_NULL_HANDLE;
    mVulkanPointers.pDeviceFunctions->vkDestroyShaderModule(
        mVulkanPointers.device, mVertShaderModule, nullptr);
    mVertShaderModule = VK_NULL_HANDLE;
}

/// <summary>
/// This helper function creates a shader module. Shader can be any type, vertex, fragment etc...
/// </summary>
/// <param name="file">Spv shader file name with format, ex. vertex.spv</param>
/// <returns>Vulkan shader module</returns>
VkShaderModule Renderer::createShaderModule(std::string file) {
    
    std::unique_ptr<uint32_t[]> data32;
    uint32_t length = 0;
    try {

        // Let's see do we find the file from this path.
        std::filesystem::path path = std::filesystem::absolute(file);
        if (std::filesystem::is_regular_file(path)) {

            // Read the contents of a file as char.
            std::unique_ptr<char[]> data;
            std::fstream reader;
            reader.open(path.c_str(), std::ios_base::binary | std::ifstream::in);
            reader.seekg(0, reader.end);
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
            
            // Transform hexadecimal char array to uint32_t array.  
            // Use this option if your SPIR-V is an array of hexadecimal numbers.
            // First separate every hexadecimal number to a single token.
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
            length = index * 4;
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
    shaderInfo.codeSize = length;
    shaderInfo.pCode = data32.get();
    VkResult err = mVulkanPointers.pDeviceFunctions->vkCreateShaderModule(
        mVulkanPointers.device, &shaderInfo, nullptr, &shaderModule);
    if (err != VK_SUCCESS) {
        qFatal("Failed to create shader module: %d", err);
    }
    return shaderModule;
}

/// <summary>
/// This function creates Vulkan swapchain, imageviews, framebuffer and synchronization. If there is 
/// an old swap chain you want to replace, define oldSwapChain parameter.
/// </summary>
/// <param name="oldSwapChain">This swap chain will be replaced.</param>
void Renderer::createSwapChain(int* colorFormat, int* depthFormat, VkSwapchainKHR oldSwapChain) {

    // Test do we have a window.
    mSwapChainImageSize = mVulkanPointers.pVulkanWindow->size() * 
        mVulkanPointers.pVulkanWindow->devicePixelRatio();
    if (mSwapChainImageSize.isEmpty())
        return;

    // Wait until possible pending work done.
    mVulkanPointers.pDeviceFunctions->vkDeviceWaitIdle(mVulkanPointers.device);

    // Get the properties of the window surface QVulkanWindowRenderer offers us. Because
    // Qt doesn't provide us all Vulkan function pointers we need here and vkGetPhysicalDeviceSurfaceSupportKHR,
    // vkGetPhysicalDeviceSurfaceCapabilitiesKHR and vkGetPhysicalDeviceSurfaceFormatsKHR
    // require a physical device connected, we get those function pointers here on demand.
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
        mVulkanPointers.pInstance->getInstanceProcAddr("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
    vkGetPhysicalDeviceSurfaceFormatsKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
        mVulkanPointers.pInstance->getInstanceProcAddr("vkGetPhysicalDeviceSurfaceFormatsKHR"));
    vkGetPhysicalDeviceSurfaceSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        mVulkanPointers.pInstance->getInstanceProcAddr("vkGetPhysicalDeviceSurfaceSupportKHR"));

    // Get the surface formats this physical device can offer and select suitable.
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(mVulkanPointers.physicalDevice, mVulkanPointers.surface, 
        &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(mVulkanPointers.physicalDevice, mVulkanPointers.surface, 
        &formatCount, surfaceFormats.data());

    VkFormat cFormat;
    VkFormat dFormat = (VkFormat)(depthFormat == nullptr ? VK_FORMAT_D24_UNORM_S8_UINT : *depthFormat);
    VkColorSpaceKHR colorSpace;
    if (colorFormat == nullptr) {

        // If the format list includes just one entry of VK_FORMAT_UNDEFINED, the surface has no 
        // preferred format. Otherwise, at least one supported format will be returned.
        if (formatCount == 1 && surfaceFormats[0].format == VK_FORMAT_UNDEFINED) {
            cFormat = VK_FORMAT_B8G8R8_UNORM;
        }
        else {
            cFormat = surfaceFormats[0].format;
        }
        colorSpace = surfaceFormats[0].colorSpace;
    }
    else {
        cFormat = (VkFormat) *colorFormat;
        colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    } 

    // Get the surface capabilities.
    VkSurfaceCapabilitiesKHR surfaceCaps {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mVulkanPointers.physicalDevice, 
        mVulkanPointers.surface, &surfaceCaps);
    
    uint32_t reqBufferCount = 2;
    if (surfaceCaps.maxImageCount == 0)

        // If surfaceCapabilities.maxImageCount == 0 there is no limit on the number of images.
        reqBufferCount = std::max<uint32_t>(2, surfaceCaps.minImageCount);
    else
        reqBufferCount = std::max(std::min<uint32_t>(surfaceCaps.maxImageCount, 2), 
            surfaceCaps.minImageCount);

    VkExtent2D bufferSize = surfaceCaps.currentExtent;
    if (bufferSize.width == uint32_t(-1)) {
        qWarning("VkSurfaceCapabilitiesKHR width is -1 pixels.");
        bufferSize.width = mSwapChainImageSize.width();
        bufferSize.height = mSwapChainImageSize.height();
    }
    else {
        mSwapChainImageSize = QSize(bufferSize.width, bufferSize.height);
    }

    // Set another properties.
    VkSurfaceTransformFlagBitsKHR preTransform =
        (surfaceCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        : surfaceCaps.currentTransform;

    VkCompositeAlphaFlagBitsKHR compositeAlpha =
        (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
        ? VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
        : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    //if (mVulkanPointers.pVulkanWindow->requestedFormat().hasAlpha()) {
    //    if (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
    //        compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    //    else if (surfaceCaps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
    //        compositeAlpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
    //}

    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    bool swapChainSupportsReadBack = (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (swapChainSupportsReadBack)
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    // Create a new swapchain for the given window surface.
    if (oldSwapChain == VK_NULL_HANDLE) oldSwapChain = mSwapChain;
    VkSwapchainCreateInfoKHR swapChainInfo {};
    swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainInfo.flags = 0;
    swapChainInfo.pNext = VK_NULL_HANDLE;
    swapChainInfo.surface = mVulkanPointers.surface;
    swapChainInfo.minImageCount = reqBufferCount;
    swapChainInfo.imageFormat = cFormat;
    swapChainInfo.imageColorSpace = colorSpace;
    swapChainInfo.imageExtent = bufferSize;
    swapChainInfo.imageArrayLayers = 1;
    swapChainInfo.imageUsage = usage;
    swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapChainInfo.preTransform = preTransform;
    swapChainInfo.compositeAlpha = compositeAlpha;
    swapChainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapChainInfo.clipped = true;
    swapChainInfo.oldSwapchain = oldSwapChain;

    VkSwapchainKHR newSwapChain;
    VkResult err = vkCreateSwapchainKHR(mVulkanPointers.device, &swapChainInfo, nullptr, 
        &newSwapChain);
    if (err != VK_SUCCESS) {
        qFatal("QVulkanWindow: Failed to create swap chain: %d", err);
        return;
    }

    // Delete old swapchain and free its memory before assinging a new one.
    if (oldSwapChain)
        deleteSwapChain();
    mSwapChain = newSwapChain;

    // We need two VkImage buffers. Check how many buffers physical device gave us...
    err = vkGetSwapchainImagesKHR(mVulkanPointers.device, mSwapChain, 
        &mActualSwapChainBufferCount, nullptr);
    if (err != VK_SUCCESS || mActualSwapChainBufferCount < 2) {
        qFatal("QVulkanWindow: Failed to get enough swapchain images: %d (count=%d)", 
            err, mActualSwapChainBufferCount);
        return;
    }
    if (mActualSwapChainBufferCount > (MAX_FRAMES_IN_FLIGHT)) {
        qFatal("QVulkanWindow: Too many swapchain buffers (%d)", mActualSwapChainBufferCount);
        return;
    }

    //...and then get those swapchain image buffers.
    err = vkGetSwapchainImagesKHR(mVulkanPointers.device, mSwapChain, 
        &mActualSwapChainBufferCount, mSwapChainRes.images);
    if (err != VK_SUCCESS) {
        qFatal("QVulkanWindow: Failed to get swapchain images: %d", err);
        return;
    }

    // Create VkImageViews for our swap chain VkImage buffers.
    VkImageViewCreateInfo imgViewInfo {};
    imgViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imgViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imgViewInfo.format = cFormat;
    imgViewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    imgViewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    imgViewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    imgViewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    imgViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imgViewInfo.subresourceRange.baseMipLevel = 0;
    imgViewInfo.subresourceRange.levelCount = 1;
    imgViewInfo.subresourceRange.baseArrayLayer = 0;
    imgViewInfo.subresourceRange.layerCount = 1;

    // Swap chain setup needs command buffer where renderpasses can be executed.
    VkCommandBufferBeginInfo cmdBeginInfo {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    cmdBeginInfo.pNext = nullptr;
    cmdBeginInfo.pInheritanceInfo = nullptr;

    // This fence is needed for waiting async Vulkan functions to accomplish their task.
    VkFenceCreateInfo fenceCreateInfo {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence submitFence;
    mVulkanPointers.pDeviceFunctions->vkCreateFence(
        mVulkanPointers.device, &fenceCreateInfo, nullptr, &submitFence);

    // Qt does not give us a CommandPool, which has VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    // set. Therefore we need a separate command pool. First need to get suitable queue
    // family index...
    uint32_t presentQueueIdx = 0;
    VkPhysicalDeviceProperties deviceProperties {};
    mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceProperties(
        mVulkanPointers.physicalDevice, &deviceProperties);

    uint32_t queueFamilyCount = 0;
    mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceQueueFamilyProperties(
        mVulkanPointers.physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceQueueFamilyProperties(
        mVulkanPointers.physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

    for (uint32_t j = 0; j < queueFamilyCount; ++j) {

        VkBool32 supportsPresent;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            mVulkanPointers.physicalDevice, j, mVulkanPointers.surface, &supportsPresent);

        if (supportsPresent && queueFamilyProperties[j].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            presentQueueIdx = j;
            break;
        }
    }

    // ...then create extra command pool which has suitable properties.
    VkCommandPoolCreateInfo commandPoolCreateInfo {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = presentQueueIdx;
    commandPoolCreateInfo.pNext = nullptr;  
    err = mVulkanPointers.pDeviceFunctions->vkCreateCommandPool(
        mVulkanPointers.device, &commandPoolCreateInfo, nullptr, &mSwapChainRes.commandPool);
    if (err != VK_SUCCESS)
        qFatal("Failed to create setup command pool.");

    // This command buffer is needed to setup a swap chain.
    VkCommandBufferAllocateInfo commandBufferAllocationInfo = {};
    commandBufferAllocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocationInfo.commandPool = mSwapChainRes.commandPool;
    commandBufferAllocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocationInfo.commandBufferCount = 1;
    err = mVulkanPointers.pDeviceFunctions->vkAllocateCommandBuffers(
        mVulkanPointers.device, &commandBufferAllocationInfo, &mSwapChainRes.setupCmdBuffer);
    if (err != VK_SUCCESS)
        qFatal("Failed to allocate setup command buffer.");

    // This command buffer is needed for rendering. We cannot use command buffer delivered
    // by Qt, because we want to implement our own synchronization.
    err = mVulkanPointers.pDeviceFunctions->vkAllocateCommandBuffers(
        mVulkanPointers.device, &commandBufferAllocationInfo, &mSwapChainRes.renderCmdBuffer);
    if (err != VK_SUCCESS)
        qFatal("Failed to allocate render command buffer.");

    // When swap chain changes between VkImage a layout transition is happening.
    // There a proper syncronization is needed using semaphores.
    bool transitioned[MAX_FRAMES_IN_FLIGHT];
    memset(transitioned, 0, sizeof(bool) * MAX_FRAMES_IN_FLIGHT);
    uint32_t doneCount = 0;
    while (doneCount != mActualSwapChainBufferCount) {

        VkSemaphore presentCompleteSemaphore;
        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCreateInfo.flags = 0;
        semaphoreCreateInfo.pNext = 0;
        mVulkanPointers.pDeviceFunctions->vkCreateSemaphore(
            mVulkanPointers.device, &semaphoreCreateInfo, nullptr, &presentCompleteSemaphore);

        // Here we get current VkImage index. 
        uint32_t nextImageIdx;
        vkAcquireNextImageKHR(mVulkanPointers.device, mSwapChain, UINT64_MAX,
            presentCompleteSemaphore, VK_NULL_HANDLE, &nextImageIdx);

        if (!transitioned[nextImageIdx]) {

            // To change a current VkImage index in this while loop need to use command buffer.
            // Start recording out image layout change barrier on our setup command buffer.
            mVulkanPointers.pDeviceFunctions->vkBeginCommandBuffer(
                mSwapChainRes.setupCmdBuffer, &cmdBeginInfo);

            VkImageMemoryBarrier layoutTransitionBarrier {};
            layoutTransitionBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            layoutTransitionBarrier.srcAccessMask = 0;
            layoutTransitionBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            layoutTransitionBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            layoutTransitionBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            layoutTransitionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            layoutTransitionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            layoutTransitionBarrier.image = mSwapChainRes.images[nextImageIdx];
            VkImageSubresourceRange resourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            layoutTransitionBarrier.subresourceRange = resourceRange;
            mVulkanPointers.pDeviceFunctions->vkCmdPipelineBarrier(
                mSwapChainRes.setupCmdBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &layoutTransitionBarrier);

            // Now we have our barrier for layout transition in command buffer and we end buffer.
            mVulkanPointers.pDeviceFunctions->vkEndCommandBuffer(mSwapChainRes.setupCmdBuffer);

            // Next execute layout transition by submitting it to the command queue.
            VkPipelineStageFlags waitStageMash[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            VkSubmitInfo submitInfo {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &presentCompleteSemaphore;
            submitInfo.pWaitDstStageMask = waitStageMash;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &mSwapChainRes.setupCmdBuffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = nullptr;
            err = mVulkanPointers.pDeviceFunctions->vkQueueSubmit(
                mVulkanPointers.pVulkanWindow->graphicsQueue(), 1, &submitInfo, submitFence);
            if (err != VK_SUCCESS) {
                qFatal("Failed to submit layout transition at swap chain setup.");
            }

            // Then wait execution and reset all.
            mVulkanPointers.pDeviceFunctions->vkWaitForFences(
                mVulkanPointers.device, 1, &submitFence, VK_TRUE, UINT64_MAX);
            mVulkanPointers.pDeviceFunctions->vkResetFences(
                mVulkanPointers.device, 1, &submitFence);
            mVulkanPointers.pDeviceFunctions->vkDestroySemaphore(
                mVulkanPointers.device, presentCompleteSemaphore, nullptr);
            mVulkanPointers.pDeviceFunctions->vkResetCommandBuffer(mSwapChainRes.setupCmdBuffer, 0);
            transitioned[nextImageIdx] = true;
            doneCount++;
        }

        VkPresentInfoKHR presentInfo {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 0;
        presentInfo.pWaitSemaphores = nullptr;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &mSwapChain;
        presentInfo.pImageIndices = &nextImageIdx;
        vkQueuePresentKHR(mVulkanPointers.pVulkanWindow->graphicsQueue(), &presentInfo);
    }

    // Each VkImage buffer need to be wrapped by VkImageView.
    for (uint32_t i = 0; i < mActualSwapChainBufferCount; ++i) {
        imgViewInfo.image = mSwapChainRes.images[i];
        err = mVulkanPointers.pDeviceFunctions->vkCreateImageView(
            mVulkanPointers.device, &imgViewInfo, nullptr, &mSwapChainRes.imageViews[i]);
        if (err != VK_SUCCESS)
            qFatal("Could not create ImageView.");
    }

    // Create one depth image which is used by all swap chain image buffers.
    VkImageCreateInfo imageCreateInfo {};
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

    VkMemoryRequirements memoryRequirements {};
    mVulkanPointers.pDeviceFunctions->vkGetImageMemoryRequirements(
        mVulkanPointers.device, mSwapChainRes.depthImage, &memoryRequirements);
    VkPhysicalDeviceMemoryProperties memProperties {};
    mVulkanPointers.pVulkanFunctions->vkGetPhysicalDeviceMemoryProperties(
        mVulkanPointers.physicalDevice, &memProperties);

    VkMemoryAllocateInfo imageAllocateInfo {};
    imageAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imageAllocateInfo.allocationSize = memoryRequirements.size;

    uint32_t memoryTypeBits = memoryRequirements.memoryTypeBits;
    VkMemoryPropertyFlags desiredMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (uint32_t i = 0; i < 32; ++i) {
        VkMemoryType memoryType = memProperties.memoryTypes[i];
        if (memoryTypeBits & 1) {
            if ((memoryType.propertyFlags & desiredMemoryFlags) == desiredMemoryFlags) {
                imageAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }
        memoryTypeBits = memoryTypeBits >> 1;
    }

    VkDeviceMemory imageMemory {};
    err = mVulkanPointers.pDeviceFunctions->vkAllocateMemory(
        mVulkanPointers.device, &imageAllocateInfo, nullptr, &imageMemory);
    if (err != VK_SUCCESS)
        qFatal("Failed to allocate device memory for depth image.");

    err = mVulkanPointers.pDeviceFunctions->vkBindImageMemory(
        mVulkanPointers.device, mSwapChainRes.depthImage, imageMemory, 0);
    if (err != VK_SUCCESS)
        qFatal("Failed to bind depth image memory.");

    // Before using this depth buffer we must change it's layout.
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    mVulkanPointers.pDeviceFunctions->vkBeginCommandBuffer(mSwapChainRes.setupCmdBuffer, &beginInfo);

    VkImageMemoryBarrier layoutTransitionBarrier {};
    layoutTransitionBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    layoutTransitionBarrier.srcAccessMask = 0;
    layoutTransitionBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    layoutTransitionBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    layoutTransitionBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    layoutTransitionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    layoutTransitionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    layoutTransitionBarrier.image = mSwapChainRes.depthImage;
    VkImageSubresourceRange resourceRange = { 
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 };
    layoutTransitionBarrier.subresourceRange = resourceRange;
    mVulkanPointers.pDeviceFunctions->vkCmdPipelineBarrier(
        mSwapChainRes.setupCmdBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &layoutTransitionBarrier);

    mVulkanPointers.pDeviceFunctions->vkEndCommandBuffer(mSwapChainRes.setupCmdBuffer);

    VkPipelineStageFlags waitStageMash[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = waitStageMash;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &mSwapChainRes.setupCmdBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;
    err = mVulkanPointers.pDeviceFunctions->vkQueueSubmit(
        mVulkanPointers.pVulkanWindow->graphicsQueue(), 1, &submitInfo, submitFence);
    if (err != VK_SUCCESS) {
        qFatal("Failed to submit depth buffer layout change at swap chain setup.");
    }

    mVulkanPointers.pDeviceFunctions->vkWaitForFences(
        mVulkanPointers.device, 1, &submitFence, VK_TRUE, UINT64_MAX);
    mVulkanPointers.pDeviceFunctions->vkResetFences(mVulkanPointers.device, 1, &submitFence);
    mVulkanPointers.pDeviceFunctions->vkResetCommandBuffer(mSwapChainRes.setupCmdBuffer, 0);

    // Create the depth image view and wrap depth image.
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    VkImageViewCreateInfo imageViewCreateInfo {};
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

    // Create render pass attachments for color and depth image views.
    VkAttachmentDescription passAttachments[2] = { };
    passAttachments[0].format = cFormat;
    passAttachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    passAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    passAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    passAttachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    passAttachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    passAttachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    passAttachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    passAttachments[1].format = dFormat;
    passAttachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    passAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    passAttachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    passAttachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    passAttachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    passAttachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    passAttachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentReference = {};
    colorAttachmentReference.attachment = 0;
    colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentReference = {};
    depthAttachmentReference.attachment = 1;
    depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Create one main subpass of our renderpass.
    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentReference;
    subpass.pDepthStencilAttachment = &depthAttachmentReference;

    // Create the main renderpass.
    VkRenderPassCreateInfo renderPassCreateInfo = {};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = 2;
    renderPassCreateInfo.pAttachments = passAttachments;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpass;
    err = mVulkanPointers.pDeviceFunctions->vkCreateRenderPass(
        mVulkanPointers.device, &renderPassCreateInfo, nullptr, &mSwapChainRes.renderPass);
    if (err != VK_SUCCESS)
        qFatal("Failed to create renderpass.");

    // Create frame buffer for each index in swap chain with two attachments, color and depth.
    VkImageView frameBufferAttachments[2];
    frameBufferAttachments[1] = mSwapChainRes.depthImageView;

    VkFramebufferCreateInfo frameBufferCreateInfo {};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.renderPass = mSwapChainRes.renderPass;
    frameBufferCreateInfo.attachmentCount = 2;
    frameBufferCreateInfo.pAttachments = frameBufferAttachments;
    frameBufferCreateInfo.width = mSwapChainImageSize.width();
    frameBufferCreateInfo.height = mSwapChainImageSize.height();
    frameBufferCreateInfo.layers = 1;

    for (uint32_t i = 0; i < mActualSwapChainBufferCount; ++i) {
        frameBufferAttachments[0] = mSwapChainRes.imageViews[i];
        err = mVulkanPointers.pDeviceFunctions->vkCreateFramebuffer(
            mVulkanPointers.device, &frameBufferCreateInfo, nullptr, &mSwapChainRes.frameBuffers[i]);
        if (err != VK_SUCCESS)
            qFatal("Failed to create framebuffer.");
    }

    // Renderpass created, now build a pipeline which uses that renderpass.
    if (mPipeline != VK_NULL_HANDLE) deleteGraphicsPipeline();
    mPipeline = mPipelineBuilder.buildPipeline(mVulkanPointers, mSwapChainRes.renderPass);

    mCurrentFrame = 0;
}

void Renderer::deleteSwapChain() {

    if (mSwapChainRes.commandPool) {
        mVulkanPointers.pDeviceFunctions->vkFreeCommandBuffers(
            mVulkanPointers.device, mSwapChainRes.commandPool,
            1, &mSwapChainRes.setupCmdBuffer);
        mVulkanPointers.pDeviceFunctions->vkDestroyCommandPool(
            mVulkanPointers.device, mSwapChainRes.commandPool, nullptr);
        mSwapChainRes.commandPool = VK_NULL_HANDLE;
    }
    if (mSwapChain) {
        vkDestroySwapchainKHR(mVulkanPointers.device, mSwapChain, nullptr);
        mSwapChain = VK_NULL_HANDLE;
    }
    for (int i = 0; i < mActualSwapChainBufferCount; ++i) {
        if (mSwapChainRes.imageViews[i]) {
            mVulkanPointers.pDeviceFunctions->vkDestroyImageView(mVulkanPointers.device,
                mSwapChainRes.imageViews[i], nullptr);
            mSwapChainRes.imageViews[i] = VK_NULL_HANDLE;
            mVulkanPointers.pDeviceFunctions->vkDestroyFramebuffer(mVulkanPointers.device,
                mSwapChainRes.frameBuffers[i], nullptr);
            mSwapChainRes.frameBuffers[i] = VK_NULL_HANDLE;
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
    if (mSwapChainRes.renderPass) {
        mVulkanPointers.pDeviceFunctions->vkDestroyRenderPass(mVulkanPointers.device,
            mSwapChainRes.renderPass, nullptr);
        mSwapChainRes.renderPass = VK_NULL_HANDLE;
    }
}

