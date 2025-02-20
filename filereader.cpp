

#include "filereader.h"

static int koe = 0;

FileReader::FileReader() {
	this->mpLoader = std::make_unique<tinygltf::TinyGLTF>();
	this->mpModel = nullptr;
}

/// <summary>
/// LoadFile function opens a glTF file and reads its content into the tinygltf::Model object.
/// </summary>
/// <param name="name">Filepath to the file to open.</param>
/// <returns></returns>
bool FileReader::loadFile(std::string name) {
	std::string err;
	std::string warn;
	tinygltf::Model temp;

//	bool ret = this->mpLoader->LoadBinaryFromFile(&temp, &err, &warn, name);  // This is for .glb files. Does not work!
	bool ret = this->mpLoader->LoadASCIIFromFile(&temp, &err, &warn, name);   // This is for .gltf files.
	if (ret && err.empty() && warn.empty()) {
		this->mpModel = std::make_unique<tinygltf::Model>(temp);
		return true;
	} else {
		if (!warn.empty()) {
			qWarning(warn.c_str());
		}
		if (!err.empty()) {
			qFatal(err.c_str());
		}
		return false;
	}
}

tinygltf::Model* FileReader::getModel() {

	if (this->mpModel == nullptr) return nullptr;
	return this->mpModel.get();

}

/// <summary>
/// Traverse function reads the vertices contents of a single node and recursively reads all child nodes.
/// </summary>
/// <param name="nodeIndex">The index of the node we want to investigate.</param>
/// <param name="vertices">Vector where to store Vertex data.</param>
/// <param name="transformation">Transformation matrix.</param>
/// <param name="model">Optional Model object consisting one glTF file.</param>
void FileReader::traverse(int& nodeIndex, std::vector<Vertex>& vertices, 
    tinygltf::Model* model, glm::mat4x4 transformation) {

    if (model == nullptr) {
        model = this->getModel();
    }
    if (model == nullptr) {
        qFatal("Tinygltf model is null!");
    }

    // Each node in glTF does or doesn't have a set of meshes. However, tinygltf can have only
    // single mesh in a node. Let's investigate it.
    int meshIndex = -1;
    if (model->nodes.size() > nodeIndex) {
        meshIndex = model->nodes[nodeIndex].mesh;

        // If meshIndex is invalid, skip it.
        if (meshIndex < 0) goto down;

        // If this node has transformation matrix, we apply it now.
        std::vector<double> vec = model->nodes[nodeIndex].matrix;
        if (vec.size() == 16) {
            glm::mat4x4 mat = { vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7],
                vec[8], vec[9], vec[10], vec[11], vec[12], vec[13], vec[14], vec[15] };
            transformation = transformation * mat;
        }

        // Let's enum every primitive in a mesh.
        for (int j = 0; j < model->meshes[meshIndex].primitives.size(); j++) {

            // Each primitive has it's part of the vertex data, ie. indexes of accessors where
            // vertex position, normal and texture coordinate data as well other attributes can be 
            // acqured. CreateVertexBuffer creates a vertex buffer, so we are only interested to get 
            // vertex positions. They can be served as indexed or non indexed data.
            int indexAccessor = -1;
            int positionAccessor = -1;
            int colorAccessor = -1;
            try {

                for (std::pair<const std::string, int> attribute : model->meshes[meshIndex].primitives[j].attributes) {
                    
                    // Vertex position data accessor.
                    if (attribute.first == "POSITION") {
                        positionAccessor = attribute.second;
                    }

                    // Vertex first color data accessor.
                    if (attribute.first == "COLOR_0") {
                        colorAccessor = attribute.second;
                    }
                }

                // Possible vertex attributes indexing accessor.
                indexAccessor = model->meshes[meshIndex].primitives[j].indices;
            }
            catch (...) {

                // Three dots means we catch any exception and continue.
                continue;
            }
            getVertexData(vertices, *model, positionAccessor, indexAccessor, colorAccessor, transformation);
        }
    }

down:

    // Meshes are done for this node. Now check are there any child nodes.
    std::vector<int> children = model->nodes[nodeIndex].children;
    for (int j = 0; j < children.size(); j++) {
        traverse(children[j], vertices, model, transformation);
    }
}

/// <summary>
/// GetVertexData function reads vertexes from tinygltf model into a std::vector. 
/// </summary>
/// <param name="vertices">A vector where vertices are added into.</param>
/// <param name="model">The source of all glTF data.</param>
/// <param name="posAcc">A glTF accessor index where to find vertex positions.</param>
/// <param name="indAcc">A glTF accessor index where to find vertex indices. If not available, ingnore.</param>
void FileReader::getVertexData(std::vector<Vertex>& vertices,
    tinygltf::Model& model, int& posAcc, int indAcc, int colorAcc, glm::mat4x4 transformation) {

    // Always need position accessor.
    if (posAcc < 0) return;

    // Index accessor is not needed necessarily.
    if (indAcc < 0) {

        // We want only positions. Ensure they are float 3D vectors.
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

            // Store each position into the vertices vector. glTF buffers have little endian
            // byte order.
            try {
                std::vector<unsigned char> buffer = model.buffers[bufferIndex].data;
                for (int i = 0; i < (model.accessors[posAcc].count); i++) {

                    // Define three position coordinates (X, Y, Z).
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

                    // If any color not returned, we shouldn't accept vertex. But glTF file can contain vertex data
                    // without vertex color. Because we don't have textures or materials implemented yet, 
                    // we paint colorless vertex as green, so we can see it.
                    auto temp = this->getColor(model, colorAcc, i);
                    if (!temp.has_value()) {
                        temp = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
                    }
                    vertices.push_back({ glm::vec3(position), temp.value()});
                }
            }
            catch (...) {
                qWarning("GetVertexData function had exception. Vertex data is invalid.");
            }
        }
    }
    else {

        // We have indexers too. Ensure they are unsigned byte, short or int scalars and positions
        // are float 3D vectors.
        int indexComponentType = -1;
        switch (model.accessors[indAcc].componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: indexComponentType = 1;
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: indexComponentType = 2;
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: indexComponentType = 4;
            break;
        }
        if (indexComponentType != -1 &&
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

            // Store each position into the vertices vector. glTF buffers have little endian
            // byte order. Now each scalar index represent the location of one vertex position 
            // whose vector can be found in that index.
            try {
                std::vector<unsigned char> bufferInd = model.buffers[bufferIndexInd].data;
                std::vector<unsigned char> bufferPos = model.buffers[bufferIndexPos].data;
                for (int i = 0; i < (model.accessors[posAcc].count); i++) {
                    
                    // Define the index.
                    uint32_t index = 0;
                    switch (indexComponentType) {
                    case 4: index = (uint32_t) (
                        ((bufferInd[3 + i * 4 + byteOffsetBufferViewInd + byteOffsetStrideInd] << 24) |
                        (bufferInd[2 + i * 4 + byteOffsetBufferViewInd + byteOffsetStrideInd] << 16) |
                        (bufferInd[1 + i * 4 + byteOffsetBufferViewInd + byteOffsetStrideInd] << 8) |
                        bufferInd[0 + i * 4 + byteOffsetBufferViewInd + byteOffsetStrideInd]));
                        break;
                    case 2: index = (uint32_t) (
                        ((bufferInd[1 + i * 2 + byteOffsetBufferViewInd + byteOffsetStrideInd] << 8) |
                        bufferInd[0 + i * 2 + byteOffsetBufferViewInd + byteOffsetStrideInd]));
                        break;
                    case 1: index = (uint32_t) bufferInd[0 + i + byteOffsetBufferViewInd + byteOffsetStrideInd];
                        break;
                    }
                    
                    // Define three position coordinates (X, Y, Z).
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

                    // If any color not returned, we shouldn't accept vertex. But glTF file can contain vertex data
                    // without vertex color. Because we don't have textures or materials implemented yet, 
                    // we paint colorless vertex as green, so we can see it.
                    auto temp = this->getColor(model, colorAcc, index);
                    if (!temp.has_value()) {
                        temp = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
                    }
                    vertices.push_back({ glm::vec3(position), glm::vec3(temp.value()) });
                }
            }
            catch (...) {
                qWarning("GetVertexData function had exception. Vertex data is invalid.");
            }
        }
    }
}

/// <summary>
/// GetColor function reads the color of a single vertex.
/// </summary>
/// <param name="model">The source of all glTF data.</param>
/// <param name="colorAcc">The index of color accessor.</param>
/// <param name="index">The index of color.</param>
/// <returns>Vec4 color vector.</returns>
std::optional<glm::vec4> FileReader::getColor(tinygltf::Model& model, const int& colorAcc, const unsigned int& index) {

        // Ensure we have correct color data.
        int colorComponentType = -1;
        if (colorAcc == -1) return std::optional<glm::vec4>(std::nullopt);
        switch (model.accessors[colorAcc].componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: colorComponentType = 1;
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: colorComponentType = 2;
            break;
        case TINYGLTF_COMPONENT_TYPE_FLOAT: colorComponentType = 4;
            break;
        }
        if (colorComponentType != -1  && model.accessors[colorAcc].type == TINYGLTF_TYPE_VEC4) {

            // Get the index and offset of the bufferView to get the buffer.
            int bufferView = model.accessors[colorAcc].bufferView;
            int byteOffsetBufferView = model.accessors[colorAcc].byteOffset;

            // Get the index, lengt and stride of the buffer to get the color vector.
            int bufferIndex = model.bufferViews[bufferView].buffer;
            int byteOffsetStride = model.bufferViews[bufferView].byteOffset;
            int byteLength = model.bufferViews[bufferView].byteLength;

            // Parse float values for vec4.
            std::vector<unsigned char> buffer = model.buffers[bufferIndex].data;
            switch (colorComponentType) {

            case 1: {
                float c1 = (float) buffer[3 + index + byteOffsetBufferView + byteOffsetStride];
                float c2 = (float) buffer[2 + index + byteOffsetBufferView + byteOffsetStride];
                float c3 = (float) buffer[1 + index + byteOffsetBufferView + byteOffsetStride];
                float c4 = (float) buffer[0 + index + byteOffsetBufferView + byteOffsetStride];
                return glm::vec4(c1 / 255.0f, c2 / 255.0f, c3 / 255.0f, c4 / 255.0f);
            }

            case 2: {

                // Get the step stride of the color data.
                int byteStride = std::max(model.bufferViews[bufferView].byteStride, size_t(2));

                float c1 = (float) ((buffer[1 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[0 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                float c2 = (float) ((buffer[3 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[2 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                float c3 = (float) ((buffer[5 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[4 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                float c4 = (float) ((buffer[7 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[6 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                return glm::vec4(c1 / 65025.0f, c2 / 65025.0f, c3 / 65025.0f, c4 / 65025.0f);
            }

            case 4: {

                // Get the step stride of the color data.
                int byteStride = std::max(model.bufferViews[bufferView].byteStride, size_t(4));

                uint32_t c1 = (uint32_t) (
                    (buffer[3 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 24) |
                    (buffer[2 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 16) |
                    (buffer[1 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[0 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                uint32_t c2 = (uint32_t) (
                    (buffer[7 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 24) |
                    (buffer[6 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 16) |
                    (buffer[5 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[4 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                uint32_t c3 = (uint32_t) (
                    (buffer[11 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 24) |
                    (buffer[10 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 16) |
                    (buffer[9 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[8 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                uint32_t c4 = (uint32_t) (
                    (buffer[15 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 24) |
                    (buffer[14 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 16) |
                    (buffer[13 + index * byteStride + byteOffsetBufferView + byteOffsetStride] << 8) |
                    (buffer[12 + index * byteStride + byteOffsetBufferView + byteOffsetStride]));
                float cc1, cc2, cc3, cc4;
                std::memcpy(&cc1, &c1, sizeof(float));
                std::memcpy(&cc2, &c2, sizeof(float));
                std::memcpy(&cc3, &c3, sizeof(float));
                std::memcpy(&cc4, &c4, sizeof(float));
                return glm::vec4(cc1, cc2, cc3, cc4);
            }
                
            }           
        }
        return std::optional<glm::vec4>(std::nullopt);
}

/// <summary>
/// SetMinMax function is used to find out the absolute bounding box consisting the whole scene.
/// </summary>
/// <param name="vec">Vertex position vector.</param>
void FileReader::setMinMax(glm::vec4& vec) {
    mMax.x = std::max(mMax.x, vec.x);
    mMax.y = std::max(mMax.y, vec.y);
    mMax.z = std::max(mMax.z, vec.z);
    mMin.x = std::min(mMin.x, vec.x);
    mMin.y = std::min(mMin.y, vec.y);
    mMin.z = std::min(mMin.z, vec.z);
}
