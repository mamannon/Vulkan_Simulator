#ifndef __filereader_h__
#define __filereader_h__


#include "definitions.h"

class FileReader {

	public:

		bool loadFile(std::string name);
		FileReader();
		tinygltf::Model* getModel();
		void traverse(int& nodeIndex,
			std::vector<Vertex>& vertices,
			tinygltf::Model* model = nullptr,
			glm::mat4x4 transformation = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
		void getVertexData(std::vector<Vertex>& vertices,
			tinygltf::Model& model, int& posAcc, int indAcc = -1, int colorAcc = -1,
			glm::mat4x4 transformation = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 });
		void setMinMax(glm::vec4& vec);
		glm::vec3 getMin() { return mMin; }
		glm::vec3 getMax() { return mMax; }
		std::optional<glm::vec4> getColor(tinygltf::Model& model, const int& colorAcc, const unsigned int& index);

	protected:

	private:

		std::unique_ptr<tinygltf::Model> mpModel = nullptr;
		std::unique_ptr<tinygltf::TinyGLTF> mpLoader = nullptr;
		glm::vec3 mMin = { -1, -1, -1 };
		glm::vec3 mMax = { 1, 1, 1 };
};

#endif
