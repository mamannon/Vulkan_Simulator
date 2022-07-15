#ifndef __filereader_h__
#define __filereader_h__


#include "definitions.h"

class FileReader {

	public:

		bool loadFile(std::string name);
		FileReader();
		tinygltf::Model* getModel();

	protected:

	private:

		std::unique_ptr<tinygltf::Model> mpModel = nullptr;
		std::unique_ptr<tinygltf::TinyGLTF> mpLoader = nullptr;
};

#endif
