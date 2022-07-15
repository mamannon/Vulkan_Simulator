

#include "filereader.h"

FileReader::FileReader() {
	this->mpLoader = std::make_unique<tinygltf::TinyGLTF>();
	this->mpModel = nullptr;
}

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
