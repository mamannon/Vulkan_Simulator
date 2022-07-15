#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;

void main() {
	mat4 modelView = ubo.model * ubo.view;
	gl_Position = vec4(inPosition, 1.0) * (modelView * ubo.proj);
}