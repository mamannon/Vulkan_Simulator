#version 450

layout(location = 0) out vec4 outColor;

void main() {
    vec3 temp = vec3(0.5, 0.0, 0.5);
    outColor = vec4(temp, 1.0);
}