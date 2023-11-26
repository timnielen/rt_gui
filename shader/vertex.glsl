#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 normal;
out vec3 worldPos;

uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * transform * vec4(aPos, 1.0);
    normal = (transform * vec4(aNormal, 0.0)).xyz;
    //normal = mat3(transpose(inverse(transform))) * aNormal;
    worldPos = (transform * vec4(aPos, 1.0)).xyz;
}