#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec3 aTangent;
layout(location = 4) in vec3 aBitangent;

out vec3 Tangent;
out vec3 Bitangent;
out vec3 Normal;
out vec3 WorldPos;
out vec2 TexCoords;
out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Tangent = normalize(vec3(model * vec4(aTangent, 0.0)));
    Bitangent = normalize(vec3(model * vec4(aBitangent, 0.0)));
    Normal = normalize(vec3(model * vec4(aNormal, 0.0)));
    TBN = mat3(Tangent, Bitangent, Normal);

    WorldPos = (model * vec4(aPos, 1.0)).xyz;
    TexCoords = aTexCoords;
}