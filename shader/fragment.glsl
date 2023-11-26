#version 330 core
out vec4 FragColor;

in vec3 color;
in vec3 normal;
in vec3 worldPos;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;


void main()
{
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    vec3 lightDir = normalize(lightPos - worldPos);
    vec3 norm = normalize(normal);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}