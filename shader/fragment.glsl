#version 330 core
out vec4 FragColor;

in vec2 texCoords;
in vec3 normal;
in vec3 worldPos;

uniform vec3 viewPos;

struct Material {
    sampler2D texture_diffuse1;
    sampler2D texture_specular1;
    float shininess;
};

uniform Material material;

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform PointLight pLight;

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform DirLight dLight;


vec3 strengthPointLight(PointLight light, vec3 normal, vec3 viewDir) {

    vec3 lightDir = normalize(worldPos - light.position);

    float diff = max(dot(normal, -lightDir), 0.0);

    vec3 reflectDir = reflect(lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

    vec3 ambient = light.ambient * vec3(texture(material.texture_diffuse1, texCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.texture_diffuse1, texCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.texture_specular1, texCoords));
    return ambient + diffuse + specular;
}

vec4 strengthDirLight(DirLight light, vec3 normal, vec3 viewDir) {

    vec3 lightDir = normalize(light.direction);

    float diff = max(dot(normal, -lightDir), 0.0);

    vec3 reflectDir = reflect(lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

    vec4 ambient = vec4(light.ambient, 1) * texture(material.texture_diffuse1, texCoords);
    vec4 diffuse = vec4(light.diffuse, 1) * diff * texture(material.texture_diffuse1, texCoords);
    vec4 specular = vec4(light.specular, 1) * spec * texture(material.texture_specular1, texCoords);
    return ambient + diffuse + specular;
}

void main()
{
    vec3 norm = normalize(normal);
    vec3 viewDir = normalize(viewPos - worldPos);

    FragColor = strengthDirLight(dLight, norm, viewDir); //strengthPointLight(pLight, norm, viewDir) + 
}