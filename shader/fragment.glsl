#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D texture_diffuse1;
    sampler2D texture_specular1;
    sampler2D texture_normal1;
    bool sampleDiffuse;
    bool sampleSpecular;
    bool sampleNormal;
    float shininess;
};

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};


struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec2 TexCoords;
in vec3 Normal;
in vec3 WorldPos;
in mat3 TBN;

uniform vec3 viewPos;
uniform Material material;
uniform PointLight pLight;
uniform DirLight dLight;

vec4 defaultDiffuse = vec4(1, 0.1, 0.5, 1);
vec4 defaultSpecular = vec4(0.1, 0.1, 0.1, 1);

vec3 strengthPointLight(PointLight light, vec3 normal, vec3 viewDir) {

    vec3 lightDir = normalize(WorldPos - light.position);

    float diff = max(dot(normal, -lightDir), 0.0);

    vec3 reflectDir = reflect(lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

    vec3 ambient = light.ambient * vec3(texture(material.texture_diffuse1, TexCoords));
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.texture_diffuse1, TexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.texture_specular1, TexCoords));
    return ambient + diffuse + specular;
}

vec4 strengthDirLight(DirLight light, vec3 normal, vec3 viewDir) {

    vec3 lightDir = normalize(light.direction);

    float diff = max(dot(normal, -lightDir), 0.0);

    vec3 reflectDir = reflect(lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    vec4 matDiffuse = defaultDiffuse;
    vec4 matSpecular = defaultSpecular;
    if (material.sampleDiffuse)
        matDiffuse = texture(material.texture_diffuse1, TexCoords);
    if (material.sampleSpecular)
        matSpecular = texture(material.texture_specular1, TexCoords);
    vec4 ambient = vec4(light.ambient, 1) * matDiffuse;
    vec4 diffuse = vec4(light.diffuse, 1) * diff * matDiffuse;
    vec4 specular = vec4(light.specular, 1) * spec * matSpecular;
    return ambient + diffuse + specular;
}

void main()
{
    vec3 normal = Normal;
    if (material.sampleNormal) {
        normal = texture(material.texture_normal1, TexCoords).rgb;
        normal = 2 * normal - 1;
        normal = normalize(TBN * normal);
    }

    vec3 viewDir = normalize(viewPos - WorldPos);

    FragColor = strengthDirLight(dLight, normal, viewDir); //strengthPointLight(pLight, norm, viewDir) + 
}