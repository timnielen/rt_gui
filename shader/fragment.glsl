#version 330 core
out vec4 FragColor;

const int MAX_TEXTURES = 1;

struct TextureStack {
    vec3 baseColor;
    sampler2D textures[MAX_TEXTURES];
    float texBlend[MAX_TEXTURES];
    int texCount;
};

struct Material {
    TextureStack diffuse;
    TextureStack normal;
    TextureStack specular;
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

vec3 evaluateTextureStack(TextureStack stack) {
    if(stack.texCount == 0) return stack.baseColor;
    vec3 color = vec3(0); 
    for (int i = 0; i < stack.texCount; i++)
        color += stack.texBlend[i] * texture(stack.textures[i], TexCoords).rgb; 
    return color;
}

//vec3 strengthPointLight(PointLight light, vec3 normal, vec3 viewDir) {
//
//    vec3 lightDir = normalize(WorldPos - light.position);
//
//    float diff = max(dot(normal, -lightDir), 0.0);
//
//    vec3 reflectDir = reflect(lightDir, normal);
//
//    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
//
//    vec3 ambient = light.ambient * vec3(texture(material.texture_diffuse1, TexCoords));
//    vec3 diffuse = light.diffuse * diff * vec3(texture(material.texture_diffuse1, TexCoords));
//    vec3 specular = light.specular * spec * vec3(texture(material.texture_specular1, TexCoords));
//    return ambient + diffuse + specular;
//}

vec4 strengthDirLight(DirLight light, vec3 normal, vec3 viewDir) {

    vec3 lightDir = normalize(light.direction);

    float diff = max(dot(normal, -lightDir), 0.0);

    vec3 reflectDir = reflect(lightDir, normal);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    
    vec4 ambient = vec4(light.ambient * evaluateTextureStack(material.diffuse), 1);
    vec4 diffuse = vec4(light.diffuse * diff * evaluateTextureStack(material.diffuse),1);
    vec4 specular = vec4(light.specular * spec * evaluateTextureStack(material.specular),1);
    return ambient + diffuse + specular;
}

void main()
{
    vec3 normal = Normal;
    if (material.normal.texCount > 0) {
        normal = evaluateTextureStack(material.normal);
        normal = 2 * normal - 1;
        normal = normalize(TBN * normal);
    }
    vec3 viewDir = normalize(viewPos - WorldPos);

    FragColor = strengthDirLight(dLight, normal, viewDir); //strengthPointLight(pLight, norm, viewDir) + 
}