#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inWorldPos;

layout (location = 0) out vec4 outFragColor;

const float PI = 3.14159265359;

// --- GGX 法线分布:表面微平面朝向 H 的集中度,roughness 越小越尖(高光越锐) ---
float D_GGX(float NdotH, float rough) {
    float a  = rough * rough;          // 迪士尼惯例:感知粗糙度平方
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// --- Smith 几何遮蔽:微平面互相挡光/挡视线的损失,直接光用 k=(r+1)^2/8 ---
float G_SchlickGGX(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}
float G_Smith(float NdotV, float NdotL, float rough) {
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    return G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
}

// --- Fresnel-Schlick:掠射角反射增强(边缘更亮) ---
vec3 F_Schlick(float HdotV, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);
}

void main()
{
	// float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);
	
	// vec3 color = inColor * texture(colorTex,inUV).xyz;
	// vec3 ambient = color * sceneData.ambientColor.xyz;

	// outFragColor = vec4(color * lightValue *  sceneData.sunlightColor.w + ambient ,1.0f);
    // ---- 组装 PBR 输入 ----
    vec3 N = normalize(inNormal);                              // ★ 修:插值后必须归一化
    vec3 V = normalize(sceneData.cameraPos.xyz - inWorldPos);  // 视线
    vec3 L = normalize(sceneData.sunlightDirection.xyz);       // 指向太阳(和你 Lambert 同约定)
    vec3 H = normalize(V + L);

    vec3  albedo    = inColor * texture(colorTex, inUV).rgb;
    vec3  mr        = texture(metalRoughTex, inUV).rgb;              // structure.glb 恒 (1,1,1)
    float metallic  = 0.6;//materialData.metal_rough_factors.x * mr.b;    // 该场景 = 0
    float roughness = clamp(materialData.metal_rough_factors.y * mr.g, 0.05, 1.0); // 该场景 = 0.5

    float NdotL = max(dot(N, L), 0.0); 
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // ---- Cook-Torrance 镜面项 ----
    vec3  F0 = mix(vec3(0.04), albedo, metallic);   // 非金属基础反射 4%,金属用 albedo 染色
    float D  = D_GGX(NdotH, roughness);
    float G  = G_Smith(NdotV, NdotL, roughness);
    vec3  F  = F_Schlick(HdotV, F0);

    vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    // ---- 能量守恒:反射掉的不再漫反射;金属没有漫反射 ----
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    // ---- 直接光(方向光)----
    vec3 radiance = sceneData.sunlightColor.rgb * sceneData.sunlightColor.w;  // w 当强度
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // ---- 环境项(替换掉原来的 max(...,0.1) 假光 hack)----
    vec3 ambient = albedo * sceneData.ambientColor.rgb * 5.0;

    vec3 color = ambient + Lo;

    // ---- 色调映射 + gamma(贴图按 UNORM 加载时需要;若过暗/过亮就注释掉这两行验一下)----
    //color = color / (color + vec3(1.0));      // Reinhard
    //color = pow(color, vec3(1.0 / 2.2));      // 线性 → sRGB

    outFragColor = vec4(color, 1.0);
}