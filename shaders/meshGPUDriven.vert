#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outWorldPos;

struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
}; 


layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

struct ObjectData{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
	uint batchId; uint _pad;
	vec4 boundsOriginRadius;
	vec4 boundsExtents;
};


layout(set = 2, binding = 0, std430) readonly buffer ObjectBuffer 
{
	ObjectData objects[];
}objectBuffer;

layout(set = 2, binding = 1, std430) readonly buffer CompactBuffer 
{
	uint compactInstances[];
}compactBuffer;


void main() 
{
	uint objIdx = compactBuffer.compactInstances[gl_InstanceIndex];
	ObjectData obj = objectBuffer.objects[objIdx];   // 不再读 PushConstants
	Vertex v = obj.vertexBuffer.vertices[gl_VertexIndex];        // 取顶点这行几乎不变
	vec4 worldPos = obj.render_matrix * vec4(v.position, 1.0);
	gl_Position   = sceneData.viewproj * worldPos;   // 原来那行改成用 worldPos
	outWorldPos   = worldPos.xyz;

	outNormal = (obj.render_matrix * vec4(v.normal, 0.f)).xyz;
	outColor = v.color.xyz * materialData.colorFactors.xyz;	
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;


}