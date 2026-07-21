#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

//#include "input_structures.glsl"
layout (location = 0) flat out uint vis;

struct IndirectCmd {
    uint indexCount, instanceCount, firstIndex;
    int vertexOffset; 
    uint firstInstance; 
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	vec4 position[];
};


layout(push_constant) uniform Push { mat4 viewproj; VertexBuffer vertexBuffer; } pc;

layout(set=0, binding=0, std430) readonly buffer CmdBuffer { IndirectCmd commands[]; };

void main() 
{
	uint obj = uint(gl_VertexIndex) / 24u;
	vis = commands[obj].instanceCount;
	gl_Position = pc.viewproj * pc.vertexBuffer.position[gl_VertexIndex];
}