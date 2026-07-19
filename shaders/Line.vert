#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

//#include "input_structures.glsl"


layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	vec4 position[];
};


layout(push_constant) uniform Push { mat4 viewproj; VertexBuffer vertexBuffer; } pc;

void main() 
{
	gl_Position = pc.viewproj * pc.vertexBuffer.position[gl_VertexIndex];
}