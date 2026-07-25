#version 450

#extension GL_GOOGLE_include_directive : require
//#include "input_structures.glsl"

layout (location = 0) flat in uint vis;
layout (location = 1) flat in uint mode;

layout (location = 0) out vec4 outFragColor;


void main() 
{
	if(mode == 0u)
	{
		outFragColor = vis > 0u ? vec4(0.0f, 1.0f , 0.0f , 1.0f) : vec4(1.0f, 0.0f , 0.0f , 1.0f);
	}else{
		outFragColor = vec4(1.0f, 0.0f , 1.0f , 1.0f);
	}

}