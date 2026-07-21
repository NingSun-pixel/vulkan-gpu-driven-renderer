#version 450

#extension GL_GOOGLE_include_directive : require
//#include "input_structures.glsl"

layout (location = 0) flat in uint vis;


layout (location = 0) out vec4 outFragColor;


void main() 
{

	if(vis > 0u)
	{
		outFragColor = vec4(0.0f, 1.0f , 0.0f , 1.0f);

	}else{
		outFragColor = vec4(1.0f, 0.0f , 0.0f , 1.0f);
	}
}