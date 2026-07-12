
#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil {
	//void vkutil::Barrier_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout,VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask)
	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
	void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize);
};