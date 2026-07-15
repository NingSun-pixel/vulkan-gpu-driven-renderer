//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_pipelines.h>

#include <chrono>
#include <thread>

#include <glm/gtx/transform.hpp>

//bootstrap library
#include "VkBootstrap.h"
#include "vk_images.h"

#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include <iostream>
#include <set>

static VKAPI_ATTR VkBool32 VKAPI_CALL my_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* userData)
{
    if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        return VK_FALSE;

    const char* tag = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        ? "\033[91m[ERROR]\033[0m"
        : "\033[93m[WARN ]\033[0m";

    std::cerr << tag << " "
        << (data->pMessageIdName ? data->pMessageIdName : "")
        << "\n  " << data->pMessage << "\n\n";

    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
#ifdef _WIN32
        //__debugbreak();
#endif
    }

    return VK_FALSE;
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    //make the vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Example Vulkan Application")
        .request_validation_layers(bUseValidationLayers)
        .set_debug_callback(my_debug_callback)        // ← 改这里！
        .set_debug_messenger_severity(
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
        )                                              // ← 顺便加这个，让 vkb 也帮你过滤
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    //grab the instance 
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    //vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features.dynamicRendering = true;
    features.synchronization2 = true;

    //vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    // 在 selector 之前加：
    VkPhysicalDeviceFeatures features10{};
    features10.multiDrawIndirect = VK_TRUE;
    features10.drawIndirectFirstInstance = VK_TRUE;   // ★ 你 indirect 命令用了 firstInstance!=0，这个也得开

    //use vkbootstrap to select a gpu. 
    //We want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_required_features(features10)
        .set_surface(_surface)
        .select()
        .value();


    //create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
        });

}

void VulkanEngine::init_commands()
{
    //create a command pool for commands submitted to the graphics queue.
    //we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++) {

        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));


        VkQueryPoolCreateInfo qpInfo{};
        qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;                       // 两个槽：geometry 开始 / 结束
        VK_CHECK(vkCreateQueryPool(_device, &qpInfo, nullptr, &_frames[i]._timestampPool));
    }


    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));



    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
        });

}

void VulkanEngine::init_sync_structures()
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        //gpu open for cpu, tell him gpu has finished
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        //GPU wait for Swapchain to provide an available image
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
    }

    _renderSemaphores.resize(_swapchainImages.size());
    //VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (auto& s : _renderSemaphores)
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &s));


    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() { vkDestroyFence(_device, _immFence, nullptr); });

}

#pragma region swapchain

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_device,_surface };

    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        //use vsync present mode
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    //store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    //draw image size will match the window
    VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    //hardcoding the draw format to 32 bit float
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    VkImageUsageFlags depthImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    //for the draw image, we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);


    //allocate and create the image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    //allocate and create the image
    vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);


    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));
    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));
    //add to deletion queues
    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
        });
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    // destroy swapchain resources
    for (int i = 0; i < _swapchainImageViews.size(); i++) {

        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}

#pragma endregion