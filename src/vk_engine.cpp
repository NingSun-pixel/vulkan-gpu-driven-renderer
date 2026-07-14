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

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include <iostream>
#include <set>


VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }



bool is_visible(const RenderObject& obj, const glm::mat4& viewproj) {
    std::array<glm::vec3, 8> corners{
        glm::vec3 { 1, 1, 1 },
        glm::vec3 { 1, 1, -1 },
        glm::vec3 { 1, -1, 1 },
        glm::vec3 { 1, -1, -1 },
        glm::vec3 { -1, 1, 1 },
        glm::vec3 { -1, 1, -1 },
        glm::vec3 { -1, -1, 1 },
        glm::vec3 { -1, -1, -1 },
    };

    glm::mat4 matrix = viewproj * obj.transform;

    glm::vec3 min = { 1.5, 1.5, 1.5 };
    glm::vec3 max = { -1.5, -1.5, -1.5 };

    for (int c = 0; c < 8; c++) {
        // project each corner into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        if (v.w <= 0)
        {
            return true;
        }
        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        min = glm::min(glm::vec3{ v.x, v.y, v.z }, min);
        max = glm::max(glm::vec3{ v.x, v.y, v.z }, max);
    }

    // check the clip space box is within the view
    if (min.z > 1.f || max.z < 0.f || min.x > 1.f || max.x < -1.f || min.y > 1.f || max.y < -1.f) {
        return false;
    }
    else {
        return true;
    }
}


//open a space to store the data,VMA = Vulkan Memory Allocator
AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
        &newBuffer.info));

    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

//从引用数组和顶点数组里还原出mesh
GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    //create vertex buffer
    newSurface.vertexBuffer = create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    //find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer = newSurface.vertexBuffer.buffer };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    //create index buffer
    newSurface.indexBuffer = create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
        });

    destroy_buffer(staging);

    return newSurface;
}



void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    init_Cull_pipelines();
    metalRoughMaterial.build_pipelines(this);
}

//skybox
void VulkanEngine::init_background_pipelines()
{
    //pipeline layout code
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader)) {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader)) {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = gradientShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    //default colors
    gradient.data.data1 = glm::vec4(1, 0, 0, 1);
    gradient.data.data2 = glm::vec4(0, 0, 1, 1);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    //change the shader module only to create the sky shader
    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    //default sky parameters
    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    //add the 2 background effects into the array
    backgroundEffects.push_back(sky);
    backgroundEffects.push_back(gradient);

    //destroy structures properly
    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
        });
}

//Cull Compute shader
void VulkanEngine::init_Cull_pipelines()
{
    VkDescriptorSetLayout layouts[] = {_objectDataDescriptorLayout, _CommandDataDescriptorLayout};

    VkPipelineLayoutCreateInfo cullObjLayout{};
    cullObjLayout.setLayoutCount = 2;
    cullObjLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cullObjLayout.pNext = nullptr;
    cullObjLayout.pSetLayouts = layouts;

    //?
    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    cullObjLayout.pPushConstantRanges = &pushConstant;
    cullObjLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &cullObjLayout, nullptr, &_cullPipelineLayout));


    VkShaderModule cullComputeShader;
    if (!vkutil::load_shader_module("../../shaders/Cull.comp.spv", _device, &cullComputeShader)) {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = cullComputeShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _cullPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    //ComputeEffect cullObj;
    CullpipelineEffects.layout = _cullPipelineLayout;
    CullpipelineEffects.name = "Cull";
    CullpipelineEffects.data = {};


    CullpipelineEffects = CullpipelineEffects;
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &CullpipelineEffects.pipeline));


    //destroy structures properly
    vkDestroyShaderModule(_device, cullComputeShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipelineLayout(_device, _cullPipelineLayout, nullptr);
        vkDestroyPipeline(_device, CullpipelineEffects.pipeline, nullptr);
        });
}

//cull compute shader draw
//对改造后的GPUObject数据补全，vkCmdPushConstants(viewproj, count)的补全，链接上command，还有bind建立
void VulkanEngine::draw_Cull(VkCommandBuffer cmd)
{
    ComputeEffect& effect = CullpipelineEffects;

    // bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _cullPipelineLayout, 0, 1, &objectDescriptor, 0, nullptr);
    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _cullPipelineLayout, 1, 1, &commandDescriptor, 0, nullptr);

    //差这里的data
    CullPush pc{};
    pc.viewproj = sceneData.viewproj;
    pc.count = (uint32_t)opaque_draws.size();   // ★ 用 uint，不是塞进 float
    vkCmdPushConstants(cmd, _cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPush), &pc);
    vkCmdDispatch(cmd, (pc.count + 63) / 64, 1, 1); // 整数取整，别用 float

}


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

void VulkanEngine::cleanup()
{
    if (_isInitialized) {

        //make sure the gpu has stopped doing its things
        vkDeviceWaitIdle(_device);

        loadedScenes.clear();

        for (int i = 0; i < FRAME_OVERLAP; i++) {

            //already written from before
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

            //destroy sync objects
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
        }

        for (auto s : _renderSemaphores) vkDestroySemaphore(_device, s, nullptr);
        _renderSemaphores.clear();

        for (auto& mesh : testMeshes) {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }
        //flush the global deletion queue
        _mainDeletionQueue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    update_scene();
    // wait until the gpu has finished rendering the last frame. Timeout of 1
    // second
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));

    if (_frameNumber >= FRAME_OVERLAP) {
        uint64_t ts[2];
        vkGetQueryPoolResults(_device, get_current_frame()._timestampPool, 0, 2, sizeof(ts), ts,
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        stats.gpu_ms_geometry = (ts[1] - ts[0]) * _timestampPeriod / 1e6f;  // ns→ms
    }

    //the second time you run this frame
    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    //request image from the swapchain
    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    //naming it cmd for shorter writing
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;


    // now that we are sure that the commands finished executing, we can safely
    // reset the command buffer to begin recording again.
    //empty it
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    //init the draw panel
    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;
    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;

    //start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    vkCmdResetQueryPool(cmd, get_current_frame()._timestampPool, 0, 2);
    //make the swapchain image into writeable mode before rendering
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    //Render
    draw_background(cmd);
    draw_init();
    draw_Cull(cmd);

    VkBufferMemoryBarrier2 bufBarrier{ .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    bufBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    bufBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    bufBarrier.buffer = indirectBuf.buffer;
    bufBarrier.offset = 0;
    bufBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.bufferMemoryBarrierCount = 1;
    depInfo.pBufferMemoryBarriers = &bufBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    //make the swapchain image into presentable mode
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    draw_geometry(cmd);
    //transtion the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    //Transfer to unreadable again
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    // set swapchain image layout to Attachment Optimal so we can draw it
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    //draw imgui into the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    // set swapchain image layout to Present so we can draw it
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    //finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    //prepare the submission to the queue. 
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, _renderSemaphores[swapchainImageIndex]);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

    //submit command buffer to the queue and execute it.
    // _render Fence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that, 
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &_renderSemaphores[swapchainImageIndex];
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;
    
    draw_clear();

    //VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));
    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }
    //increase the number of frames drawn
    _frameNumber++;

}

//skybox
void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    // bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);
    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw_init()
{
    opaque_draws.reserve(mainDrawContext.OpaqueSurfaces.size());


    for (int i = 0; i < mainDrawContext.OpaqueSurfaces.size(); i++) {
        opaque_draws.push_back(i);
    }

    if (opaque_draws.empty()) {
        return;
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaque_draws.begin(), opaque_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = mainDrawContext.OpaqueSurfaces[iA];
        const RenderObject& B = mainDrawContext.OpaqueSurfaces[iB];
        if (A.material == B.material) {
            if (A.indexBuffer == B.indexBuffer)
            {
                A.firstIndex < B.firstIndex;
            }
            return A.indexBuffer < B.indexBuffer;
        }
        else {
            return A.material < B.material;
        }
        });

    std::vector<GPUObjectData> objects;

    objects.reserve(mainDrawContext.OpaqueSurfaces.size() + mainDrawContext.TransparentSurfaces.size());
    for (auto& s : opaque_draws)
        objects.push_back({ mainDrawContext.OpaqueSurfaces[s].transform,
            mainDrawContext.OpaqueSurfaces[s].vertexBufferAddress,
            0,
            mainDrawContext.OpaqueSurfaces[s].bounds.origin,
            mainDrawContext.OpaqueSurfaces[s].bounds.sphereRadius,
            glm::vec4(mainDrawContext.OpaqueSurfaces[s].bounds.extents,1)});
    for (auto& s : mainDrawContext.TransparentSurfaces) objects.push_back({ s.transform, s.vertexBufferAddress });

    // 4b. 建 SSBO
    AllocatedBuffer objectBuffer = create_buffer(objects.size() * sizeof(GPUObjectData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(objectBuffer); });

    name_buffer(objectBuffer.buffer, "objectBuffer");
    // 4c. 填数据
    memcpy(objectBuffer.allocation->GetMappedData(), objects.data(), objects.size() * sizeof(GPUObjectData));

    // 4d. 分配 + 写描述符（set 2）
    objectDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _objectDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, objectBuffer.buffer, objects.size() * sizeof(GPUObjectData), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, objectDescriptor);
    }


    //在init里拿到indirectbuf传递给Culldraw
    //既要和之前的比，又要和之后的比
    //和之前的比，是为了少切换pipeline和少重新赋值
    //和之后的比，是为了合批进行自动instance
    const RenderObject* rep = nullptr;     // 当前批的代表物体（取 indexCount/firstIndex/material 等）
    uint32_t batchStart = 0, instanceCount = 1;//firstindex = batchStart,指名当前是取GPUObject里哪个数据
    std::vector<VkDrawIndexedIndirectCommand> commands;//命令合批3.2

    //instanceCount 

    auto sameBatch = [](const RenderObject& a, const RenderObject& b) {
        return a.material == b.material && a.indexBuffer == b.indexBuffer
            && a.firstIndex == b.firstIndex && a.indexCount == b.indexCount;   // 同 surface
        };



    auto flush = [&]() {
        if (instanceCount == 0) return;
        const RenderObject& r = *rep;

        if (groups.empty() || r.material != groups.back().material)
            groups.push_back({ r.material, (uint32_t)commands.size(), (uint32_t)instanceCount });  // ← offset/count 都错
        else
            groups.back().cmdCount += instanceCount;


        for (int i = 0; i < instanceCount; i++)
        {
            commands.push_back({
            .indexCount = r.indexCount,
            .instanceCount = 1,   // 批大小，和阶段2 一样
            .firstIndex = r.firstIndex,    // 已经是 mega 相对(baseIndex+startIndex)，3.1 做好的
            .vertexOffset = 0,               // per-mesh BDA，不 rebase
            .firstInstance = batchStart + (uint32_t)i,      // 这批在 objects[] 里的起点
                });
            //add counters for triangles and draws
            stats.drawcall_count++;
            stats.triangle_count += r.indexCount / 3;
        }

        };

    rep = &mainDrawContext.OpaqueSurfaces[opaque_draws[0]];
    for (size_t j = 0; j < opaque_draws.size() - 1; j++) {
        const RenderObject& r = mainDrawContext.OpaqueSurfaces[opaque_draws[j + 1]];
        if (instanceCount > 0 && sameBatch(*rep, r)) {
            instanceCount++;                       // 同批，数量+1
        }
        else {
            flush();                            // 先画掉上一批
            rep = &r; batchStart = j + 1; instanceCount = 1;   // 开新批
        }
    }

    flush();   // ★ 别忘了最后一批

    size_t bytes = (commands.size()) * sizeof(VkDrawIndexedIndirectCommand);
    indirectBuf = create_buffer(bytes,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,        // STORAGE 是给阶段4 compute 写用，现在加上无妨
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    memcpy(indirectBuf.allocation->GetMappedData(), commands.data(), bytes);
    // 每帧建的话记得丢进 get_current_frame()._deletionQueue
    name_buffer(indirectBuf.buffer, "indirectBuf");

    AllocatedBuffer indirectToFree = indirectBuf;   // 本帧这块的快照(局部)
    get_current_frame()._deletionQueue.push_function(
        [=, this]() { destroy_buffer(indirectToFree); });   // 删快照,不删 this->indirectBuf


    //get_current_frame()._deletionQueue.push_function([=, this]() {
    //    destroy_buffer(indirectBuf);
    //    });

    // 4d. 分配 + 写描述符（set 2）
    commandDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _CommandDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, indirectBuf.buffer, bytes, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, commandDescriptor);
    }
}

void VulkanEngine::draw_clear()
{
    opaque_draws.clear();
    groups.clear();
}



void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    if (opaque_draws.empty()) {
        return;
    }

    //reset counters
    stats.drawcall_count = 0;
    stats.triangle_count = 0;
    //begin clock
    auto start = std::chrono::system_clock::now();
    //begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, &depthAttachment);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, get_current_frame()._timestampPool, 0);

    vkCmdBeginRendering(cmd, &renderInfo);

    //set dynamic viewport and scissor
    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = _drawExtent.width;
    viewport.height = _drawExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _drawExtent.width;
    scissor.extent.height = _drawExtent.height;

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    //allocate a new uniform buffer for the scene data
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //add it to the deletion queue of this frame so it gets deleted once its been used
    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(gpuSceneDataBuffer);
        });

    //write the buffer
    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = sceneData;

    //create a descriptor set that binds that buffer and update it
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, globalDescriptor);

    //defined outside of the draw function, this is the state we will try to skip
    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;
    uint32_t lastFirstIndex = -1;




    vkCmdBindIndexBuffer(cmd, _megaIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);  // 整场景只绑一次
    for (auto& g : groups) {
        // —— bind material（保留你的 lastMaterial/lastPipeline 跳过逻辑）——
        if (g.material != lastMaterial) {
            lastMaterial = g.material;
            if (g.material->pipeline != lastPipeline) {
                lastPipeline = g.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.material->pipeline->layout, 2, 1, &objectDescriptor, 0, nullptr);
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.material->pipeline->layout, 1, 1, &g.material->materialSet, 0, nullptr);
        }
        //bind(g.material 的 pipeline + set0 / set1 / set2);   // 沿用你的 lastMaterial/lastPipeline 跳过逻辑
        vkCmdDrawIndexedIndirect(cmd, indirectBuf.buffer,
            g.cmdOffset * sizeof(VkDrawIndexedIndirectCommand),  // offset
            g.cmdCount,                                          // drawCount = 这组命令条数，改成非instance之后，命令条数=instance数目
            sizeof(VkDrawIndexedIndirectCommand));               // stride
    }



    auto drawTranparent = [&](const RenderObject& r, uint32_t objectIndex){
        if (r.material != lastMaterial) {
            lastMaterial = r.material;
            //rebind pipeline and descriptors if the material changed
            if (r.material->pipeline != lastPipeline) {
                lastPipeline = r.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 2, 1, &objectDescriptor, 0, nullptr);
            }
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1, &r.material->materialSet, 0, nullptr);
        }
        if (r.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = r.indexBuffer;
            vkCmdBindIndexBuffer(cmd, r.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }
        vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, objectIndex);

        //add counters for triangles and draws
        stats.drawcall_count++;
        stats.triangle_count += r.indexCount / 3;
        };


    for (size_t j = 0; j < mainDrawContext.TransparentSurfaces.size(); j++)
        drawTranparent(mainDrawContext.TransparentSurfaces[j],
            opaque_draws.size() + j);      // transparent：接在 opaque 后面

	vkCmdEndRendering(cmd);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, get_current_frame()._timestampPool, 1);
    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    auto end = std::chrono::system_clock::now();

    //convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time_CPU = elapsed.count() / 1000.f;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _render Fence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::init_imgui()
{
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    //dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;


    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    // add the destroy the imgui created structures
    _mainDeletionQueue.push_function([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
        });
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        //begin clock
        auto start = std::chrono::system_clock::now();

        SDL_Event e;
        bool bQuit = false;


            // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;


            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
            mainCamera.processSDLEvent(e);
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested) {
            resize_swapchain();
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", stats.frametime_CPU);
        ImGui::Text("draw time %f ms", stats.mesh_draw_time_CPU);
        ImGui::Text("update time %f ms", stats.scene_update_time_CPU);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);

        stats.gpu_ms_history[stats.gpu_ms_offset] = stats.gpu_ms_geometry; // 当前值写到 offset 位置
        stats.gpu_ms_offset = (stats.gpu_ms_offset + 1) % 120;            // 指针前进一格，到 120 绕回 0

        char overlay[32];
        snprintf(overlay, sizeof(overlay), "gpu %.2f ms", stats.gpu_ms_geometry);
        ImGui::PlotLines("##gpu", stats.gpu_ms_history, 120, stats.gpu_ms_offset,
            overlay, 15.0f, 60.0f, ImVec2(0, 60));

        ImGui::End();

        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.f);
            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0, backgroundEffects.size() - 1);

            ImGui::InputFloat4("data1", (float*)&selected.data.data1);
            ImGui::InputFloat4("data2", (float*)&selected.data.data2);
            ImGui::InputFloat4("data3", (float*)&selected.data.data3);
            ImGui::InputFloat4("data4", (float*)&selected.data.data4);
        }
        ImGui::End();

        ImGui::Render();

        //our draw function

        draw();

        //get clock again, compare with start clock
        auto end = std::chrono::system_clock::now();

        //convert to microseconds (integer), and then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime_CPU = elapsed.count() / 1000.f;
    }
}

void VulkanEngine::init_default_data() {
    std::array<Vertex, 4> rect_vertices;

    rect_vertices[0].position = { 0.5,-0.5, 0 };
    rect_vertices[1].position = { 0.5,0.5, 0 };
    rect_vertices[2].position = { -0.5,-0.5, 0 };
    rect_vertices[3].position = { -0.5,0.5, 0 };

    rect_vertices[0].color = { 0,0, 0,1 };
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1 };
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t, 6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = uploadMesh(rect_indices, rect_vertices);

    //delete the rectangle data on engine shutdown
    _mainDeletionQueue.push_function([&]() {
        destroy_buffer(rectangle.indexBuffer);
        destroy_buffer(rectangle.vertexBuffer);
        });

    testMeshes = loadGltfMeshes(this, "..\\..\\assets\\basicmesh.glb").value();

    //3 default textures, white, grey, black. 1 pixel each
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = create_image((void*)&white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = create_image((void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 1));
    _blackImage = create_image((void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    //checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16 > pixels; //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : white;
        }
    }
    _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };

    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_errorCheckerboardImage);
        });
    GLTFMetallic_Roughness::MaterialResources materialResources;
    //default the material textures
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;

    //set the uniform buffer for the material data
    AllocatedBuffer materialConstants = create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //write the buffer
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData = (GLTFMetallic_Roughness::MaterialConstants*)materialConstants.allocation->GetMappedData();
    sceneUniformData->colorFactors = glm::vec4{ 1,1,1,1 };
    sceneUniformData->metal_rough_factors = glm::vec4{ 1,0.5,0,0 };

    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(materialConstants);
        });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, get_current_frame()._frameDescriptors);

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

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            &copyRegion);

        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

    destroy_buffer(uploadbuffer);

    return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

void VulkanEngine::update_scene()
{
    mainCamera.update();

    glm::mat4 view = mainCamera.getViewMatrix();

    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);

    mainDrawContext.OpaqueSurfaces.clear();

    loadedScenes["structure"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    sceneData.view = view;
    // camera projection
    sceneData.proj = projection;

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(1.f);
    sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);
  
}

void VulkanEngine::build_mega_index_buffer(std::vector<std::shared_ptr<MeshAsset>>& allMeshes) {
    //allmeshes
    uint32_t AllIndexCountBefore = 0;
    uint32_t GetTotalSizeToCreate = 0;
    for (auto M : allMeshes)
    {
        for (auto Geo : M->surfaces)
            GetTotalSizeToCreate += Geo.count;
    }
    _megaIndexBuffer = create_buffer(GetTotalSizeToCreate * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT , VMA_MEMORY_USAGE_GPU_ONLY);


    immediate_submit([&](VkCommandBuffer cmd) {
        for (auto &M : allMeshes)
        {
            uint32_t OneMeshIndexCount = 0;
            M->baseIndex = AllIndexCountBefore;

            for (auto &Geo : M->surfaces)
            {
                OneMeshIndexCount += Geo.count;
            }
            VkBufferCopy copy{};
            copy.srcOffset = 0;
            copy.dstOffset = (VkDeviceSize)AllIndexCountBefore * sizeof(uint32_t);
            copy.size = (VkDeviceSize)OneMeshIndexCount * sizeof(uint32_t);
            vkCmdCopyBuffer(cmd, M->meshBuffers.indexBuffer.buffer, _megaIndexBuffer.buffer, 1, &copy);
            AllIndexCountBefore += OneMeshIndexCount;
        }
    });

    _mainDeletionQueue.push_function([&]() { destroy_buffer(_megaIndexBuffer); });
}

void VulkanEngine::name_buffer(VkBuffer buf, const char* name) {
    static auto func = (PFN_vkSetDebugUtilsObjectNameEXT)
        vkGetDeviceProcAddr(_device, "vkSetDebugUtilsObjectNameEXT");
    if (!func) return;   // 没启用 debug_utils 就直接跳过,不崩

    VkDebugUtilsObjectNameInfoEXT info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    info.objectType = VK_OBJECT_TYPE_BUFFER;
    info.objectHandle = (uint64_t)buf;
    info.pObjectName = name;
    func(_device, &info);
}

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.frag.spv", engine->_device, &meshFragShader)) {
        fmt::println("Error when building the triangle fragment shader module");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/meshGPUDriven.vert.spv", engine->_device, &meshVertexShader)) {
        fmt::println("Error when building the triangle vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->_gpuSceneDataDescriptorLayout,
        materialLayout,engine->_objectDataDescriptorLayout };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 3;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = newLayout;

    // finally build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // create the transparent variant
    pipelineBuilder.enable_blending_additive();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    }
    else {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);


    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex + mesh->baseIndex;
        def.indexBuffer = loadedEngine->_megaIndexBuffer.buffer;//mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        if (s.material->data.passType == MaterialPass::Transparent) {
            ctx.TransparentSurfaces.push_back(def);
        }
        else {
            ctx.OpaqueSurfaces.push_back(def);
        }
    }

    // recurse down
    Node::Draw(topMatrix, ctx);
}
