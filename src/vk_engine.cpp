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


void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    


    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(_chosenGPU, &props);
    _timestampPeriod = props.limits.timestampPeriod;







    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(_chosenGPU, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(_chosenGPU, &n, fams.data());
    _gpuTimingEnabled = fams[_graphicsQueueFamily].timestampValidBits > 0;

    init_imgui();


    init_default_data();

    // everything went fine
    _isInitialized = true;

    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(30.f, -00.f, -085.f);

    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    std::string structurePath = std::string(PROJECT_ROOT) + "/assets/structure.glb";
    auto structureFile = loadGltf(this, structurePath);

    assert(structureFile.has_value());

    loadedScenes["structure"] = *structureFile;

    std::vector<std::shared_ptr<MeshAsset>> allMeshes;
    for (auto& [name, mesh] : structureFile->get()->meshes)
    {
        allMeshes.push_back(mesh);
    }
    build_mega_index_buffer(allMeshes);
}

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

        uint64_t prims = 0;
        vkGetQueryPoolResults(_device, get_current_frame()._pipelineStatsPool, 0, 1,
            sizeof(prims), &prims, sizeof(prims), VK_QUERY_RESULT_64_BIT);
        stats.triangle_count_GPU = (int)prims;   // ★ 真·剔除后

        if (_benchmarking && _camMode == CamMode::Playing)
            _bench.push_back({ stats.gpu_ms_geometry,
                               stats.draw_init_cpu + stats.mesh_draw_time_CPU,   // CPU:建命令 + 录制
                               stats.triangle_count_GPU, stats.drawcall_count });
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
    vkCmdResetQueryPool(cmd, get_current_frame()._pipelineStatsPool, 0, 1);

    //make the swapchain image into writeable mode before rendering
    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);



    //Switch CPU muti thread demo or Cull demo
    switch (CurrentDemo)
    {
    case Demo::CPUMutiThread:
    {
        const uint32_t W = _drawImage.imageExtent.width;
        const uint32_t H = _drawImage.imageExtent.height;
        std::vector<glm::vec4> color(H * W);
        //Ray tracing
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                color[W * i + j] = glm::vec4(0, 1, 1, 1);
            }
        }

        // ---- float -> RGBA16F(half),匹配 _drawImage 格式 ----
        // 每像素 2 个 uint32:低32=RG,高32=BA
        std::vector<glm::uint> packed(W * H * 2);
        for (size_t p = 0; p < color.size(); p++) {
            packed[p * 2 + 0] = glm::packHalf2x16(glm::vec2(color[p].x, color[p].y));
            packed[p * 2 + 1] = glm::packHalf2x16(glm::vec2(color[p].z, color[p].w));
        }
        VkDeviceSize bytes = packed.size() * sizeof(glm::uint);

        AllocatedBuffer staging = create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        memcpy(staging.info.pMappedData, packed.data(), bytes);
        get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(staging); });

        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copyRegion = {};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;

        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = VkExtent3D{ _windowExtent.width ,_windowExtent.height ,1};

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, staging.buffer, _drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            &copyRegion);

        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }
        break;
    case Demo::Cull:
    {
        //Render
        draw_background(cmd);
        draw_init();
        if (_benchConfig != 3) draw_Cull(cmd);   // naive:不走 compute(无 indirect、无剔除)

        VkMemoryBarrier2 memBarrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;          // ← memory 组,不是 buffer 组
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        //make the swapchain image into presentable mode
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        draw_geometry(cmd);
    }
        break;
    default:
    {
        vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }
        break;
    }

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
            if (_camMode != CamMode::Playing) mainCamera.processSDLEvent(e);   // 仅第一人称回放屏蔽输入(观察态可动)
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

        const ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(
            ImVec2(io.DisplaySize.x - 10, 10),   // x=屏幕右边缘往里 10,y=顶部往下 10
            ImGuiCond_Always,
            ImVec2(1.0f, 0.0f));                  // pivot=(1,0) → 坐标指的是窗口右上角
        
        ImGui::Begin("Stats");

        ImGui::Text("frametime %f ms", stats.frametime_CPU);
        ImGui::Text("draw time %f ms", stats.mesh_draw_time_CPU);
        ImGui::Text("update time %f ms", stats.scene_update_time_CPU);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);
        ImGui::Text("triangles_GPU %i", stats.triangle_count_GPU);

        stats.gpu_ms_history[stats.gpu_ms_offset] = stats.gpu_ms_geometry; // 当前值写到 offset 位置
        stats.gpu_ms_offset = (stats.gpu_ms_offset + 1) % 120;            // 指针前进一格，到 120 绕回 0

        char overlay[32];
        snprintf(overlay, sizeof(overlay), "gpu %.2f ms", stats.gpu_ms_geometry);
        ImGui::PlotLines("##gpu", stats.gpu_ms_history, 120, stats.gpu_ms_offset,
            overlay, 15.0f, 60.0f, ImVec2(0, 60));

        // CPU 折线(建命令 + 录制)
        float cpuNow = stats.draw_init_cpu + stats.mesh_draw_time_CPU;
        stats.cpu_ms_history[stats.cpu_ms_offset] = cpuNow;
        stats.cpu_ms_offset = (stats.cpu_ms_offset + 1) % 120;
        char cpuOverlay[40];
        snprintf(cpuOverlay, sizeof(cpuOverlay), "cpu %.2f ms (init %.2f)", cpuNow, stats.draw_init_cpu);
        ImGui::PlotLines("##cpu", stats.cpu_ms_history, 120, stats.cpu_ms_offset,
            cpuOverlay, 0.0f, 10.0f, ImVec2(0, 60));

        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);   // (10,10)=离左上角留点边距
        if (ImGui::Begin("Settings")) {
            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.f);
            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];
            const char* _CurrentDemo[] = { "MutiThreadCPU", "CullRender" };
            ImGui::Combo("CurrentDemo", &CurrentDemo, _CurrentDemo, IM_ARRAYSIZE(_CurrentDemo));

            if (CurrentDemo == Demo::Cull)
            {
                const char* cfgNames[] = { "baseline-noinstance", "baseline-instance", "instance-gpucull", "naive" };
                ImGui::Combo("Bench Cfg", &_benchConfig, cfgNames, IM_ARRAYSIZE(cfgNames));

                ImGui::SeparatorText("Culling");
                ImGui::Checkbox("Freeze Cull Frustum", &_freezeCull);
                ImGui::Checkbox("ShowAABB", &_ShowAABB);
            }
            if (ImGui::CollapsingHeader("Background Effect")) {
                ImGui::Text("Selected: %s", selected.name);
                ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0, backgroundEffects.size() - 1);
                ImGui::InputFloat4("data1", (float*)&selected.data.data1);
                ImGui::InputFloat4("data2", (float*)&selected.data.data2);
                ImGui::InputFloat4("data3", (float*)&selected.data.data3);
                ImGui::InputFloat4("data4", (float*)&selected.data.data4);
            }
        }
        ImGui::End();

        if (ImGui::Begin("Camera Path")) {
            if (ImGui::Button("Capture")) {                              // 抓当前相机
                _pathPoints.push_back({ mainCamera.position, mainCamera.yaw, mainCamera.pitch });
            }
            ImGui::SameLine();
            if (ImGui::Button("Delete Last")) {                         // 弹最后一个
                if (!_pathPoints.empty()) _pathPoints.pop_back();
            }

            ImGui::Text("points: %d", (int)_pathPoints.size());
            for (int i = 0; i < (int)_pathPoints.size(); i++) {          // 列出来,验证用
                const auto& p = _pathPoints[i];
                ImGui::Text("[%d] (%.1f, %.1f, %.1f)  yaw %.2f  pitch %.2f",
                    i, p.pos.x, p.pos.y, p.pos.z, p.yaw, p.pitch);
            }
            if (ImGui::Button("Save")) savePath("../../paths/station.json");
            ImGui::SameLine();
            if (ImGui::Button("Load")) loadPath("../../paths/station.json");

            ImGui::SeparatorText("Playback");
            bool busy = (_camMode != CamMode::Free);

            if (busy) ImGui::BeginDisabled();                            // 运行中锁定速度
            ImGui::SliderFloat("Speed", &_playSpeed, 0.5f, 50.f, "%.1f u/s");
            if (busy) ImGui::EndDisabled();

            if (_camMode == CamMode::Free) {
                if (ImGui::Button("Play") && _pathPoints.size() >= 2) {  // 第一人称飞路径
                    _playSpeedRun = _playSpeed;   // ★ 读一次速度,锁定本次回放
                    _playDist = 0.f;
                    _lutDirty = true;             // 点可能变过,重建弧长表
                    _camMode = CamMode::Playing;
                    _bench.clear();
                    _benchmarking = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Observe Cull") && _pathPoints.size() >= 2) {  // 第三视角观察裁切
                    _playSpeedRun = _playSpeed;
                    _playDist = 0.f;
                    _lutDirty = true;
                    _freezeCull = true;           // 冻结:让 cull 不再跟主相机,交给路径驱动
                    _camMode = CamMode::ObserveCull;
                }
            } else {
                if (ImGui::Button("Stop"))
                {
                    _camMode = CamMode::Free;     // stop in the process
                    _benchmarking = false;
                }
            }
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


void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex + mesh->baseIndex;
        def.indexBuffer = loadedEngine->_megaIndexBuffer.buffer;
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


glm::vec4 VulkanEngine::Ray::RayFunction(glm::vec4 ori, glm::vec4 dir)
{
    glm::vec4 Des = ori + dir;
    return Des;
}