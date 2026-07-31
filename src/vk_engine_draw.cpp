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
    pc.viewproj = _cullViewProj;
    pc.count = (uint32_t)opaque_draws.size();   // ★ 用 uint，不是塞进 float
    vkCmdPushConstants(cmd, _cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPush), &pc);
    vkCmdDispatch(cmd, (pc.count + 63) / 64, 1, 1); // 整数取整，别用 float

}

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
    //reset counters
    stats.drawcall_count = 0;
    stats.triangle_count = 0;

    if (!_freezeCull) _cullViewProj = sceneData.viewproj;
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
            glm::vec4(mainDrawContext.OpaqueSurfaces[s].bounds.extents,1) });
    for (auto& s : mainDrawContext.TransparentSurfaces) objects.push_back({ s.transform, s.vertexBufferAddress });

    // 4b. 建 SSBO,Obj(顶点数据)
    AllocatedBuffer objectBuffer = create_buffer(objects.size() * sizeof(GPUObjectData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(objectBuffer); });

    name_buffer(objectBuffer.buffer, "objectBuffer");
    // 4c. 填数据
    memcpy(objectBuffer.info.pMappedData, objects.data(), objects.size() * sizeof(GPUObjectData));

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
    memcpy(indirectBuf.info.pMappedData, commands.data(), bytes);
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

void VulkanEngine::draw_line(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _linePipeline);
    GPULineBuffers lineBuffer = PushVertex();
    //delete the rectangle data on engine shutdown
    get_current_frame()._deletionQueue.push_function([lineBuffer, this]() {
        destroy_buffer(lineBuffer.vertexBuffer);
        });

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _linePipelineLayout, 0, 1, &commandDescriptor, 0, nullptr);

    LinePush pc{};
    pc.viewproj = sceneData.viewproj;
    pc.vertexBufferAddress = lineBuffer.vertexBufferAddress;
    pc.mode = 0;
    vkCmdPushConstants(cmd, _linePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(LinePush), &pc);
    vkCmdDraw(cmd, (uint32_t)(mainDrawContext.OpaqueSurfaces.size()) * 24, 1, 0, 0);

    pc.mode = 1;
    vkCmdPushConstants(cmd, _linePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(LinePush), &pc);
    vkCmdDraw(cmd, (uint32_t)24, 1, (uint32_t)(mainDrawContext.OpaqueSurfaces.size()) * 24, 0);
    //vkCmdDraw(cmd, (uint32_t)24, 1, 0, 0);

}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    if (opaque_draws.empty()) {
        return;
    }


    //begin clock
    auto start = std::chrono::system_clock::now();
    //begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, &depthAttachment);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, get_current_frame()._timestampPool, 0);
    vkCmdBeginQuery(cmd, get_current_frame()._pipelineStatsPool, 0, 0);
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
    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.info.pMappedData;
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



    auto drawTranparent = [&](const RenderObject& r, uint32_t objectIndex) {
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


    if(_ShowAABB)
        draw_line(cmd);

    vkCmdEndRendering(cmd);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, get_current_frame()._timestampPool, 1);
    vkCmdEndQuery(cmd, get_current_frame()._pipelineStatsPool, 0);
    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    auto end = std::chrono::system_clock::now();

    //convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time_CPU = elapsed.count() / 1000.f;
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
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData = (GLTFMetallic_Roughness::MaterialConstants*)materialConstants.info.pMappedData;
    sceneUniformData->colorFactors = glm::vec4{ 1,1,1,1 };
    sceneUniformData->metal_rough_factors = glm::vec4{ 1,0.5,0,0 };

    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(materialConstants);
        });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, get_current_frame()._frameDescriptors);

}

void VulkanEngine::update_scene()
{
    mainCamera.update();

    glm::mat4 view = mainCamera.getViewMatrix();

    // camera projection
    float renderFar = _freezeCull ? 10000.f : 300.f; 
    glm::mat4 projection = glm::perspective(glm::radians(70.f),
        (float)_windowExtent.width / (float)_windowExtent.height, renderFar, 1.0f);
    mainDrawContext.OpaqueSurfaces.clear();

    loadedScenes["structure"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    sceneData.view = view;
    // camera projection
    sceneData.proj = projection;
    sceneData.cameraPos = glm::vec4(glm::vec3(glm::inverse(view)[3]), 1.f);
    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(1.f);
    sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);

}


GPULineBuffers VulkanEngine::PushVertex()
{
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
    int line[24] = {
    0,1,
    0,2,
    4,5,
    4,6,
    0,4,
    1,5,
    2,6,
    3,7,
    2,3,
    6,7,
    1,3,
    5,7 };

    // 视锥专用边表:近环 + 远环 + 4 条连接
    int frustumLine[24] = {
        0,1, 1,2, 2,3, 3,0,    // 近平面一圈
        4,5, 5,6, 6,7, 7,4,    // 远平面一圈
        0,4, 1,5, 2,6, 3,7     // 近→远 四条连接棱
    };

    std::vector<glm::vec4> lineVerts;
    lineVerts.reserve(mainDrawContext.OpaqueSurfaces.size() * 24);
    auto LineVertexPush = [&](const RenderObject& obj) {


        glm::mat4 matrix = obj.transform;
        float xMax = 0, xMin = 0, yMax = 0, yMin = 0, zMax = 0, zMin = 0;

        glm::vec4 Cube[8] = {};

        for (int c = 0; c < 8; c++) {
            // project each corner into world space
            Cube[c] = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);
            //transfer form Obj AABB to world AABB
            if (c == 0)
            {
                xMax = Cube[c].x;
                xMin = Cube[c].x;
                yMax = Cube[c].y;
                yMin = Cube[c].y;
                zMax = Cube[c].z;
                zMin = Cube[c].z;
            }
            else {
                xMax = glm::max(xMax, Cube[c].x);
                xMin = glm::min(xMin, Cube[c].x);
                yMax = glm::max(yMax, Cube[c].y);
                yMin = glm::min(yMin, Cube[c].y);
                zMax = glm::max(zMax, Cube[c].z);
                zMin = glm::min(zMin, Cube[c].z);
            }
        }

        std::array<glm::vec3, 8> worldCorners{
        glm::vec3 { xMax, yMax, zMax },
        glm::vec3 { xMax, yMax, zMin },
        glm::vec3 { xMax, yMin, zMax },
        glm::vec3 { xMax, yMin, zMin },
        glm::vec3 { xMin, yMax, zMax },
        glm::vec3 { xMin, yMax, zMin },
        glm::vec3 { xMin, yMin, zMax },
        glm::vec3 { xMin, yMin, zMin },
        };

        for (auto i : line)
            lineVerts.push_back(glm::vec4(worldCorners[i],1.0f));
    };

    for (auto& idx : opaque_draws)
    {
        LineVertexPush(mainDrawContext.OpaqueSurfaces[idx]);
    }

    //ADD grid of frustum
    glm::mat4 invVP = glm::inverse(_cullViewProj);   // ★ 用冻结的那个 viewproj
    // NDC 8 角(Vulkan z∈[0,1]):x,y ∈ {-1,1},z ∈ {0,1}
    glm::vec3 ndc[8] = {
        {-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0},   // 近平面
        {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1},   // 远平面
    };
    glm::vec3 frustumCorners[8];
    for (int i = 0; i < 8; i++) {
        glm::vec4 w = invVP * glm::vec4(ndc[i], 1.0f);
        frustumCorners[i] = glm::vec3(w) / w.w;       // ★ 除 w:透视除法反过来
    }

    //转24个点
    for (auto i : frustumLine)
        lineVerts.push_back(glm::vec4(frustumCorners[i], 1.0f));

    GPULineBuffers LineBuffer;
    LineBuffer = uploadLine(lineVerts);
    return LineBuffer;
}