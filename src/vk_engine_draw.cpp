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
#include <fstream>

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
    pc.cullEnabled = (_benchConfig == 2) ? 1u : 0u;
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
    auto _initStart = std::chrono::system_clock::now();
    //reset counters
    stats.drawcall_count = 0;
    stats.triangle_count = 0;


    if (!_freezeCull) _cullViewProj = sceneData.viewproj;

    // ==== _stressDup ====
    //if (_stressDup > 1) {
    //    auto& surf = mainDrawContext.OpaqueSurfaces;
    //    size_t orig = surf.size();
    //    surf.reserve(orig * _stressDup);
    //    for (int k = 1; k < _stressDup; k++) {
    //        glm::mat4 off = glm::translate(glm::mat4(1.f), glm::vec3(60.f * k, 0.f, 0.f));
    //        for (size_t s = 0; s < orig; s++) {
    //            RenderObject r = surf[s];        
    //            r.transform = off * r.transform;   
    //            surf.push_back(r);
    //        }
    //    }
    //}

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
        if (A.material == B.material) //same mat
        {
            if (A.indexBuffer == B.indexBuffer)//same mesh
            {
                return A.firstIndex < B.firstIndex;
            }
            return A.indexBuffer < B.indexBuffer;//not same mesh
        }
        else {
            return A.material < B.material;//even not same mat
        }
        });

    std::vector<GPUObjectData> objects;

    objects.reserve(mainDrawContext.OpaqueSurfaces.size() + mainDrawContext.TransparentSurfaces.size());
    for (auto& s : opaque_draws)
        objects.push_back({ mainDrawContext.OpaqueSurfaces[s].transform,
            mainDrawContext.OpaqueSurfaces[s].vertexBufferAddress,
            0,
            0,
            mainDrawContext.OpaqueSurfaces[s].bounds.origin,
            mainDrawContext.OpaqueSurfaces[s].bounds.sphereRadius,
            glm::vec4(mainDrawContext.OpaqueSurfaces[s].bounds.extents,1) });
    for (auto& s : mainDrawContext.TransparentSurfaces) objects.push_back({ s.transform, s.vertexBufferAddress });

    // 4b. 建 SSBO,Obj(顶点数据)
    AllocatedBuffer objectBuffer = create_buffer(objects.size() * sizeof(GPUObjectData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(objectBuffer); });

    //4b1. SSBO compact information
    size_t compactBytes = objects.size() * sizeof(uint32_t);   // ★ 覆盖 opaque+transparent(透明也读它,别只 opaque)
    AllocatedBuffer compactBuffer = create_buffer(compactBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(compactBuffer); });
    uint32_t* ci = (uint32_t*)compactBuffer.info.pMappedData;
    for (uint32_t k = 0; k < objects.size(); k++) ci[k] = k;     // 恒等:透明靠这个直取自己;cull 只覆盖 opaque 段,transparent 段保持恒等
    name_buffer(objectBuffer.buffer, "objectBuffer");
    name_buffer(compactBuffer.buffer, "compactInstances");



    //在init里拿到indirectbuf传递给Culldraw
    //既要和之前的比，又要和之后的比
    //和之前的比，是为了少切换pipeline和少重新赋值
    //和之后的比，是为了合批进行自动instance
    const RenderObject* rep = nullptr;     // 当前批的代表物体（取 indexCount/firstIndex/material 等）
    uint32_t batchStart = 0, instanceCount = 1;//firstindex = batchStart,指名当前是取GPUObject里哪个数据
    std::vector<VkDrawIndexedIndirectCommand> commands;//命令合批3.2

    //instanceCount 

    bool instancing = (_benchConfig != 0);
    auto sameBatch = [instancing](const RenderObject& a, const RenderObject& b) {
        if (!instancing) return false;     
        return a.material == b.material && a.indexBuffer == b.indexBuffer
            && a.firstIndex == b.firstIndex && a.indexCount == b.indexCount;
        };



    auto flush = [&]() {
        if (instanceCount == 0) return;
        const RenderObject& r = *rep;
        //this cmd's idx
        uint32_t cmdIdx = commands.size();

        if (groups.empty() || r.material != groups.back().material)
            groups.push_back({ r.material, cmdIdx, 1 }); 
        else
            groups.back().cmdCount += 1;


        //for (int i = 0; i < instanceCount; i++)
        //{
        commands.push_back({
        .indexCount = r.indexCount,
        .instanceCount = 0,   // change to instance, atomicadd by CS
        .firstIndex = r.firstIndex,    // 已经是 mega 相对(baseIndex+startIndex)，3.1 做好的
        .vertexOffset = 0,               // per-mesh BDA，不 rebase
        .firstInstance = batchStart,      // start:this set of instance
        });
        //    //add counters for triangles and draws
        stats.drawcall_count++;
        stats.triangle_count += r.indexCount * instanceCount / 3;
        //}
        for (int start = batchStart; start < batchStart + instanceCount; start++)
        {
            objects[start].batchId = cmdIdx;
        }

            
        };
    if (_benchConfig != 3) {   // naive 不建 indirect 命令(不然双重计数 drawcall + 虚高 CPU)
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
    }
    else {
        commands.push_back({ 0,0,0,0,0 });
    }


    // 4c. 填数据
    memcpy(objectBuffer.info.pMappedData, objects.data(), objects.size() * sizeof(GPUObjectData));
    //compact would be assignment in cs so no need to init

    // 4d. 分配 + 写描述符（set 2）
    objectDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _objectDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, objectBuffer.buffer, objects.size() * sizeof(GPUObjectData), 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, compactBuffer.buffer, compactBytes, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, objectDescriptor);
    }



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


    size_t bytesVis = (opaque_draws.size()) * sizeof(uint32_t);
    AllocatedBuffer visBuffer = create_buffer(bytesVis,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,        // STORAGE 是给阶段4 compute 写用，现在加上无妨
        VMA_MEMORY_USAGE_GPU_ONLY);
    name_buffer(visBuffer.buffer, "visBuffer");
    get_current_frame()._deletionQueue.push_function(
        [=, this]() { destroy_buffer(visBuffer); });

    // 4d. 分配 + 写描述符（set 2）
    commandDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _CommandDataDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_buffer(0, indirectBuf.buffer, bytes, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.write_buffer(1, visBuffer.buffer, bytesVis, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        writer.update_set(_device, commandDescriptor);
    }

    auto _initEnd = std::chrono::system_clock::now();
    stats.draw_init_cpu = std::chrono::duration_cast<std::chrono::microseconds>(_initEnd - _initStart).count() / 1000.f;
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

    if (_benchConfig == 3) {
        // naive
        for (uint32_t oi = 0; oi < opaque_draws.size(); oi++) {
            const RenderObject& r = mainDrawContext.OpaqueSurfaces[opaque_draws[oi]];
            if (r.material != lastMaterial) {
                lastMaterial = r.material;
                if (r.material->pipeline != lastPipeline) {
                    lastPipeline = r.material->pipeline;
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 2, 1, &objectDescriptor, 0, nullptr);
                }
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.material->pipeline->layout, 1, 1, &r.material->materialSet, 0, nullptr);
            }
            vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, oi);   // firstInstance=oi → compactInstances[oi]=oi → objects[oi]
            stats.drawcall_count++;
            stats.triangle_count += r.indexCount / 3;
        }
    }
    else {
        //GPU-indirect
        for (auto& g : groups) {
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
            vkCmdDrawIndexedIndirect(cmd, indirectBuf.buffer,
                g.cmdOffset * sizeof(VkDrawIndexedIndirectCommand),
                g.cmdCount,
                sizeof(VkDrawIndexedIndirectCommand));
        }
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

// ============ 相机路径:样条 + 弧长匀速回放 ============
static glm::vec3 catmullRom(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, float t)
{
    float t2 = t * t, t3 = t2 * t;
    return 0.5f * ((2.f * p1) + (-p0 + p2) * t
        + (2.f * p0 - 5.f * p1 + 4.f * p2 - p3) * t2
        + (-p0 + 3.f * p1 - 3.f * p2 + p3) * t3);
}

static float lerpAngle(float a, float b, float t)   // yaw 走最短路,防跨 ±π 甩头
{
    const float PI = 3.14159265358979323846f;
    float d = b - a;
    while (d > PI)  d -= 2.f * PI;
    while (d < -PI) d += 2.f * PI;
    return a + d * t;
}

glm::vec3 VulkanEngine::evalSpline(float u)          // u∈[0,N-1],整数命中控制点
{
    int N = (int)_pathPoints.size();
    int i = glm::clamp((int)floor(u), 0, N - 2);
    float t = u - i;
    return catmullRom(_pathPoints[glm::max(i - 1, 0)].pos, _pathPoints[i].pos,
        _pathPoints[i + 1].pos, _pathPoints[glm::min(i + 2, N - 1)].pos, t);
}

void VulkanEngine::rebuildArcLUT()
{
    _arcLUT.clear();
    int N = (int)_pathPoints.size();
    if (N < 2) { _totalLen = 0.f; return; }
    const int STEPS = 1000;
    float s = 0.f;
    glm::vec3 prev = evalSpline(0.f);
    _arcLUT.push_back({ 0.f, 0.f });
    for (int k = 1; k <= STEPS; k++) {
        float u = (float)k / STEPS * (N - 1);
        glm::vec3 p = evalSpline(u);
        s += glm::length(p - prev);
        prev = p;
        _arcLUT.push_back({ u, s });
    }
    _totalLen = s;
}

float VulkanEngine::arcLengthToU(float targetS)
{
    if (_arcLUT.empty()) return 0.f;
    if (targetS <= 0.f)        return _arcLUT.front().x;
    if (targetS >= _totalLen)  return _arcLUT.back().x;
    int lo = 0, hi = (int)_arcLUT.size() - 1;         // 二分找 s
    while (lo + 1 < hi) {
        int m = (lo + hi) / 2;
        if (_arcLUT[m].y < targetS) lo = m; else hi = m;
    }
    float s0 = _arcLUT[lo].y, s1 = _arcLUT[hi].y;
    float f = (s1 - s0 > 1e-6f) ? (targetS - s0) / (s1 - s0) : 0.f;
    return glm::mix(_arcLUT[lo].x, _arcLUT[hi].x, f);
}

void VulkanEngine::updatePlayback(float dt)
{
    int N = (int)_pathPoints.size();
    if (N < 2) { _camMode = CamMode::Free; return; }
    if (_lutDirty) { rebuildArcLUT(); _lutDirty = false; }

    _playDist += _playSpeedRun * dt;                  // 匀速:距离 = 锁定速度 × 时间
    if (_playDist >= _totalLen) { 
        _camMode = CamMode::Free; 
        if (_benchmarking) { _benchmarking = false; dumpBenchmark(); }
        return; 
    }  // 走完自动停

    float u = arcLengthToU(_playDist);
    mainCamera.position = evalSpline(u);              // 位置:弧长匀速

    // 朝向:同一 u 上插值 capture 存的 yaw/pitch(自动合相机约定)
    int i = glm::clamp((int)floor(u), 0, N - 2);
    float t = u - i;
    mainCamera.yaw   = lerpAngle(_pathPoints[i].yaw,  _pathPoints[i + 1].yaw,  t);
    mainCamera.pitch = glm::mix(_pathPoints[i].pitch, _pathPoints[i + 1].pitch, t);
}

// 第三视角:cull 视锥沿录制路径飞行(循环),渲染相机(mainCamera)保持自由
void VulkanEngine::updateObserveCull(float dt)
{
    int N = (int)_pathPoints.size();
    if (N < 2) { _camMode = CamMode::Free; return; }
    if (_lutDirty) { rebuildArcLUT(); _lutDirty = false; }

    _playDist += _playSpeedRun * dt;
    if (_totalLen > 0.f) _playDist = std::fmod(_playDist, _totalLen);   // 循环观察

    float u = arcLengthToU(_playDist);
    int i = glm::clamp((int)floor(u), 0, N - 2);
    float t = u - i;

    Camera cullCam;                                     // 临时相机:复用同一套 yaw/pitch 约定
    cullCam.position = evalSpline(u);
    cullCam.yaw   = lerpAngle(_pathPoints[i].yaw,  _pathPoints[i + 1].yaw,  t);
    cullCam.pitch = glm::mix(_pathPoints[i].pitch, _pathPoints[i + 1].pitch, t);

    glm::mat4 view = cullCam.getViewMatrix();
    float aspect = (float)_windowExtent.width / (float)_windowExtent.height;
    glm::mat4 proj = glm::perspective(glm::radians(70.f), aspect, 300.f, 1.0f); // cull 远平面=300(与常规一致)
    proj[1][1] *= -1;                                   // 和 sceneData.proj 同款 Y 翻转,保证 cull/线框一致
    _cullViewProj = proj * view;
}

void VulkanEngine::update_scene()
{
    if (_camMode == CamMode::Playing)
        updatePlayback(stats.frametime_CPU / 1000.f);   // 第一人称飞路径
    else
        mainCamera.update();                            // Free / ObserveCull:自由飞观察相机

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

    // 第三视角观察:cull 视锥沿录制路径飞,渲染仍是自由的 mainCamera
    if (_camMode == CamMode::ObserveCull)
        updateObserveCull(stats.frametime_CPU / 1000.f);   // 覆盖 _cullViewProj(freeze 已开,draw_init 不会再改它)

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(1.f);
    sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);

}

void VulkanEngine::dumpBenchmark()
{
    if (_bench.empty()) return;
    const char* names[] = { "baseline-noinstance", "baseline-instance", "instance-gpucull", "naive" };

    // gpu_ms:升序、丢 warmup、算 avg / 1%low / worst
    std::vector<float> g; for (auto& s : _bench) g.push_back(s.gpu_ms);
    std::sort(g.begin(), g.end());
    int warmup = std::min((int)g.size() / 10, 20);
    if ((int)g.size() > warmup) g.erase(g.begin(), g.begin() + warmup);
    int n = (int)g.size();
    int k = std::max(1, n / 100);
    double sum = 0;  for (float x : g) sum += x;
    double low = 0;  for (int i = n - k; i < n; i++) low += g[i];   // 尾部=最高=最差1%
    float avg = (float)(sum / n), low1 = (float)(low / k), worst = g[n - 1];

    // 三角形 / draws / cpu 取全程均值
    double ts = 0, ds = 0, cs = 0; for (auto& s : _bench) { ts += s.tris; ds += s.draws; cs += s.cpu_ms; }
    float triAvg = (float)(ts / _bench.size()), drawAvg = (float)(ds / _bench.size());
    float cpuAvg = (float)(cs / _bench.size());

    std::string path = "../../paths/bench_" + std::string(names[_benchConfig]) + ".csv";
    std::ofstream f(path);
    f << "frame,gpu_ms,cpu_ms,triangles,draws\n";                    // 逐帧明细(画折线)
    for (size_t i = 0; i < _bench.size(); i++)
        f << i << "," << _bench[i].gpu_ms << "," << _bench[i].cpu_ms << ","
          << _bench[i].tris << "," << _bench[i].draws << "\n";
    f << "\nconfig,avg_ms,cpu_avg_ms,low1_ms,worst_ms,tri_avg,draw_avg,frames\n";  // 汇总(拼对比表)
    f << names[_benchConfig] << "," << avg << "," << cpuAvg << "," << low1 << "," << worst << ","
        << triAvg << "," << drawAvg << "," << n << "\n";
    f.close();
    fmt::print("benchmark dumped: {} ({} frames)\n", path, _bench.size());
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