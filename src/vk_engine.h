// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <camera.h>

struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
	glm::vec4 data5;
};

struct ComputeEffect {
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		deletors.clear();
	}
};

struct FrameData {
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	VkSemaphore _swapchainSemaphore;
	VkFence _renderFence;

	DeletionQueue _deletionQueue;
	DescriptorAllocatorGrowable _frameDescriptors;

	VkQueryPool _timestampPool;
	VkQueryPool _pipelineStatsPool;
};

struct GPUSceneData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; // w for sun power
	glm::vec4 sunlightColor;
	glm::vec4 cameraPos;
};

constexpr unsigned int FRAME_OVERLAP = 2;
constexpr bool bUseValidationLayers = true;

//
//// base class for a renderable dynamic object
//class IRenderable {
//
//	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
//};


struct RenderObject {
	uint32_t firstIndex;
	uint32_t indexCount;
   	VkBuffer indexBuffer;

	MaterialInstance* material;
	Bounds bounds;
	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct GPUObjectData {
	glm::mat4    render_matrix;
	VkDeviceAddress vertexBuffer; 
	uint64_t _pad; // 8  → 偏移 72，凑到 80
	glm::vec3 origin; 
	float sphereRadius;
	glm::vec4 extents;      
};



struct CullPush {
	glm::mat4 viewproj;   // 偏移 0，占 64
	uint32_t  count;      // 偏移 64，uint ← 和 shader 的 uint count 对上
};

struct LinePush {
	glm::mat4 viewproj;   // 偏移 0，占 64
	VkDeviceAddress vertexBufferAddress;    // 偏移 64，uint ← 和 shader 的 uint count 对上
	int mode;
};


struct MatGroup { 
	MaterialInstance* material; 
	uint32_t cmdOffset, cmdCount;
};

struct DrawContext {
	std::vector<RenderObject> OpaqueSurfaces;
	std::vector<RenderObject> TransparentSurfaces;
};

struct GLTFMetallic_Roughness {
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;

	VkDescriptorSetLayout materialLayout;

	struct MaterialConstants {
		glm::vec4 colorFactors;
		glm::vec4 metal_rough_factors;
		//padding, we need it anyway for uniform buffers
		glm::vec4 extra[14];
	};

	struct MaterialResources {
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_pipelines(VulkanEngine* engine);
	void clear_resources(VkDevice device);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};


struct MeshNode : public Node {

	std::shared_ptr<MeshAsset> mesh;

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct EngineStats {
	float frametime_CPU;
	int triangle_count;
	int drawcall_count;
	float scene_update_time_CPU;
	float mesh_draw_time_CPU;

	float gpu_ms_geometry = 0;
	float gpu_ms_history[120] = {};
	int   gpu_ms_offset = 0;

	int triangle_count_GPU = 0;
};

struct PathPoint { glm::vec3 pos; float yaw, pitch; };

enum class CamMode { Free, Playing, ObserveCull };

class VulkanEngine {
public:
	EngineStats stats;
	Camera mainCamera;
	VmaAllocator _allocator;

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };


	static VulkanEngine& Get();

	//allocate Pool
	DescriptorAllocator globalDescriptorAllocator;

	//get mode
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	//get actual memories
	VkDescriptorSet _drawImageDescriptors;

	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	VkPipeline _cullPipeline;
	VkPipelineLayout _cullPipelineLayout;



	VkPipelineLayout _linePipelineLayout;
	VkPipeline _linePipeline;

	bool resize_requested;
	//initializes everything in the engine
	void init();


	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	void draw_background(VkCommandBuffer cmd);
	void draw_init();
	void init_line();
	void draw_clear();


	void draw_Cull(VkCommandBuffer cmd);
	void draw_line(VkCommandBuffer cmd);
	void draw_geometry(VkCommandBuffer cmd);

	// --- omitted ---

	VkInstance _instance;// Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger;// Vulkan debug output handle
	VkPhysicalDevice _chosenGPU;// GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface;// Vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	std::vector<VkSemaphore> _renderSemaphores;   // size == _swapchainImages.size()
	VkExtent2D _swapchainExtent;

	//FrameWork
	FrameData _frames[FRAME_OVERLAP];

	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	DeletionQueue _mainDeletionQueue;

	//draw resources
	AllocatedImage _drawImage;
	AllocatedImage _depthImage;
	VkExtent2D _drawExtent;
	float renderScale = 1.f;
	// immediate submit structures
	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	std::vector<ComputeEffect> backgroundEffects;
	ComputeEffect CullpipelineEffects;
	int currentBackgroundEffect{ 0 };

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
	GPULineBuffers uploadLine(std::span<glm::vec4> vertices);
	GPULineBuffers PushVertex();
	//buffer
	void destroy_buffer(const AllocatedBuffer& buffer);
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	GPUMeshBuffers rectangle;


	std::vector<std::shared_ptr<MeshAsset>> testMeshes;

	GPUSceneData sceneData;

	VkDescriptorSetLayout _objectDataDescriptorLayout;
	VkDescriptorSetLayout _lineObjectDataDescriptorLayout;

	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
	VkDescriptorSetLayout _singleImageDescriptorLayout;

	VkDescriptorSetLayout _CommandDataDescriptorLayout;


	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& img);

	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckerboardImage;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	MaterialInstance defaultData;
	GLTFMetallic_Roughness metalRoughMaterial;

	std::vector<MatGroup> groups;


	AllocatedBuffer _megaIndexBuffer;
	AllocatedBuffer indirectBuf;

	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

	void update_scene();
	void build_mega_index_buffer(std::vector<std::shared_ptr<MeshAsset>>& allMeshes);

	//timestamp
	float _timestampPeriod;
	bool _gpuTimingEnabled;

	std::vector<uint32_t> opaque_draws;
	VkDescriptorSet objectDescriptor;
	VkDescriptorSet commandDescriptor;
	void name_buffer(VkBuffer buf, const char* name);

	//freeze View
	glm::mat4 _cullViewProj;      // 喂给 compute 的剔除矩阵
	bool      _freezeCull = false;
	bool      _ShowAABB = true;

	std::vector<PathPoint> _pathPoints;

	void savePath(const std::string& path);
	void loadPath(const std::string& path);

	// ---- 相机路径回放(样条 + 弧长匀速)----
	CamMode _camMode      = CamMode::Free;
	float   _playSpeed    = 5.0f;   // 面板可调(单位/秒)
	float   _playSpeedRun = 5.0f;   // Play 时快照一次,回放全程只用它
	float   _playDist     = 0.0f;   // 已走弧长
	std::vector<glm::vec2> _arcLUT; // (u, s) 弧长表
	float   _totalLen     = 0.0f;
	bool    _lutDirty      = true;

	glm::vec3 evalSpline(float u);         // Catmull-Rom,u∈[0,N-1]
	void      rebuildArcLUT();             // 采样建弧长表
	float     arcLengthToU(float targetS); // 弧长反查参数
	void      updatePlayback(float dt);    // 每帧回放:写回 mainCamera
	void      updateObserveCull(float dt); // 第三视角:cull 视锥飞路径,mainCamera 自由
private:
	void init_descriptors();
	void init_pipelines();
	void init_background_pipelines();
	void init_Cull_pipelines();
	void init_Line_pipelines();
	void init_vulkan();
	void init_swapchain();
	void resize_swapchain();
	void init_commands();
	void init_sync_structures();
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	void init_imgui();

	void init_default_data();
};

extern VulkanEngine* loadedEngine;