//> includes
#include "vk_engine.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;

void VulkanEngine::savePath(const std::string& path) {
    json j;
    // j["duration"] = _pathDuration;   // 有 duration 就存,现在没有可省
    for (auto& p : _pathPoints)
        j["points"].push_back({
            {"pos", {p.pos.x, p.pos.y, p.pos.z}},
            {"yaw", p.yaw}, {"pitch", p.pitch}
            });
    std::filesystem::create_directories(std::filesystem::path(path).parent_path()); // 目录不存在就建
    std::ofstream(path) << j.dump(2);    // 缩进 2 空格,人可读可手改
}


void VulkanEngine::loadPath(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;                       // 文件不存在直接返回,不崩
    json j; f >> j;
    _pathPoints.clear();
    for (auto& e : j["points"]) {
        PathPoint p;
        p.pos = { e["pos"][0], e["pos"][1], e["pos"][2] };
        p.yaw = e["yaw"];
        p.pitch = e["pitch"];
        _pathPoints.push_back(p);
    }
}