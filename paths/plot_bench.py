#!/usr/bin/env python3
"""
读 paths/ 下所有 bench_*.csv,画各配置逐帧 gpu_ms 折线对比,输出 bench_gpu_ms.png。
用法:  python plot_bench.py       (在 paths/ 目录下跑,或从任意目录跑都行)
依赖:  pip install matplotlib
CSV 格式(dumpBenchmark 产出):
    frame,gpu_ms,triangles,draws
    0,10.05,17920,136
    ...
    <空行>
    config,avg_ms,low1_ms,worst_ms,tri_avg,draw_avg,frames
    gpucull,8.1,12.3,...
"""
import csv, glob, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    """返回 (frames[], gpu_ms[], summary_dict|None)"""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    frames, gpu = [], []
    i = 1  # 跳过表头 "frame,gpu_ms,..."
    while i < len(rows) and rows[i] and rows[i][0] != "":
        frames.append(int(rows[i][0]))
        gpu.append(float(rows[i][1]))
        i += 1

    summary = None
    for j in range(i, len(rows)):
        if rows[j] and rows[j][0] == "config":
            summary = dict(zip(rows[j], rows[j + 1]))
            break
    return frames, gpu, summary


def main():
    files = sorted(glob.glob(os.path.join(HERE, "bench_*.csv")))
    if not files:
        print("no bench_*.csv found in", HERE)
        return

    plt.figure(figsize=(10, 5))
    print(f"{'config':<10} {'avg_ms':>8} {'1%low':>8} {'worst':>8} {'draws':>8} {'frames':>7}")
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0].replace("bench_", "")
        frames, gpu, s = load(path)
        label = name
        if s:
            label = f"{name}  avg {float(s['avg_ms']):.2f}ms · 1%low {float(s['low1_ms']):.2f}ms"
            print(f"{name:<10} {float(s['avg_ms']):>8.2f} {float(s['low1_ms']):>8.2f} "
                  f"{float(s['worst_ms']):>8.2f} {float(s['draw_avg']):>8.0f} {s['frames']:>7}")
        plt.plot(frames, gpu, linewidth=1.0, label=label)

    plt.xlabel("frame")
    plt.ylabel("GPU time (ms)")
    plt.title("GPU-driven culling — per-frame GPU time along camera path")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(HERE, "bench_gpu_ms.png")
    plt.savefig(out, dpi=140)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
