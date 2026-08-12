#!/usr/bin/env python3
"""
读 paths/ 下所有 bench_*.csv,画 gpu_ms / (cpu_ms) / triangles 对比折线图 + 打印汇总表。
用法:  python plot_bench.py          (在 paths/ 目录跑)
- 按列名解析(表头感知),所以旧的无 cpu_ms 的 CSV 也能读。
- 任一 CSV 有 cpu_ms 列 → 自动多画一个 CPU 面板。
每次重录生成新 CSV 后再跑一遍即可刷新;新增 config 自动纳入。
"""
import csv, glob, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

def read_bench(path):
    name = os.path.basename(path).replace("bench_", "").replace(".csv", "")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    hdr = {c: i for i, c in enumerate(rows[0])}          # 列名 → 下标
    gpu, cpu, tris = [], [], []
    i = 1
    while i < len(rows) and rows[i] and rows[i][0] != "":
        r = rows[i]
        gpu.append(float(r[hdr["gpu_ms"]]))
        if "cpu_ms" in hdr:
            cpu.append(float(r[hdr["cpu_ms"]]))
        tris.append(float(r[hdr["triangles"]]))
        i += 1
    # 汇总段
    summary = {}
    for j in range(len(rows)):
        if rows[j] and rows[j][0] == "config":
            summary = dict(zip(rows[j], rows[j + 1])); break
    return name, gpu, cpu, tris, summary

def main():
    files = sorted(glob.glob(os.path.join(HERE, "bench_*.csv")))
    if not files:
        print("没找到 bench_*.csv"); return
    data = [read_bench(p) for p in files]
    has_cpu = any(len(cpu) > 0 for _, _, cpu, _, _ in data)

    npanels = 3 if has_cpu else 2
    fig, axes = plt.subplots(npanels, 1, figsize=(11, 3.6 * npanels), sharex=True)
    ax_gpu = axes[0]
    ax_cpu = axes[1] if has_cpu else None
    ax_tri = axes[-1]
    colors = plt.cm.tab10.colors

    # 打印汇总表(median 更稳,一并打印)
    print("\n| config | avg_ms | median_ms | cpu_avg | 1%low | draws |")
    print("|---|---|---|---|---|---|")
    for k, (name, gpu, cpu, tris, s) in enumerate(data):
        c = colors[k % 10]
        n = len(gpu)
        x = [f / (n - 1) * 100 for f in range(n)] if n > 1 else [0]
        gpu_avg, gpu_med = st.mean(gpu), st.median(gpu)
        ax_gpu.plot(x, gpu, color=c, lw=1.1, label=f"{name} (med {gpu_med:.2f} ms)")
        ax_gpu.axhline(gpu_med, color=c, ls="--", lw=1, alpha=0.6)
        if has_cpu and cpu:
            ax_cpu.plot(x, cpu, color=c, lw=1.1, label=f"{name} (med {st.median(cpu):.2f} ms)")
            ax_cpu.axhline(st.median(cpu), color=c, ls="--", lw=1, alpha=0.6)
        ax_tri.plot(x, [t / 1e6 for t in tris], color=c, lw=1.1, label=name)
        cpu_avg = f"{st.mean(cpu):.2f}" if cpu else "-"
        low1 = s.get("low1_ms", "-")
        print(f"| {s.get('config', name)} | {gpu_avg:.2f} | {gpu_med:.2f} | {cpu_avg} | {low1} | {s.get('draw_avg','')} |")

    ax_gpu.set_ylabel("GPU time (ms)")
    ax_gpu.set_title("GPU-Driven pipeline — per-frame along fixed camera path (dashed = median)")
    ax_gpu.grid(alpha=0.3); ax_gpu.legend(loc="upper right", fontsize=8)
    if ax_cpu is not None:
        ax_cpu.set_ylabel("CPU time (ms)\n(build + record)")
        ax_cpu.grid(alpha=0.3); ax_cpu.legend(loc="upper right", fontsize=8)
    ax_tri.set_ylabel("Triangles (M)"); ax_tri.set_xlabel("Path progress (%)")
    ax_tri.grid(alpha=0.3); ax_tri.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = os.path.join(HERE, "bench_chart.png")
    fig.savefig(out, dpi=150)
    print(f"\n图已保存: {out}")

if __name__ == "__main__":
    main()
