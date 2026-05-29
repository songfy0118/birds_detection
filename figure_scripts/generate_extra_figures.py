# generate_extra_figures.py
#
# 作用：生成论文里的两张定量图
#   - birds/figures/traj_bar.png
#   - birds/figures/uav_metrics.png
#
# 运行方式：
#   cd day1
#   python generate_extra_figures.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============ 路径配置 ============
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "birds" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============ 1. 轨迹预测对比图 ============
def plot_traj_bar():
    """
    画一张 Baseline vs Ours 的 ADE/FDE 对比柱状图。
    把下面的数字改成你最终的实验结果即可。
    """

    methods = ["Baseline", "Ours"]

    # TODO: 这里换成你真实的实验结果（像素或米）
    ade = np.array([7.4, 6.1])   # 示例：Baseline ADE=7.4, Ours=6.1
    fde = np.array([12.1, 10.5]) # 示例：Baseline FDE=12.1, Ours=10.5

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(x - width/2, ade, width, label="ADE")
    ax.bar(x + width/2, fde, width, label="FDE")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Error")
    ax.set_title("Trajectory Prediction Error on FBD-SV")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = FIG_DIR / "traj_bar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path.name} generated.")


# ============ 2. UAV 影响与检测性能图 ============
def plot_uav_metrics():
    """
    画一张展示 UAV presence 影响的图：
      - 例子：No-UAV vs With-UAV 的 ADE
      - 再叠加 UAV recall/precision (单独坐标轴)
    你可以只用 ADE，也可以改成 mADE / F1，都行。
    """

    # TODO: 这里换成你真实的结果
    # 假设：10 个视频，No-UAV ADE=6.0, With-UAV ADE=7.5
    ade_no_uav = 6.0
    ade_with_uav = 7.5

    # 假设：UAV presence detection 的召回和精度
    recall = 0.80  # 80%
    precision = 0.75

    scenarios = ["No UAV", "With UAV"]
    ade = np.array([ade_no_uav, ade_with_uav])

    x = np.arange(len(scenarios))

    fig, ax1 = plt.subplots(figsize=(4.8, 3.5))

    # ADE 柱状图
    ax1.bar(x, ade, width=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylabel("ADE")
    ax1.set_title("Effect of UAV Presence on Prediction and Detection")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # 第二个 y 轴画 UAV recall / precision
    ax2 = ax1.twinx()
    ax2.plot([x[1], x[1]], [recall, precision], marker="o")  # 只是示意，你可以改成两条线
    ax2.set_ylabel("UAV metrics")
    ax2.set_ylim(0.0, 1.0)

    # 简单标注一下点
    ax2.text(x[1] + 0.02, recall, f"Recall={recall:.2f}", va="bottom", fontsize=8)
    ax2.text(x[1] + 0.02, precision, f"Prec={precision:.2f}", va="bottom", fontsize=8)

    out_path = FIG_DIR / "uav_metrics.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] {out_path.name} generated.")


if __name__ == "__main__":
    print("Generating extra LaTeX figures into birds/figures ...")
    plot_traj_bar()
    plot_uav_metrics()
    print("All done.")
