import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse
import numpy as np
import os


def generate_perfect_teaser():
    # 1. 准备画布 (18cm 宽，SCI 标准)
    fig = plt.figure(figsize=(12, 4), dpi=300)

    # 2. 加载你现有的素材 (请确保这三张图在当前目录下)
    # 如果没有 image_db5081.png，请替换为你任意一张背景图
    bg_path = 'image_db5081.png'

    if not os.path.exists(bg_path):
        print(f"❌ 错误：找不到背景图 {bg_path}。请下载一张你的测试集图片并重命名为 image_db5081.png")
        return

    img = mpimg.imread(bg_path)
    h, w, c = img.shape

    # --- [PANEL A]: Input ---
    ax1 = fig.add_axes([0.01, 0.1, 0.32, 0.8])  # [left, bottom, width, height]
    ax1.imshow(img)
    ax1.set_title("(a) Raw Video Stream", fontsize=11, fontweight='bold', y=-0.15)
    ax1.axis('off')
    # 标签
    ax1.text(0.05, 0.92, "Input", transform=ax1.transAxes, color='white', fontsize=9, fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))

    # --- [PANEL B]: Method (示意图) ---
    ax2 = fig.add_axes([0.34, 0.1, 0.32, 0.8])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("(b) Unified Framework", fontsize=11, fontweight='bold', y=-0.15)
    ax2.axis('off')

    # 画框框 (代表你的模型)
    rect_model = Rectangle((2, 2), 6, 6, facecolor='#f0f0f0', edgecolor='#333333', linewidth=1.5, zorder=1)
    ax2.add_patch(rect_model)
    ax2.text(5, 7, "Proposed System", ha='center', va='center', fontsize=10, fontweight='bold')

    # 内部模块 (Predictor)
    rect_pred = Rectangle((3, 5), 4, 1.5, facecolor='#dbebf7', edgecolor='blue', linewidth=1)
    ax2.add_patch(rect_pred)
    ax2.text(5, 5.75, "Trajectory\nPredictor", ha='center', va='center', fontsize=8, color='darkblue')

    # 内部模块 (Detector)
    rect_det = Rectangle((3, 3), 4, 1.5, facecolor='#fadddd', edgecolor='red', linewidth=1)
    ax2.add_patch(rect_det)
    ax2.text(5, 3.75, "UAV Detector\n(Zero-shot)", ha='center', va='center', fontsize=8, color='darkred')

    # 箭头
    ax2.annotate("", xy=(2, 5), xytext=(0, 5), arrowprops=dict(arrowstyle="->", lw=2))
    ax2.annotate("", xy=(10, 5), xytext=(8, 5), arrowprops=dict(arrowstyle="->", lw=2))

    # --- [PANEL C]: Output (关键可视化) ---
    ax3 = fig.add_axes([0.67, 0.1, 0.32, 0.8])
    ax3.imshow(img)
    ax3.set_title("(c) Prediction & Alert", fontsize=11, fontweight='bold', y=-0.15)
    ax3.axis('off')

    # 1. 画模拟的无人机框 (假设在右上角)
    # 你可以根据背景图调整 box 的位置 [x, y, w, h]
    uav_rect = Rectangle((w * 0.7, h * 0.2), w * 0.15, h * 0.1, fill=False, edgecolor='red', linewidth=2)
    ax3.add_patch(uav_rect)
    ax3.text(w * 0.7, h * 0.18, "ALERT: UAV", color='red', fontsize=8, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # 2. 画模拟的鸟类轨迹 (这就是你要的"好预测")
    # 假设有一只鸟从左下飞向中间
    # 过去轨迹 (绿色实线)
    past_x = np.linspace(w * 0.2, w * 0.35, 10)
    past_y = np.linspace(h * 0.7, h * 0.6, 10) + np.sin(np.linspace(0, 2, 10)) * 20
    ax3.plot(past_x, past_y, color='#00ff00', linewidth=2, label='Past')

    # 未来预测 (红色虚线)
    fut_x = np.linspace(w * 0.35, w * 0.55, 20)
    fut_y = np.linspace(h * 0.6, h * 0.45, 20) + np.sin(np.linspace(2, 5, 20)) * 30
    ax3.plot(fut_x, fut_y, color='red', linewidth=2, linestyle='--', label='Prediction')

    # 3. 画不确定性椭圆 (Student-t 的特征)
    # 在预测末端画一个半透明椭圆
    ellipse = Ellipse((fut_x[-1], fut_y[-1]), width=w * 0.08, height=h * 0.05, angle=-20,
                      facecolor='red', alpha=0.2, edgecolor='red')
    ax3.add_patch(ellipse)

    # 图例
    ax3.legend(loc='lower left', fontsize=6, framealpha=0.8)

    # 保存
    if not os.path.exists('figures'):
        os.makedirs('figures')
    save_path = 'figures/figure1_teaser.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ 完美 Figure 1 已生成！保存在: {save_path}")
    plt.show()


if __name__ == "__main__":
    generate_perfect_teaser()