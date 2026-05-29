import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from math import ceil

# --- 配置 ---
SOURCE_DIR = 'best_uav_shots'  # 你放精选图片的文件夹
OUTPUT_PATH = 'figures/final_results/uav_detection_examples.png'  # 输出路径
GRID_COLS = 3  # 一行放几张图 (建议 2 或 3)


def create_collage():
    # 1. 获取图片
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 找不到文件夹: {SOURCE_DIR}，请先创建并放入图片！")
        return

    images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg'))]
    if not images:
        print("❌ 文件夹里没图片！")
        return

    # 限制数量，最多展示6张，多了太乱
    images = images[:6]
    n_images = len(images)
    n_rows = ceil(n_images / GRID_COLS)

    # 2. 创建画布
    fig, axes = plt.subplots(n_rows, GRID_COLS, figsize=(5 * GRID_COLS, 4 * n_rows))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)  # 图片间距极小，更紧凑

    # 展平 axes 方便遍历
    axes = axes.flatten() if n_images > 1 else [axes]

    print(f"🖼️ 正在拼接 {n_images} 张图片...")

    for i, img_name in enumerate(images):
        img_path = os.path.join(SOURCE_DIR, img_name)
        img = mpimg.imread(img_path)

        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')  # 去掉坐标轴

        # 可选：在左上角加标签 (a), (b) 等
        # label = chr(97 + i) # a, b, c...
        # ax.text(0.05, 0.95, f"({label})", transform=ax.transAxes,
        #         color='white', fontsize=14, fontweight='bold', va='top',
        #         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # 3. 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ 拼接图已生成: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_collage()