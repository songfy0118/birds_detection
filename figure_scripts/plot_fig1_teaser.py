import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import numpy as np
import os

# --- 风格设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


def draw_sci_box(ax, x, y, w, h, text, subtext=None, color='#E3F2FD', edge='#1565C0'):
    """绘制 SCI 风格的圆角框"""
    # 阴影
    shadow = patches.FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    fc='gray', ec='none', alpha=0.1, zorder=9)
    ax.add_patch(shadow)
    # 主体
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  fc=color, ec=edge, lw=1.5, zorder=10)
    ax.add_patch(rect)
    # 文字
    cx, cy = x + w / 2, y + h / 2
    if subtext:
        ax.text(cx, cy + 0.25, text, ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)
        ax.text(cx, cy - 0.3, subtext, ha='center', va='center', fontsize=10, color='#424242', zorder=11)
    else:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)
    return (x + w, y + h / 2), (x, y + h / 2)  # 返回连接点


def draw_connection(ax, x1, y1, x2, y2, color='#455A64'):
    """绘制平滑箭头"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                                lw=2.5, color=color, shrinkA=0, shrinkB=0), zorder=5)


def create_teaser_v2():
    # 1. 准备画布 (18:5 比例，让每个子图接近 16:9 的视频比例)
    fig = plt.figure(figsize=(18, 5), dpi=300, facecolor='white')

    # GridSpec: [Input, arrow, Framework, arrow, Output]
    # 调整 width_ratios 让左右两边图稍微宽一点，中间窄一点，视觉平衡
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1.6, 0.15, 1.2, 0.15, 1.6], wspace=0.02)

    # 2. 素材
    file_input = 'image_db5081.png'
    file_output = 'image_db50c8.png'

    # --- (a) Input Stream (左图) ---
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')

    if os.path.exists(file_input):
        img = mpimg.imread(file_input)
        # 获取图片尺寸，用于统一高度
        h_img, w_img = img.shape[:2]
        ax1.imshow(img)

        # 装饰：Input 标签
        ax1.text(0.03, 0.92, "Input Video", transform=ax1.transAxes, color='white', fontsize=11, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3))
    else:
        # 占位
        ax1.set_xlim(0, 16);
        ax1.set_ylim(0, 9)
        ax1.add_patch(patches.Rectangle((0, 0), 16, 9, fc='#EEEEEE', ec='gray'))
        ax1.text(8, 4.5, "Raw Video\n(Missing)", ha='center', va='center')

    ax1.set_title("(a) Raw Video Stream", y=-0.15, fontsize=14, fontweight='bold')

    # --- (b) Unified Framework (中图) ---
    # 我们手动设定它的坐标系，让它看起来跟左右两张图"等高"
    # 假设左右图是 16:9，我们也给中间设定类似的比例感
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 10)  # 这里的 0-10 对应左右图的高度
    ax2.axis('off')

    # 系统大框
    sys_box = patches.FancyBboxPatch((0.5, 0.5), 11, 9, boxstyle="round,pad=0.2,rounding_size=0.3",
                                     fc='#F5F5F5', ec='#616161', lw=2, ls='--')
    ax2.add_patch(sys_box)
    ax2.text(6, 8.8, "Proposed System", ha='center', fontsize=13, fontweight='bold', color='#424242')

    # 模块 1: Trajectory Predictor
    tp_r, tp_l = draw_sci_box(ax2, 2, 5.8, 8, 2.2,
                              "Trajectory Predictor", "Student-t MDN",
                              color='#E1F5FE', edge='#0288D1')

    # 模块 2: UAV Detector
    ud_r, ud_l = draw_sci_box(ax2, 2, 2.0, 8, 2.2,
                              "UAV Detector", "Zero-shot OWL-ViT",
                              color='#FFEBEE', edge='#C62828')

    ax2.set_title("(b) Unified Framework", y=-0.15, fontsize=14, fontweight='bold')

    # --- (c) Prediction & Alert (右图) ---
    ax3 = fig.add_subplot(gs[4])
    ax3.axis('off')

    bg_img = mpimg.imread(file_output) if os.path.exists(file_output) else (
        mpimg.imread(file_input) if os.path.exists(file_input) else None)

    if bg_img is not None:
        ax3.imshow(bg_img)
        h, w = bg_img.shape[:2]

        # 1. 绘制轨迹
        t = np.linspace(0, 1, 50)
        sx, sy = w * 0.3, h * 0.6
        ex, ey = w * 0.7, h * 0.35

        # 过去轨迹 (绿色实线)
        past_x = sx - (ex - sx) * 0.3 * t[::-1]
        past_y = sy - (ey - sy) * 0.3 * t[::-1] + np.sin(t * 3) * 15
        line_past, = ax3.plot(past_x, past_y, color='#00E676', lw=3, label='Past Trajectory')

        # 未来预测 (红色虚线)
        fut_x = sx + (ex - sx) * t
        fut_y = sy + (ey - sy) * t + np.sin(t * 3) * 30
        line_pred, = ax3.plot(fut_x, fut_y, color='#FF1744', lw=3, ls='--', label='Predicted Future')

        # 2. 绘制不确定性椭圆
        ell = patches.Ellipse((fut_x[-1], fut_y[-1]), width=w * 0.12, height=h * 0.08, angle=-15,
                              fc='#FF1744', alpha=0.25, ec='none', label='Uncertainty')
        ax3.add_patch(ell)

        # 3. UAV Alert
        uav_x, uav_y = w * 0.7, h * 0.15
        uav_w, uav_h = w * 0.18, h * 0.12
        rect = patches.Rectangle((uav_x, uav_y), uav_w, uav_h, lw=2.5, ec='#D50000', fc='none', label='UAV Detection')
        ax3.add_patch(rect)
        ax3.text(uav_x, uav_y - 10, "ALERT: UAV", color='white', fontsize=9, fontweight='bold',
                 bbox=dict(fc='#D50000', ec='none', pad=2))

        # 4. 添加专业图例 (Legend) -- 这是重点！
        # 我们手动创建图例句柄，以获得最佳控制
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='#00E676', lw=3),
            Line2D([0], [0], color='#FF1744', lw=3, ls='--'),
            patches.Patch(facecolor='#FF1744', alpha=0.3, edgecolor='none'),  # 椭圆示意
            patches.Patch(facecolor='none', edgecolor='#D50000', lw=2)  # 红框示意
        ]
        ax3.legend(custom_lines, ['Past (8 frames)', 'Prediction (60 frames)', 'Uncertainty', 'UAV Alert'],
                   loc='lower left', fontsize=9, framealpha=0.9, facecolor='white', edgecolor='gray')

        # 标签
        ax3.text(0.03, 0.92, "Output", transform=ax3.transAxes, color='white', fontsize=11, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3))

    else:
        ax3.text(0.5, 0.5, "Output Missing", ha='center')

    ax3.set_title("(c) Prediction & Alert", y=-0.15, fontsize=14, fontweight='bold')

    # --- 绘制跨子图箭头 ---
    # 使用 Figure 坐标系，确保箭头连接紧密且对齐

    # 箭头 1: Input -> System
    # 从 ax1 右边缘中心 -> ax2 左边缘中心
    # 这里简单处理，在 gs[1] (间隔) 画箭头
    ax_arr1 = fig.add_subplot(gs[1]);
    ax_arr1.axis('off');
    ax_arr1.set_ylim(0, 1)
    ax_arr1.annotate("", xy=(1, 0.65), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color='#455A64'))  # 上路
    ax_arr1.annotate("", xy=(1, 0.35), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color='#455A64'))  # 下路

    # 箭头 2: System -> Output
    # 在 gs[3] (间隔) 画箭头
    ax_arr2 = fig.add_subplot(gs[3]);
    ax_arr2.axis('off');
    ax_arr2.set_ylim(0, 1)
    ax_arr2.annotate("", xy=(1, 0.5), xytext=(0, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color='#455A64'))  # 上路汇聚
    ax_arr2.annotate("", xy=(1, 0.5), xytext=(0, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color='#455A64'))  # 下路汇聚

    # 保存
    if not os.path.exists('figures'): os.makedirs('figures')
    save_path = 'figures/teaser_enhanced.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ 最终完美版 Figure 1 已生成: {save_path}")


if __name__ == "__main__":
    create_teaser_v2()