import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as path_effects

# 设置 SCI 风格字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# 画布设置
fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
ax.set_xlim(0, 22)
ax.set_ylim(0, 9)
ax.axis('off')


# ================== 🎨 核心绘图函数 ==================

def draw_sci_box(x, y, w, h, label, sub_label=None, color='#E3F2FD', edge='#1565C0', style='-', zorder=10):
    """绘制带阴影和双层文字的 SCI 风格矩形框"""
    # 阴影
    shadow = patches.FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc='gray', ec='none', alpha=0.2, zorder=zorder - 1)
    ax.add_patch(shadow)

    # 主体
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                  fc=color, ec=edge, lw=1.5, ls=style, zorder=zorder)
    ax.add_patch(rect)

    # 文字
    cx, cy = x + w / 2, y + h / 2
    if sub_label:
        ax.text(cx, cy + 0.3, label, ha='center', va='center', fontsize=11, fontweight='bold', color='#000000',
                zorder=zorder + 1)
        ax.text(cx, cy - 0.3, sub_label, ha='center', va='center', fontsize=9, color='#424242', zorder=zorder + 1)
    else:
        ax.text(cx, cy, label, ha='center', va='center', fontsize=11, fontweight='bold', color='#000000',
                zorder=zorder + 1)

    return (x + w, y + h / 2), (x, y + h / 2)  # 返回右中点, 左中点


def draw_sci_arrow(x1, y1, x2, y2, text=None, color='#455A64'):
    """绘制带文字的平滑箭头"""
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='-|>', mutation_scale=15,
                                    color=color, lw=1.5, zorder=5)
    ax.add_patch(arrow)
    if text:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        t = ax.text(mid_x, mid_y + 0.2, text, ha='center', va='bottom', fontsize=9, color=color, zorder=6)
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])


def draw_node_circle(x, y, r, label, color='#FFF9C4', edge='#FBC02D'):
    """绘制参数节点圆形"""
    circle = patches.Circle((x, y), r, fc=color, ec=edge, lw=1.5, zorder=15)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=16)


# ================== 🏗️ 开始构建架构 ==================

# --- 1. 输入阶段 (Input Stage) ---
# 轨迹序列框
in_r, _ = draw_sci_box(1, 3.5, 2.5, 2, "Trajectory", "$X = \{\Delta p_t\}$", color='#F3E5F5', edge='#8E24AA')

# 线性投影
proj_r, proj_l = draw_sci_box(4.5, 3.5, 2, 2, "Linear", "Embed", color='#E0F2F1', edge='#00695C')

# 箭头: Input -> Linear
draw_sci_arrow(in_r[0], in_r[1], proj_l[0], proj_l[1])

# 位置编码 (Pos Enc)
pos_x, pos_y = 5.5, 7.0
draw_node_circle(pos_x, pos_y, 0.6, "PE", color='#FFF9C4', edge='#FBC02D')

# 加法操作 (Add)
add_x, add_y = 5.5, 5.7
ax.text(add_x, add_y, "$\oplus$", fontsize=20, ha='center', va='center', color='#424242')

# 箭头: Linear -> Add (隐式)
draw_sci_arrow(5.5, 5.5, 5.5, 5.9, color='#00695C')  # 上指
# 箭头: PE -> Add
draw_sci_arrow(pos_x, pos_y - 0.6, add_x, add_y + 0.2)

# --- 2. 编码器 (Encoder) ---
# 虚线大框
enc_rect = patches.Rectangle((7, 2.5), 5, 5.5, lw=1.5, ec='#1565C0', fc='none', ls='--')
ax.add_patch(enc_rect)
ax.text(9.5, 8.2, "Mini-BirdFormer Encoder", ha='center', fontsize=10, fontweight='bold', color='#1565C0',
        bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Layer 1
l1_r, l1_l = draw_sci_box(7.5, 3.2, 4, 1.2, "Transformer Layer 1", "MHSA + FFN", color='#E3F2FD', edge='#2196F3')
# Layer 2
l2_r, l2_l = draw_sci_box(7.5, 5.8, 4, 1.2, "Transformer Layer 2", "MHSA + FFN", color='#BBDEFB', edge='#1976D2')

# 箭头: Add -> Layer 1 (入口)
draw_sci_arrow(add_x + 0.3, 5.7, l1_l[0], 3.8)  # 稍微调整箭头路径
# 箭头: Layer 1 -> Layer 2
draw_sci_arrow(9.5, 4.4, 9.5, 5.8)

# --- 3. 输出头 (MDN Head) ---
# 从 Encoder 出来
mdn_l = (13.5, 4.5)
draw_sci_arrow(l2_r[0], l2_r[1], mdn_l[0] - 1.5, l2_r[1])  # 水平出
draw_sci_arrow(mdn_l[0] - 1.5, l2_r[1], mdn_l[0], mdn_l[1] + 1)  # 折线入 (为了美观)

# MDN Head 框
head_r, head_l = draw_sci_box(13.5, 3.5, 2, 3, "MDN Head", "MLP Layers", color='#FFEBEE', edge='#C62828')

# 分叉箭头指向参数
params_start_x = head_r[0]
params_start_y = head_r[1]

# 绘制四个 Student-t 参数
param_configs = [
    (18, 7.5, "$\pi$", "Weights", '#D1C4E9'),
    (18, 6.0, "$\mu$", "Means", '#C5CAE9'),
    (18, 4.5, "$\Sigma$", "Scales", '#BBDEFB'),
    (18, 3.0, "$\\nu$", "DOF (Heavy-tail)", '#FFCC80')  # 高亮这个
]

for px, py, sym, name, col in param_configs:
    # 箭头
    draw_sci_arrow(params_start_x, params_start_y, px - 0.6, py)
    # 圆圈
    draw_node_circle(px, py, 0.6, sym, color=col, edge='gray')
    # 文字
    ax.text(px + 0.8, py, name, ha='left', va='center', fontsize=10)

# --- 4. 最终输出示意 (Output Distribution) ---
# 在最右侧画一个象征性的分布图框
dist_box = patches.Rectangle((20, 4), 1.5, 2, fc='white', ec='black', lw=1)
# ax.add_patch(dist_box) # 这个框可选，如果太挤就不画了
# 汇聚
# draw_sci_arrow(19.5, 5, 20, 5)

# --- 底部标注 ---
ax.text(11, 0.5,
        "Figure 2: The Mini-BirdFormer Architecture. The encoder extracts temporal features,\nwhich are decoded into Student-t mixture parameters ($\pi, \mu, \Sigma, \\nu$) to capture heavy-tailed uncertainty.",
        ha='center', fontsize=10, style='italic', color='#555555')

# 保存
import os

if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig('figures/architecture_diagram.png', bbox_inches='tight', pad_inches=0.1)
print("✅ SCI-Level Architecture Diagram generated: figures/architecture_diagram.png")
# plt.show()