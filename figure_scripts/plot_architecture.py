import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# ================== 绘图辅助函数 ==================
def draw_box(x, y, w, h, text, color='#e1f5fe', edge='black', style='solid', fontsize=11):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=edge, facecolor=color, linestyle=style, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=3)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'))

def draw_circle(x, y, r, text, color='#fff9c4'):
    circle = patches.Circle((x, y), r, linewidth=2, edgecolor='black', facecolor=color, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)

# ================== 1. 输入层 (Input) ==================
# 文本框：Input Sequence
draw_box(0.5, 4, 2.5, 2, "Input\nTrajectory\n$X_t = \{\Delta p_t\}$", color='#fce4ec')
draw_arrow(3.0, 5, 4.0, 5)

# Linear Projection
draw_box(4.0, 4, 2, 2, "Linear\nProj", color='#e0f2f1')
draw_arrow(6.0, 5, 7.0, 5)

# Positional Encoding (圆形 +号)
draw_circle(5.0, 7.5, 0.6, "Pos\nEnc", color='#fff9c4')
draw_arrow(5.0, 6.9, 5.0, 6.1) # 向下指到加号位置

# 加号示意
ax.text(5.0, 6.0, "⊕", fontsize=20, ha='center', va='center', zorder=4, color='#333')

# ================== 2. Encoder (Mini-BirdFormer) ==================
# 大虚线框表示 Encoder 整体
rect = patches.Rectangle((7.0, 2.5), 5.0, 6.0, linewidth=2, edgecolor='#1565c0', facecolor='none', linestyle='--')
ax.add_patch(rect)
ax.text(9.5, 8.8, "Transformer Encoder\n(Mini: 2 Layers)", ha='center', fontsize=10, color='#1565c0', fontweight='bold')

# 内部层 Layer 1
draw_box(7.5, 3.5, 4.0, 1.5, "Layer 1\n(Self-Attn + MLP)", color='#e3f2fd', fontsize=10)
# 内部层 Layer 2
draw_box(7.5, 6.0, 4.0, 1.5, "Layer 2\n(Self-Attn + MLP)", color='#bbdefb', fontsize=10)

# 层间连接箭头
draw_arrow(9.5, 5.0, 9.5, 6.0)

# Encoder 输出箭头
draw_arrow(12.0, 5, 13.0, 5)

# ================== 3. MDN Head (核心创新) ==================
# 全连接层
draw_box(13.0, 4, 2, 2, "MDN\nHead\n(Linear)", color='#f3e5f5')

# 分叉箭头指向四个参数
# 这里的坐标精心调整过，为了好看
draw_arrow(15.0, 5, 16.0, 8) # 指向 pi
draw_arrow(15.0, 5, 16.0, 6) # 指向 mu
draw_arrow(15.0, 5, 16.0, 4) # 指向 sigma
draw_arrow(15.0, 5, 16.0, 2) # 指向 nu

# 参数圆圈 (Student-t 的四个参数)
draw_circle(16.5, 8, 0.6, "$\pi$", color='#d1c4e9')      # 混合权重
draw_circle(16.5, 6, 0.6, "$\mu$", color='#d1c4e9')      # 均值
draw_circle(16.5, 4, 0.6, "$\Sigma$", color='#d1c4e9')   # 协方差
draw_circle(16.5, 2, 0.6, "$\\nu$", color='#ffcc80')     # 自由度 (Highlight!)

# 参数文字说明
ax.text(17.3, 8, "Weights", ha='left', va='center', fontsize=10)
ax.text(17.3, 6, "Means", ha='left', va='center', fontsize=10)
ax.text(17.3, 4, "Scales", ha='left', va='center', fontsize=10)
ax.text(17.3, 2, "Degrees of\nFreedom", ha='left', va='center', fontsize=10, fontweight='bold', color='#ef6c00')

# ================== 4. 最终输出 ==================
# 汇聚箭头到最终结果框
draw_arrow(17.5, 8, 18.8, 5.5)
draw_arrow(17.5, 6, 18.8, 5.5)
draw_arrow(17.5, 4, 18.8, 4.5)
draw_arrow(17.5, 2, 18.8, 4.5)

# 最终输出框
draw_box(18.8, 4, 1.2, 2, "$P(Y|X)$", color='#fff3e0', edge='#ef6c00')

plt.tight_layout()
# 保存到 figures 文件夹
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig('figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
print("✅ 架构图已生成: figures/architecture_diagram.png")
# plt.show()