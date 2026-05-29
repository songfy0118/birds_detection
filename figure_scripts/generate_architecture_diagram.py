import os
from graphviz import Digraph


def create_architecture_diagram():
    # 创建一个有向图
    dot = Digraph(comment='Mini-BirdFormer Architecture')

    # 设置图的属性：从左到右布局，节点形状等
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.8', ranksep='1.0')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E8F4F8', fontname='Helvetica')

    # --- 1. 输入部分 ---
    # 定义输入节点
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Sequence', style='dashed', color='gray')
        c.node('input', 'Input Trajectory\n$\\Delta p_{t-T+1} \\dots \\Delta p_t$\n(Shape: $T \\times 2$)',
               shape='parallelogram', fillcolor='#FFF2CC')
        c.node('linear', 'Linear Projection\n($2 \\to 96$)', fillcolor='#D9EAD3')
        c.node('pos', 'Positional\nEncoding', shape='circle', width='0.8', fillcolor='#FCE5CD')
        c.node('add', '+', shape='circle', width='0.5', fixedsize='true')

        # 连接输入部分
        c.edge('input', 'linear')
        c.edge('linear', 'add')
        c.edge('pos', 'add')

    # --- 2. Encoder 部分 (Mini-Transformer) ---
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Mini-Transformer Encoder', style='bold', color='#4A86E8', bgcolor='#F4F8FF')

        # 画两个层代表 N=2
        c.node('enc1', 'Transformer Layer 1\n(Self-Attention + MLP)', width='2.5')
        c.node('enc2', 'Transformer Layer 2\n(Self-Attention + MLP)', width='2.5')

        # 参数标注节点 (不可见，仅作标注)
        c.node('params', 'Hyperparams:\n$d_{model}=96$\n$N_{heads}=4$\n$N_{layers}=2$', shape='note',
               fillcolor='#FFF2CC', width='1.5')

        c.edge('enc1', 'enc2')

    # --- 3. Output Head 部分 (Student-t MDN) ---
    with dot.subgraph(name='cluster_head') as c:
        c.attr(label='Student-t MDN Head', style='bold', color='#E06666', bgcolor='#FFF0F0')

        c.node('fc', 'Output Projection\n(Linear)', width='2.0')

        # 输出参数节点
        c.node('pi', 'Mixing Coeffs\n$\\pi$ (Softmax)', shape='ellipse', fillcolor='#D9D2E9')
        c.node('mu', 'Means\n$\\mu$ (Linear)', shape='ellipse', fillcolor='#D9D2E9')
        c.node('sigma', 'Scales\n$\\Sigma$ (Exp)', shape='ellipse', fillcolor='#D9D2E9')
        c.node('nu', 'Degrees of Freedom\n$\\nu$ (Softplus)', shape='ellipse', fillcolor='#FFD966',
               penwidth='2.0')  # 重点高亮 nu

        c.edge('fc', 'pi')
        c.edge('fc', 'mu')
        c.edge('fc', 'sigma')
        c.edge('fc', 'nu')

    # --- 4. 最终输出 ---
    dot.node('output', 'Probabilistic Forecast\n$P(Y|X)$', shape='doubleoctagon', fillcolor='#dbebf7',
             fontname='Helvetica-Bold')

    # --- 连接各个模块 ---
    dot.edge('add', 'enc1')
    dot.edge('enc2', 'fc')

    # 从各个参数指向最终输出
    dot.edge('pi', 'output')
    dot.edge('mu', 'output')
    dot.edge('sigma', 'output')
    dot.edge('nu', 'output')

    # --- 保存图片 ---
    # 确保 figures 目录存在
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'architecture_diagram')
    # render 会生成 PDF 和 Source 文件，这里我们要 png
    dot.render(output_path, format='png', cleanup=True)

    print(f"✅ Network Architecture Diagram generated at: {output_path}.png")


if __name__ == '__main__':
    # 检查 graphviz 是否安装
    try:
        create_architecture_diagram()
    except Exception as e:
        print("❌ 生成失败。请确保您安装了 graphviz。")
        print(f"错误信息: {e}")
        print("提示: 您可能需要先安装 graphviz 软件 (不仅仅是 pip install graphviz)")
        print("如果无法安装，您可以使用简单的 PPT 截图代替。")