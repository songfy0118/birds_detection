import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob


def visualize_predictions():
    # --- 配置路径 ---
    # 1. 背景图片 (你上传的那张绿色田野图)
    # 请确保文件名正确，或者替换为你本地的任意一张测试集图片
    bg_img_path = 'image_db5081.png'

    # 2. 预测数据文件夹 (你截图里的那个文件夹)
    csv_dir = 'output/preds_csv_mini_best'

    # 3. 输出图片保存位置
    output_file = 'figures/qualitative_results.png'

    # --- 检查文件 ---
    if not os.path.exists(bg_img_path):
        print(f"❌ 错误：找不到背景图片 {bg_img_path}，请把图片放在当前目录下！")
        return

    if not os.path.exists(csv_dir):
        print(f"❌ 错误：找不到CSV文件夹 {csv_dir}")
        return

    # 获取所有 csv 文件
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("❌ 错误：CSV 文件夹是空的！")
        return

    print(f"✅ 找到 {len(csv_files)} 个轨迹文件。正在绘图...")

    # --- 开始绘图 ---
    # 设置高清画布
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # 1. 显示背景图
    img = mpimg.imread(bg_img_path)
    ax.imshow(img)

    # 2. 随机选几个 CSV 画上去 (选太多会乱，选5个左右)
    # 为了效果好，你可以手动指定几个文件名，或者随机选
    import random
    selected_files = random.sample(csv_files, min(len(csv_files), 5))

    for csv_file in selected_files:
        try:
            # 读取 CSV
            # 假设列名是 x, y (像素坐标)
            df = pd.read_csv(csv_file)

            # 检查列名，适配你的格式
            if 'x' not in df.columns or 'y' not in df.columns:
                # 如果没有表头，假设第4,5列是x,y (根据常见格式盲猜)
                x = df.iloc[:, -2].values
                y = df.iloc[:, -1].values
            else:
                x = df['x'].values
                y = df['y'].values

            # 绘制轨迹线 (红色虚线代表预测)
            ax.plot(x, y, color='red', linewidth=2.5, linestyle='--', alpha=0.8, label='Predicted')

            # 绘制终点 (X标记)
            ax.scatter(x[-1], y[-1], color='yellow', marker='x', s=100, zorder=10, linewidth=2)

        except Exception as e:
            print(f"跳过文件 {csv_file}: {e}")
            continue

    # 3. 美化图片
    ax.set_title("Bird Trajectory Prediction Results (Mini-BirdFormer)", fontsize=16, fontweight='bold', color='white',
                 backgroundcolor='black')
    ax.axis('off')  # 去掉坐标轴

    # 添加图例 (只显示一个)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend([handles[0]], ['Predicted Trajectory'], loc='upper right', fontsize=12, frameon=True)

    # 4. 保存
    os.makedirs('figures', exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    print(f"🎉 成功！图片已保存至: {output_file}")
    plt.show()


if __name__ == "__main__":
    visualize_predictions()