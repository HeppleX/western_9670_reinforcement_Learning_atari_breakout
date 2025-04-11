# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
file1_path = "breakout_online_4/loss.xlsx"
file2_path = "breakout_offline_3/loss.xlsx"
df1 = pd.read_excel(file1_path, engine='openpyxl')
df2 = pd.read_excel(file2_path, engine='openpyxl')
# 查看数据
#print(df.head())

x = df1['epoch'][:50]
y1 = df1['loss'][:50]          # 第一个 Y 轴的数据
y2 = df2['loss'][:50]     # 第二个 Y 轴的数据

# 创建图形和第一个 Y 轴
fig, ax1 = plt.subplots()

# 绘制第一个 Y 轴的曲线
ax1.plot(x, y1, 'r-', label='online training loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('online training loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# 创建第二个 Y 轴
ax2 = ax1.twinx()

# 绘制第二个 Y 轴的曲线
ax2.plot(x, y2, 'b-', label='offline training loss')
ax2.set_ylabel('offline training loss', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图像
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('training loss compare')
plt.tight_layout()
plt.show()
