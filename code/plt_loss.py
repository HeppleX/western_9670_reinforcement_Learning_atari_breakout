# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
#file_path = "breakout_online_4/loss.xlsx"
file_path = "breakout_offline_3/loss.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# 查看数据
print(df.head())

# 绘制折线图
plt.figure(figsize=(10, 6))  # 设置图表尺寸
plt.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='b', label='loss')


# 关闭横轴的科学计数法
plt.ticklabel_format(style='plain', axis='x', useOffset=False)

# 添加标题和标签
plt.title('loss trend in training')
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.xticks(rotation=45)  # 旋转日期以便显示

# 显示图例
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.show()

