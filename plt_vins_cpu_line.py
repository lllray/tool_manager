import matplotlib.pyplot as plt

# 从外部文本文件读取数据
filename = '../3588_test/cpu.txt'  # 外部文本文件的路径和名称
values = []  # 存储倒数第三列数据的列表

with open(filename, 'r') as file:
    for line in file:
        data = line.split()  # 使用空格分割每行数据
        if len(data) >= 3:
            value = float(data[-3])  # 提取倒数第三列数据
            values.append(value)

# 绘制曲线
plt.plot(values)
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Curve Plot')
plt.show()
