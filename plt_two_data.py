import matplotlib.pyplot as plt
import sys
# 读取数据
def read_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    x1, y1 = [], []
    for line in data:
        values = line.split()
        x1.append(float(values[0]))
        y1.append(float(values[1])+1.5)
    return x1, y1

# 绘制曲线
def plot_curves(x1, y1):
    plt.figure(figsize=(10, 6))
    # 横坐标为序号，纵坐标为原始数据
    plt.plot(range(len(x1)), x1, marker='.', label='marker_h')
    plt.plot(range(len(y1)), y1, marker='.', label='navpose_z')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Two Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# 主函数
def main():
    filename = sys.argv[1]
    x1, y1 = read_data(filename)
    plot_curves(x1, y1)

if __name__ == '__main__':
    main()
