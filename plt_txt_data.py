import sys
import matplotlib.pyplot as plt

x_aix = 'Timestamp(s)'
y_aix = 'Times(s)'

y_lable = ['offset', 'filter_offset', 'offset_delta']

cut_num  = 0


def plot_coordinates(filename):
    x = []
    y = []

    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) >= 2:
                x.append(float(values[0]))
                y.append([float(value) for value in values[1:]])

    plt.figure()
    for i in range(len(y[0])-cut_num):
        plt.plot(x, [y_val[i] for y_val in y], label=y_lable[i])

    plt.xlabel(x_aix)
    plt.ylabel(y_aix)
    plt.legend()
    plt.show()

# 从命令行参数中获取文件路径
if len(sys.argv) > 1:
    filename = sys.argv[1]
    plot_coordinates(filename)
else:
    print("请提供文本文件的路径作为命令行参数")
