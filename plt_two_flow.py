import re
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse

def parse_gt_data(gt_file_path, data_gt):
    data_gt["ID"] = []
    data_gt["VX"] = []
    data_gt["VY"] = []
    data_gt["HG"] = []
    with open(gt_file_path, "r") as file:
        for line in file:
            values = line.strip().split()
            data_gt["VX"].append(float(values[0]))
            data_gt["VY"].append(float(values[1]))
            data_gt["ID"].append(int(values[2]))
            data_gt["HG"].append(float(values[3]))

def parse_result_data(log_file_path, data):
    # Regular expression patterns to extract numbers, Type, and ID
    number_pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
    data["ID"] = []
    data["TIME"] = []
    data["FPS"] = []
    data["VX"] = []
    data["VY"] = []
    data["QUAL"] = []
    # Read the log file line by line
    with open(log_file_path, "r") as file:
        for line in file:
            # Extract numbers, Type, and ID from the current line using regular expressions
            numbers = re.findall(number_pattern, line)

            # Remove the plus/minus signs from the numbers
            numbers = [num.replace("+", "") for num in numbers]

            if numbers:
                # Store the extracted values in the dictionary
                data["ID"].append(int(numbers[0]))
                data["TIME"].append(int(numbers[1]))
                data["FPS"].append(float(numbers[2]))
                data["VX"].append(float(numbers[3]))
                data["VY"].append(float(numbers[4]))
                data["QUAL"].append(float(numbers[5]))

# 创建解析器对象
parser = argparse.ArgumentParser(description='命令行参数解析')

# 添加命令行参数
parser.add_argument('-f', '--fft', type=str, help='fft flow result path')
parser.add_argument('-k', '--klt', type=str, help='klt flow result path')
parser.add_argument('-d', '--deep_sea', type=str, help='deep_sea flow result path')
parser.add_argument('-g', '--gt', type=str, help='gt flow result path')

# 解析命令行参数
args = parser.parse_args()

data_num = 0
gt_num = 0
data_dict = {}
# 数据读取
if args.fft:
    print('fft flow result path:', args.fft)
    data_dict['fft'] = {}
    parse_result_data(args.fft, data_dict['fft'])
    data_num += 1
if args.klt:
    print('klt flow result path:', args.klt)
    data_dict['klt'] = {}
    parse_result_data(args.klt, data_dict['klt'])
    data_num += 1
if args.deep_sea:
    print('deep_sea flow result path:', args.deep_sea)
    data_dict['deep_sea'] = {}
    parse_result_data(args.deep_sea, data_dict['deep_sea'])
    data_num += 1
if args.gt:
    print('gt flow result path:', args.gt)
    data_dict['gt'] = {}
    parse_gt_data(args.gt, data_dict['gt'])
    gt_num = 1

#数据格式转换
for data_key in data_dict:
    data_dict[data_key] = {key: np.array(value) for key, value in data_dict[data_key].items()}


fx_dt = 138.323/30
fy_dt = 184.426/30
offset = 2

# Plot the curves
if gt_num > 0:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6), sharex=True, sharey=True)
else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)

for key in data_dict:
    if key == "fft":
        ax1.plot(data_dict[key]["ID"], data_dict[key]["VX"]/fx_dt, label="fft_vx")
        ax2.plot(data_dict[key]["ID"], data_dict[key]["VY"]/fy_dt, label="fft_vy")
    elif key == "klt":
        print(data_dict[key]["VX"])
        ax1.plot(data_dict[key]["ID"], data_dict[key]["VX"]/fx_dt, label="klt_vx")
        ax2.plot(data_dict[key]["ID"], -data_dict[key]["VY"]/fy_dt, label="klt_vy")
    elif key == "deep_sea":
        ax1.plot(data_dict[key]["ID"], -data_dict[key]["VX"]/fx_dt, label="deep_sea_vx")
        ax2.plot(data_dict[key]["ID"], data_dict[key]["VY"]/fy_dt, label="deep_sea_vy")

if gt_num > 0:
    ax1.plot(data_dict["gt"]["ID"]-offset, -data_dict["gt"]["VY"], label="gt_vx") # -gt_y = fft_x
    ax2.plot(data_dict["gt"]["ID"]-offset, -data_dict["gt"]["VX"], label="gt_vy") # -gt_x = fft_y
    for key in data_dict:
        if key == "fft":
            ax3.plot(data_dict["gt"]["ID"][:len(data_dict["gt"]["ID"])-offset], abs(-data_dict["gt"]["VY"][offset:] - (data_dict[key]["VX"][:len(data_dict["gt"]["ID"])-offset]/fx_dt)), label="error_fft_vx")
            ax4.plot(data_dict["gt"]["ID"][:len(data_dict["gt"]["ID"])-offset], abs(-data_dict["gt"]["VX"][offset:] - (data_dict[key]["VY"][:len(data_dict["gt"]["ID"])-offset]/fy_dt)), label="error_fft_vy")
        elif key == "klt":
            ax3.plot(data_dict["gt"]["ID"][:len(data_dict["gt"]["ID"])-offset], abs(-data_dict["gt"]["VY"][offset:] - (data_dict[key]["VX"][:len(data_dict["gt"]["ID"])-offset]/fx_dt)), label="error_klt_vx")
            ax4.plot(data_dict["gt"]["ID"][:len(data_dict["gt"]["ID"])-offset], abs(-data_dict["gt"]["VX"][offset:] + (data_dict[key]["VY"][:len(data_dict["gt"]["ID"])-offset]/fy_dt)), label="error_klt_vy")
        elif key == "deep_sea":
            ax3.plot(data_dict["gt"]["ID"][:len(data_dict["gt"]["ID"])-offset], abs(-data_dict["gt"]["VY"][offset:] + (data_dict[key]["VX"][:len(data_dict["gt"]["ID"])-offset]/fx_dt)), label="error_deep_sea_vx")
            ax4.plot(data_dict["gt"]["ID"][:len(data_dict["gt"]["ID"])-offset], abs(-data_dict["gt"]["VX"][offset:] - (data_dict[key]["VY"][:len(data_dict["gt"]["ID"])-offset]/fy_dt)), label="error_deep_sea_vy")
    ax3.plot(data_dict["gt"]["ID"], data_dict["gt"]["HG"], label="gt_h") # -gt_y = fft_x
    ax4.plot(data_dict["gt"]["ID"], data_dict["gt"]["HG"], label="gt_h") # -gt_y = fft_x
    ax3.legend()
    ax4.legend()
    ax3.grid(True)
    ax4.grid(True)
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
plt.xlabel("TimeStamps(ms)")
plt.ylabel("Velocities(m/s)")
plt.show()
