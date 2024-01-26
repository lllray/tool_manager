import re
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse

def calculate_rmse_errors(error_data):
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(np.mean(error_data ** 2))
    return rmse

def calculate_pixel_errors(error_x, error_y, threshold):
    # 计算位移向量的欧几里得距离
    displacement = np.sqrt(error_x ** 2 + error_y ** 2)

    # 计算1像素误差的数量
    pixel_error_count = np.sum(displacement > threshold)

    # 计算总像素数量
    total_pixels = np.prod(displacement.shape)

    # 计算像素误差的百分比
    pixel_error_percentage = (pixel_error_count / total_pixels) * 100
    return pixel_error_percentage

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
                data["TIME"].append(float(numbers[1]))
                data["FPS"].append(float(numbers[2]))
                data["VX"].append(float(numbers[3]))
                data["VY"].append(float(numbers[4]))
                data["QUAL"].append(float(numbers[5]))

# 创建解析器对象
parser = argparse.ArgumentParser(description='命令行参数解析')

# 添加命令行参数
parser.add_argument('-f', '--fft', type=str, help='fft flow result path')
parser.add_argument('-cv', '--opencv', type=str, help='opencv flow result path')
parser.add_argument('-d', '--ds_klt', type=str, help='ds_klt flow result path')
parser.add_argument('-m', '--mcu_klt', type=str, help='mcu_klt flow result path')
parser.add_argument('-ol', '--online', type=str, help='online flow result path')
parser.add_argument('-g', '--gt', type=str, help='gt flow result path')
parser.add_argument('-e', '--eval', type=str, help='eval result path')
parser.add_argument('-s', '--start', type=int, help='start count')

# 解析命令行参数
args = parser.parse_args()

data_num = 0
gt_num = 0
data_dict = {}
gt_data = {}
# 数据读取
if args.fft:
    print('fft flow result path:', args.fft)
    data_dict['fft'] = {}
    parse_result_data(args.fft, data_dict['fft'])
    data_num += 1
if args.opencv:
    print('opencv_klt flow result path:', args.opencv)
    data_dict['opencv_klt'] = {}
    parse_result_data(args.opencv, data_dict['opencv_klt'])
    data_num += 1
if args.ds_klt:
    print('ds_klt flow result path:', args.ds_klt)
    data_dict['ds_klt'] = {}
    parse_result_data(args.ds_klt, data_dict['ds_klt'])
    data_num += 1
if args.mcu_klt:
    print('mcu_klt flow result path:', args.mcu_klt)
    data_dict['mcu_klt'] = {}
    parse_result_data(args.mcu_klt, data_dict['mcu_klt'])
    data_num += 1
if args.online:
    print('online flow result path:', args.online)
    data_dict['mcu_klt_opt'] = {}
    parse_result_data(args.online, data_dict['mcu_klt_opt'])
    data_num += 1
if args.gt:
    print('gt flow result path:', args.gt)
    parse_gt_data(args.gt, gt_data)
    gt_num = 1

start_count = 0
if args.start:
    print('start count:', args.start)
    start_count = args.start

#数据格式转换
for data_key in data_dict:
    data_dict[data_key] = {key: np.array(value) for key, value in data_dict[data_key].items()}
    filtered_indexes = np.where(data_dict[data_key]['ID'] > start_count)
    data_dict[data_key] = {key: value[filtered_indexes] for key, value in data_dict[data_key].items()}
    print(data_dict[data_key])
gt_data = {key: np.array(value) for key, value in gt_data.items()}
filtered_indexes = np.where(gt_data['ID'] > start_count)
gt_data = {key: value[filtered_indexes] for key, value in gt_data.items()}
print(gt_data)
fx_dt = 138.323/30
fy_dt = 184.426/30
offset = 2

# Plot the curves
if gt_num > 0:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)

for key in data_dict:
    ax1.plot(data_dict[key]["ID"], data_dict[key]["VX"]/fx_dt, label="{}_vx".format(key))
    ax2.plot(data_dict[key]["ID"], data_dict[key]["VY"]/fy_dt, label="{}_vy".format(key))

error_dict = {}
if gt_num > 0:
    ax1.plot(gt_data["ID"]-offset, gt_data["VY"], label="gt_vx") # -gt_y = fft_x
    ax2.plot(gt_data["ID"]-offset, gt_data["VX"], label="gt_vy") # -gt_x = fft_y
    for key in data_dict:
        error_dict[key] = {}
        error_dict[key]["error_vx"] = gt_data["VY"][offset:] - (data_dict[key]["VX"][:len(gt_data["ID"])-offset]/fx_dt)
        error_dict[key]["error_vy"] = gt_data["VX"][offset:] - (data_dict[key]["VY"][:len(gt_data["ID"])-offset]/fy_dt)
        ax3.plot(gt_data["ID"][:len(gt_data["ID"])-offset], abs(error_dict[key]["error_vx"]), label="error_{}_vx".format(key))
        ax4.plot(gt_data["ID"][:len(gt_data["ID"])-offset], abs(error_dict[key]["error_vy"]), label="error_{}_vy".format(key))
    ax1.plot(gt_data["ID"], gt_data["HG"], label="gt_h") # -gt_y = fft_x
    ax2.plot(gt_data["ID"], gt_data["HG"], label="gt_h") # -gt_y = fft_x
    ax3.legend()
    ax4.legend()
    ax3.grid(True)
    ax4.grid(True)
    ax3.set_ylabel("error_x(rad/s)")
    ax4.set_ylabel("error_y(rad/s)")
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
plt.xlabel("TimeStamps(ms)")
ax1.set_ylabel("v_x(rad/s)")
ax2.set_ylabel("v_y(rad/s)")

print("\n")
if gt_num > 0 and args.eval:
    pixel_threshold = 1.0
    error_max = 0
    with open(args.eval, 'w') as file:
        file.write('start count:{}\n'.format(args.start))
        for key in data_dict:
            title = "Eval {} flow:\n".format(key)
            file.write(title)
            print(title)
            vx_rmse = calculate_rmse_errors(error_dict[key]["error_vx"])
            vy_rmse = calculate_rmse_errors(error_dict[key]["error_vy"])
            error_max = max(error_max,vx_rmse)
            error_max = max(error_max,vy_rmse)
            pixel_error = calculate_pixel_errors(error_dict[key]["error_vx"] * fx_dt, error_dict[key]["error_vy"] * fy_dt, pixel_threshold)
            result = "  vx_rmse:{:.3f}, vy_rmse:{:.3f}, {:.1f} pixel_error:{:.3f}%\n".format(vx_rmse, vy_rmse, pixel_threshold, pixel_error)
            file.write(result)
            print(result)
    ax3.set_ylim(-0.01, 3*error_max)
    ax4.set_ylim(-0.01, 3*error_max)
    plt.savefig(args.eval.replace('.txt', '.png'))
plt.show()

