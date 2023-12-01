import re
import matplotlib.pyplot as plt

# 从外部文本文件读取数据
filename = '../3588_test/time.txt'  # 外部文本文件的路径和名称
feature_tracker_times = []  # 存储featureTracker时间的列表
process_times = []  # 存储process时间的列表

with open(filename, 'r') as file:
    for line in file:
        feature_tracker_match = re.search(r'featureTracker time: (\d+\.\d+)', line)
        process_match = re.search(r'process time: (\d+\.\d+)', line)
        if feature_tracker_match:
            feature_tracker_time = float(feature_tracker_match.group(1))
            feature_tracker_times.append(feature_tracker_time)
        elif process_match:
            process_time = float(process_match.group(1))
            process_times.append(process_time)

# 绘制曲线
plt.plot(feature_tracker_times, label='featureTracker Time')
plt.plot(process_times, label='Process Time')
plt.xlabel('Index')
plt.ylabel('Time (ms)')
plt.title('Curve Plot')
plt.legend()
plt.show()
