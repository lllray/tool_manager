import matplotlib.pyplot as plt
import sys
import numpy as np
# 读取文件并解析数据
file_path = sys.argv[1]  # 替换为实际的文件路径
yaws = []
rolls = []
pitches = []
real_rolls = []
real_pitches = []

with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('focus'):
            _, yaw_str, roll_str, pitch_str, real_roll_str, real_pitch_str = line.strip().split(' ')
            yaw = float(yaw_str.split(':')[1])
            roll = float(roll_str.split(':')[1])
            pitch = float(pitch_str.split(':')[1])
            real_roll = float(real_roll_str.split(':')[1])
            real_pitch = float(real_pitch_str.split(':')[1])
            yaws.append(yaw)
            rolls.append(roll)
            pitches.append(pitch)
            real_rolls.append(real_roll)
            real_pitches.append(real_pitch)

# 绘制曲线
plt.plot(yaws, label='Yaw')
plt.plot(rolls, label='Roll')
plt.plot(pitches, label='Pitch')
plt.plot(real_rolls, label='Real Roll')
plt.plot(real_pitches, label='Real Pitch')
print("mean_real_rolls:{}, mean_real_pitches:{}".format(np.mean(real_rolls), np.mean(real_pitches)))
plt.xlabel('Sample')
plt.ylabel('Angle')
plt.title('Roll and Pitch')
plt.legend()
plt.show()