import sys

# 从命令行参数中获取输入和输出文件路径
input_file = sys.argv[1]
output_file = sys.argv[2]

# 读取输入文件内容
with open(input_file, 'r') as f:
    data = f.read()

lines = data.strip().split('\n')

# 过滤ID从1开始到再次为0结束的数据
filtered_data = []
start_time = None
for line in lines:
    parts = line.split()
    if int(float(parts[0])) <= 0:
        continue
    if start_time is None:
        start_time = float(parts[-1])
    filtered_data.append(line)

# 将时间戳整体减去过滤后的第一个时间
processed_data = []
for line in filtered_data:
    parts = line.split()
    timestamp = int((float(parts[-1]) - start_time) * 1e9)
    gyro_x = int(float(parts[1]) * 2000)
    gyro_y = int(float(parts[2]) * 2000)
    gyro_z = int(float(parts[3]) * 2000)
    accel_x = int(float(parts[4]) * 500)
    accel_y = int(float(parts[5]) * 500)
    accel_z = int(float(parts[6]) * 500)
    processed_data.append(f'{timestamp} {accel_x} {accel_y} {accel_z} {gyro_x} {gyro_y} {gyro_z} ')
    # uint64_t timestamp; /*<   */
    # int16_t accel_x;    /*<  500  */
    # int16_t accel_y;    /*<    */
    # int16_t accel_z;    /*<    */
    # int16_t gyro_x;     /*<  2000  */
    # int16_t gyro_y;     /*<    */
    # int16_t gyro_z;     /*<    */

# 将处理后的数据写入输出文件
with open(output_file, 'w') as f:
    f.write('\n'.join(processed_data))