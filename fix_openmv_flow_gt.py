import sys

def fix_missing_data(input_file, output_file):
    data = []
    last_id = 0

    # 读取输入文件并保存数据
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                current_id = int(parts[2])
                if current_id - last_id > 1:
                    # 发现跳过的ID，进行填充
                    for id in range(last_id + 1, current_id):
                        filled_line = parts[:2] + [str(id)] + parts[3:]
                        data.append(filled_line)
                data.append(parts)
                last_id = current_id

    # 写入输出文件
    with open(output_file, 'w') as f:
        for line in data:
            f.write(' '.join(line) + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python script.py input_file output_file')
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        fix_missing_data(input_file, output_file)
