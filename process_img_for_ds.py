import os
import sys

# 从命令行参数中获取路径
path = sys.argv[1]
fps =  sys.argv[2]

# 获取路径下的所有文件
files = os.listdir(path)

# 过滤出以img_开头且以.jpg结尾的文件，并按照时间戳排序
image_files = sorted([file for file in files if file.startswith('img_') and file.endswith('.jpg')], key=lambda x: int(x.split('_')[1].split('.')[0]))

# 生成文档内容，将时间戳乘以30并放在文件名前面
document_content = '\n'.join([f"{int((int(file.split('_')[1].split('.')[0]) - 1)*30*1e6)} {file}" for file in image_files])

# 生成txt文档
output_file = path+"/image_list.txt"
with open(output_file, 'w') as f:
    f.write(document_content)

print(f"生成的文档已保存为 {output_file}")