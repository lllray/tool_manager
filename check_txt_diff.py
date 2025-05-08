import sys

def compare_files(file1, file2):
    try:
        # 打开两个文件并读取所有行
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        # 获取两文件的最大行数
        max_lines = max(len(lines1), len(lines2))

        # 标记是否有不同行
        differences_found = False

        # 逐行比较
        for i in range(max_lines):
            line1 = lines1[i].strip() if i < len(lines1) else None
            line2 = lines2[i].strip() if i < len(lines2) else None

            if line1 != line2:
                differences_found = True
                print(f"Line {i + 1} differs:")
                print(f"  File1: {line1}")
                print(f"  File2: {line2}")
                print("-" * 40)

        # 如果没有发现差异，打印提示
        if not differences_found:
            print("The two files are identical.")

    except FileNotFoundError as e:
        print(f"Error: {e}")

# 示例用法
if __name__ == "__main__":
    # 检查命令行参数数量是否正确
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1> <file2>")
        sys.exit(1)

    # 从命令行获取文件路径
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # 调用比较函数
    compare_files(file1, file2)
