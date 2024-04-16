import sys
import os

class BenchmarkBase:
    def __init__(self, title1 = "算法", sort_with_name = True):
        self.title1 = title1
        self.sort_with_name = sort_with_name

    def get_base_and_parent_name(self,current_path):
        return os.path.basename(os.path.dirname(current_path)) + "/" + os.path.basename(current_path)
    def get_relpath(self,path2, path1):
        return os.path.relpath(path2, path1)
    def remove_repeat_line(self,data):
        # 根据第二列元素创建一个空字典
        dict_data = {}
        # 遍历二维数组
        for row in data:
            _, element, *rest = row  # 获取第二列元素及其后面的列
            # 更新字典中第二列元素为当前行
            dict_data[element] = [_, element, *rest]
        # 提取字典中的值，构建结果列表
        result = list(dict_data.values())
        return result

    def parase_data(self,data_file):
        data = []
        max_size = 0
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = line.split(':')
                max_size = max(len(values),max_size)
            for line in lines:
                values = line.split(':')
                data.append(values + [''] * (max_size - len(values)))
        return data
    def data_to_table_html(self,data, h_first = False, z_first = False):
        data_table_html = '<table>\n'
        for i,values in enumerate(data):
            data_table_html += '<tr>'
            for j,value in enumerate(values):
                if 'html' in value:
                    data_table_html += '<td><a href="{0}">{1}</a></td>'.format(value,value)
                    continue
                if h_first is True and i == 0:
                    data_table_html += '<th>{0}</th>'.format(value)
                else:
                    if z_first is True and j == 0:
                        data_table_html += '<th>{0}</th>'.format(value)
                    else:
                        data_table_html += '<td>{0}</td>'.format(value)
            data_table_html += '</tr>\n'
        data_table_html += '</table>'
        return data_table_html

    def generate_html(self, input_path, benchmark_path = None):
        result_file = os.path.join(input_path, '0_result.txt')
        info_file = os.path.join(input_path, '0_info.txt')
        processed_file = os.path.join(input_path, '0_result_processed.txt')
        html_file = os.path.join(input_path, 'result.html')
        if benchmark_path is not None:
            benchmark_file = os.path.join(benchmark_path, 'benchmark.txt')

        # 解析0_result.txt
        title = []
        results = []
        with open(result_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                timestamp = lines[i].strip('#').strip().split()[0]
                file_name = lines[i].split()[1]
                values = lines[i+1].split()
                image_names = lines[i+2].split()
                results.append({
                    'timestamp': timestamp,
                    'file_name': file_name,
                    'values': values,
                    'image_names': image_names
                })
            if self.sort_with_name:
                sorted_results = sorted(results[1:], key=lambda x: float(x['file_name']), reverse=False)
            else :
                sorted_results = sorted(results[1:], key=lambda x: float(x['values'][0]), reverse=True) # sort with first value
        #print(results)
        # 生成表格
        if len(results) == 0:
            return

        table_html = '<table>\n'
        table_html += '<tr>'
        table_html += '<th>file_name</th>'
        for value_name in results[0]['values']:
            table_html += '<th>{0}</th>'.format(value_name)
        for img_name in results[0]['image_names']:
            table_html += '<th>{0}</th>'.format(img_name)
        table_html += '</tr>\n'

        for result in sorted_results:
            table_html += '<tr>'
            table_html += '<th>{0}</th>'.format(result['file_name'])
            for value_name in result['values']:
                table_html += '<td>{0:.3f}</td>'.format(float(value_name))
            for img_name in result['image_names']:
                table_html += '<td><a href="{0}" data-lightbox="image-group" target="_blank" rel="noopener"><img src="{0}" width="300"></a></td>'.format(img_name)
            table_html += '</tr>\n'
        table_html += '</table>'

        # 统计均值并写入0_result_processed.txt
        processed_content = ""
        value_size = len(results[0]['values'])
        for i in range(value_size):
            values = [float(result['values'][i]) for result in results[1:]]
            values_mean = sum(values)/len(values)
            processed_content += 'mean_{0}: {1:.3f}\n'.format(results[0]['values'][i],values_mean)


        with open(processed_file, 'w') as f:
            f.write(processed_content)

        # 解析0_result_processed.txt
        processed_data = self.parase_data(processed_file)
        processed_table_html = self.data_to_table_html(processed_data, z_first = True)
        info_data = self.parase_data(info_file)
        info_table_html = self.data_to_table_html(info_data,  z_first = True)

        if benchmark_path is not None:
            benchmark_content = '{0}:{1}:'.format(info_data[0][1].replace("\n", ""),info_data[1][1].replace("\n", ""))
            for values in processed_data:
                benchmark_content += '{0:.3f}:'.format(float(values[1]))
            benchmark_content += '{0}\n'.format(self.get_relpath(html_file,benchmark_path))
            print(benchmark_content)
            with open(benchmark_file, 'a') as f:
                f.write(benchmark_content)

        # 生成最终的HTML内容
        html_content = '''
        <html>
        <head>
            <style>
                h1 {{
                    text-align: center;
                }}
                table {{
                    table-layout: auto;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                }}
                th {{
                    font-weight: bold;
                    /*background-color: #D0D0D0; /* 横坐标表头的背景颜色 */
                    /*max-width: 1000;*/
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h1>{0}评估</h1>
            <h2>评估结果</h2>
            {1}
            <h2>评估参数</h2>
            {2}
            <h2>详细数据</h2>
            {3}
        </body>
        </html>
        '''.format(self.title1, processed_table_html, info_table_html, table_html)

        # 将HTML内容写入index.html文件
        with open(html_file, 'w') as f:
            f.write(html_content)
        print(f"评估已完成，生成在:{html_file}")

    def benchmark_html(self,benchmark_path):

        benchmark_file = os.path.join(benchmark_path, 'benchmark.txt')
        benchmark_html_file = os.path.join(benchmark_path, 'benchmark.html')

        # 解析 benchmark.txt

        # 生成表格
        benchmark_data = self.parase_data(benchmark_file)
        benchmark_data = self.remove_repeat_line(benchmark_data)
        benchmark_data = [benchmark_data[0]] + sorted(benchmark_data[1:], key=lambda x: float(x[2]))
        benchmark_table_html = self.data_to_table_html(benchmark_data,h_first = True, z_first = True)
        # 生成最终的HTML内容
        html_content = '''
        <html>
        <head>
            <style>
                h1 {{
                    text-align: center;
                }}
                table {{
                    table-layout: auto;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                }}
                th {{
                    font-weight: bold;
                    /*background-color: #D0D0D0; /* 横坐标表头的背景颜色 */
                    /*max-width: 1000;*/
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h1>{0}BenchMark</h1>
            {1}
        </body>
        </html>
        '''.format(self.title1,benchmark_table_html)

        # 将HTML内容写入index.html文件
        with open(benchmark_html_file, 'w') as f:
            f.write(html_content)
        print(f"benchmark已更新，生成在:{benchmark_html_file}")


benchmark = BenchmarkBase("双目匹配", True)
# 获取命令行参数
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("usage: python benchmark_base.py eval_result/2023-10-19-17-07-40_MY-CUDA-SGM/ eval_result/")
    sys.exit(1)
input_path = sys.argv[1]
if len(sys.argv) == 2:
    benchmark.generate_html(input_path)
elif len(sys.argv) == 3:
    benchmark_path = sys.argv[2]
    benchmark.generate_html(input_path, benchmark_path)
    benchmark.benchmark_html(benchmark_path)
