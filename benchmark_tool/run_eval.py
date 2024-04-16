import sys
import os
from benchmark_base import BenchmarkBase

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
