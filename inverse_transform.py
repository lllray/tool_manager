import numpy as np

matrix = np.array([
  [0.9999200973572613, -0.012442932184760516, 0.002229874366319243, -0.10011948786043312],
  [0.012464719999762171, 0.9998719169985854, -0.010038941821753238, -0.0003533052811685481],
  [-0.0021046748850227963, 0.010065934443382201, 0.9999471222556765, 0.000932310156861502],
  [0.0, 0.0, 0.0, 1.0]
])

inverse_matrix = np.linalg.inv(matrix)

# 格式化输出逆矩阵
formatted_inverse_matrix = ['{:e}'.format(x) for x in inverse_matrix.flatten()]

# 将逆矩阵按照原始格式重新组织
output = ', '.join(formatted_inverse_matrix)

print(output)
