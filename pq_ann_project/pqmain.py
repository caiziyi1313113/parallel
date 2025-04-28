import numpy as np
from pq_train import pq_train
from pq_quantize import pq_quantize
from datachagne import changedata
# 加载数据
def load_fbin(file_path):
    with open(file_path, 'rb') as f:
        # 读取前两个 int32：n（样本数量），d（维度）
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]

        # 读取接下来的 n*d 个 float32 数据
        data = np.fromfile(f, dtype=np.float32, count=n*d)

    # 重塑为 (n, d) 的二维数组
    return data.reshape(n, d)

if __name__ == '__main__':
    file_path = r'anndata\DEEP100K.base.100k.fbin'
    data = load_fbin(file_path)
    pq_train(data)
    pq_quantize(data)
    changedata()
    print("Done!")
