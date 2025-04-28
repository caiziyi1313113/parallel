import numpy as np
from sklearn.cluster import KMeans #调用现成的工具

#我们传入预处理后的数据，即将一维数据，一维数据的长度N*D，重塑为二维数组，形状为 (N, D)
def pq_train(data, M=12, Ks=256, save_path=r'files\codebook.npy'):
    N, D = data.shape  # 获取样本数量 N 和维度 D
    assert D % M == 0  # 检查维度是否能被 M 整除
    d = D // M  # 每个子向量的维度

    codebooks = []  # 用于存储每个子向量的聚类中心

    # 对每个子向量进行聚类
    for m in range(M):
        # 分割数据，得到第 m 段的子向量
        sub_vectors = data[:, m*d:(m+1)*d]
        
        # 使用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=Ks, n_init='auto').fit(sub_vectors)#auto 表示自动选择最佳的初始化次数
        codebooks.append(kmeans.cluster_centers_)
        

    # 保存 codebooks，shape 为 (M, Ks, d)，注意保存为numpy,大小为子空间数量*聚类中心数量*子空间维度（eg96那（4，256，24））
    np.save(save_path, np.array(codebooks))

if __name__ == '__main__':
    # 加载测试数据（base数据是一维数组）
    data = np.load('files/train.npy')  # shape: (N * D,) ，如何不需输入数据在cpp里封装
    
    # 将数据重塑为二维数组，形状为 (N, D)
    N = int(data.shape[0] // 96)  # 计算样本数量
    D = 96  # 每个样本的维度为 96
    data = data.reshape(N, D)  # 将数据重塑为 (N, D)
    
    # 调用 pq_train 函数进行训练
    pq_train(data, M=12, Ks=256, save_path='files/codebook2.npy')

#进行训练，得到码本