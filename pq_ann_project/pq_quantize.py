import numpy as np

def pq_quantize(data, codebook_path=r'files\codebook.npy', save_path=r'files\pq_codes.npy'):
    codebooks = np.load(codebook_path)  # shape: (M, Ks, d)，大小为子空间数量*聚类中心数量*子空间维度（eg96那（4，256，24））
    M, Ks, d = codebooks.shape
    N, D = data.shape
    assert D == M * d

    codes = np.empty((N, M), dtype=np.uint8)#由于每个字空间256个聚类，所以用uint8表示，码本索引的大小为(N,M)即样本数量，子空间数量

    for m in range(M):
        sub_vectors = data[:, m*d:(m+1)*d]
        centers = codebooks[m]  # shape: (Ks, d)，第m个子空间的所有聚类中心
        dists = np.dot(sub_vectors, centers.T)  # shape: (N, Ks)，表示每个子向量与每个中心的内积
        codes[:, m] = np.argmax(dists, axis=1)  # 内积越大越相似，取最大值，取出最大内积对应的中心编号，越大表示越相似
        

    np.save(save_path, codes)

if __name__ == '__main__':
    data = np.load('files/train2.npy')
    pq_quantize(data)

#编写main函数，串联数据
#进行量化，得到码本索引