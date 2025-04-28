import numpy as np
import struct

def changedata():
        # 加载 .npy 文件

    codebooks = np.load("files/codebook.npy")  # shape = (M, Ks, d)
    codes = np.load("files/pq_codes.npy")          # shape = (N, M), dtype=uint8

    M, Ks, d = codebooks.shape
    N = codes.shape[0]

    # 保存为二进制格式
    with open("files/pq_index2.bin", "wb") as f:
        f.write(struct.pack("I", M))
        f.write(struct.pack("I", Ks))
        f.write(struct.pack("I", d))
        f.write(struct.pack("I", N))
    
        # codebooks: float32
        f.write(codebooks.astype(np.float32).tobytes())

        # codes: uint8
        f.write(codes.astype(np.uint8).tobytes())
        #二进制文件内容，为四个数据m、k、d、n，然后是codebook(M, Ks, d)和codeindex(N, M)