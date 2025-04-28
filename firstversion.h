#pragma once
#include <queue>


//已知道，base_number为向量的数目，vecdim为向量的维度（向量的维度，每个向量由 vecdim 个浮动值组成），k是要求的最近前K个数据
//base基础数据集，类型为 float*，假设每个数据点是一个浮动的向量，base 数组存储了所有数据点的向量。
//query查询数据集，类型为 float*，假设每个数据点是一个浮动的向量，query 数组存储了查询数据点的向量。
std::priority_queue<std::pair<float, uint32_t> > flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;
    //返回一个优先队列，队列中存储的是成对的浮动值和索引。浮动值表示查询向量与某个数据点的相似度（这里是距离），而索引是该数据点在 base 中的位置。无符号int型

    //通过循环遍历 base 中的每个向量，计算每个向量与查询向量 query 之间的相似度。这里的 dis 是用来保存相似度的值。
    for(int i = 0; i < base_number; ++i) {
        float dis = 0;

        // DEEP100K数据集使用ip距离
        for(int d = 0; d < vecdim; ++d) {
            dis += base[d + i*vecdim]*query[d];
            //内积，向量计算
            //由于 base 是一个一维数组，而每个向量有 vecdim 维，所以 base[d + i * vecdim] 代表基础数据集中第 i 个向量的第 d 维分量。！一维数组
        }
        dis = 1 - dis;//保持翻转

        //使用最大堆，维护距离最近的前 k 个数据点
        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;//返回的数据类型为std::priority_queue<std::pair<float, uint32_t> > q;最大堆，且<float, uint32_t>，浮点数和无符号32位数，表示位置or索引
}