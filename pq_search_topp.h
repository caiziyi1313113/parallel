#pragma once
#include <queue>
#include <utility>
#include <cstdint>
#include <algorithm>

std::priority_queue<std::pair<float, uint32_t>> pq_adc_search(
    float** codebooks,
    uint8_t** pq_codes,
    float* query,//传入的待测数据
    float* original_vectors,  // 原始数组，base
    uint32_t M,//子空间数目
    uint32_t Ks,//聚类中心数目
    uint32_t d,//每个子空间的维度？
    uint32_t N,//样本数目
    uint32_t topk,//二轮筛选
    uint32_t topp)//一轮筛选
{
    /************ 第一阶段：粗筛 topp 个候选 ************/
    
    // Step 1: 构建距离表 [M][Ks]，预处理
    float** distance_table = new float*[M];
    for (uint32_t m = 0; m < M; ++m) {
        distance_table[m] = new float[Ks];
        for (uint32_t k = 0; k < Ks; ++k) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < d; ++i) {
                dot += query[m*d + i] * codebooks[m][k*d + i];
            }
            distance_table[m][k] = 1.0f - dot;
        }
    }

    // Step 2: 计算ADC距离并维护最大堆
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;
    for (uint32_t i = 0; i < N; ++i) {
        float dis = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            dis += distance_table[m][pq_codes[i][m]];
        }
        
        if (coarse_heap.size() < topp) {
            coarse_heap.emplace(dis, i);
        } else if (dis < coarse_heap.top().first) {
            coarse_heap.pop();
            coarse_heap.emplace(dis, i);
        }
    }

    // 清理距离表
    for (uint32_t m = 0; m < M; ++m) delete[] distance_table[m];
    delete[] distance_table;

    /************ 第二阶段：精筛 topk 个结果 ************/
    
    // 提取并反转候选索引
    const uint32_t candidate_count = coarse_heap.size();
    uint32_t* candidates = new uint32_t[candidate_count];
    for (int i = candidate_count-1; i >= 0; --i) { // 反向填充实现反转，因为目前是最大堆，距离最远的在最前面
        candidates[i] = coarse_heap.top().second;//记录索引
        coarse_heap.pop();
    }

    // Step 3: 精确计算原始向量内积
    const uint32_t D = M * d;//维度vidimension
    std::priority_queue<std::pair<float, uint32_t>> fine_heap;
    
    for (uint32_t i = 0; i < candidate_count; ++i) {
        const uint32_t idx = candidates[i];
        float exact_dot = 0.0f;
        for (uint32_t j = 0; j < D; ++j) {//原始方法进行内积计算
            exact_dot += original_vectors[idx * D + j] * query[j];
        }
        exact_dot=1-exact_dot;
        if (fine_heap.size() < topk) {
            fine_heap.emplace(exact_dot, idx);
        } else if (exact_dot < fine_heap.top().first) {
            fine_heap.pop();
            fine_heap.emplace(exact_dot, idx);
        }
    }

    delete[] candidates; // 释放候选数组

    return fine_heap;
}