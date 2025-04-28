#pragma once
#include <queue>
#include <utility>
#include <cstdint>
#include <cmath>

// 进行搜索，返回 topk 个结果
std::priority_queue<std::pair<float, uint32_t>> pq_adc_search(
    float** codebooks,           // [M][Ks * d]
    uint8_t** pq_codes,          // [N][M]
    float* query,                // [query_size]
    uint32_t M,
    uint32_t Ks,
    uint32_t d,
    uint32_t N,
    uint32_t topk) {

    // Step 1: 构建距离表 distance_table[M][Ks]
    float** distance_table = new float*[M];
    for (uint32_t m = 0; m < M; ++m) {
        distance_table[m] = new float[Ks];
        for (uint32_t k = 0; k < Ks; ++k) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < d; ++i) {
                dot += query[m * d + i] * codebooks[m][k * d + i];
            }
            distance_table[m][k] = 1.0f - dot;
        }
    }

    // Step 2~4: 计算每个样本的距离并维护 topk 堆
    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (uint32_t i = 0; i < N; ++i) {
        float dis = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            uint8_t cid = pq_codes[i][m];
            dis += distance_table[m][cid];
        }

        if (heap.size() < topk) heap.emplace(dis, i);
        else if (dis < heap.top().first) {
            heap.emplace(dis, i);
            heap.pop();
        }
    }

    // 释放 distance_table
    for (uint32_t m = 0; m < M; ++m) {
        delete[] distance_table[m];
    }
    delete[] distance_table;

    return heap;
}
