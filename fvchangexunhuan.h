#pragma once
#include <arm_neon.h>
#include <queue>
#include <cstdlib>  // 用于aligned_alloc

std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query,
    size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    constexpr size_t BLOCK_SIZE = 16;  // 每次处理16个元素
    constexpr size_t NUM_BLOCKS = 96 / BLOCK_SIZE; // 6 blocks
    constexpr size_t ALIGNMENT = 64;   // 缓存行对齐

    // 预加载所有query块到寄存器数组
    struct alignas(ALIGNMENT) QueryBlock {
        float32x4_t q[4]; // 每个块包含4个向量
    };
    QueryBlock query_blocks[NUM_BLOCKS];

    // 阶段1：预加载query数据
    for (size_t b = 0; b < NUM_BLOCKS; ++b) {
        const size_t offset = b * BLOCK_SIZE;
        query_blocks[b].q[0] = vld1q_f32(query + offset);
        query_blocks[b].q[1] = vld1q_f32(query + offset + 4);
        query_blocks[b].q[2] = vld1q_f32(query + offset + 8);
        query_blocks[b].q[3] = vld1q_f32(query + offset + 12);
    }

    // 阶段2：使用对齐内存分配距离数组
    float* distances = static_cast<float*>(aligned_alloc(ALIGNMENT, base_number * sizeof(float)));
    std::memset(distances, 0, base_number * sizeof(float));  // 初始化为0

    // 分块处理循环
    for (size_t b = 0; b < NUM_BLOCKS; ++b) {
        const auto& q_block = query_blocks[b];
        
        for (size_t i = 0; i < base_number; ++i) {
            const float* vec = base + i * vecdim + b * BLOCK_SIZE;
            
            // 加载当前向量块（确保内存对齐访问）
            const float32x4_t v0 = vld1q_f32(vec);
            const float32x4_t v1 = vld1q_f32(vec + 4);
            const float32x4_t v2 = vld1q_f32(vec + 8);
            const float32x4_t v3 = vld1q_f32(vec + 12);

            // SIMD乘加计算
            const float32x4_t p0 = vmulq_f32(v0, q_block.q[0]);
            const float32x4_t p1 = vmulq_f32(v1, q_block.q[1]);
            const float32x4_t p2 = vmulq_f32(v2, q_block.q[2]);
            const float32x4_t p3 = vmulq_f32(v3, q_block.q[3]);

            // 累加计算结果
            const float32x4_t sum = vaddq_f32(vaddq_f32(p0, p1), vaddq_f32(p2, p3));
            distances[i] += vaddvq_f32(sum);
        }
    }

    // 阶段3：结果处理
    for (size_t i = 0; i < base_number; ++i) {
        const float dis = 1.0f - distances[i];
        
        if (q.size() < k) {
            q.emplace(dis, static_cast<uint32_t>(i));
        } else if (dis < q.top().first) {
            q.pop();
            q.emplace(dis, static_cast<uint32_t>(i));
        }
    }

    free(distances);  // 释放对齐内存
    return q;
}