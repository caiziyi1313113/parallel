#pragma once
#include <queue>
#include <utility>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>

// 预定义常量（根据已知参数）
constexpr uint32_t SUBSPACE_DIM = 12;//子空间的维度
constexpr uint32_t TOTAL_DIM = 96;//维度
constexpr uint32_t SUBSPACE_NUM = 8;//子空间的数目

std::priority_queue<std::pair<float, uint32_t>> pq_adc_search(
    float** codebooks,
    uint8_t** pq_codes,
    float* query,
    float* original_vectors,
    uint32_t  M ,       // 固定为8之后可能会编程12（12*8=96维）
    uint32_t Ks,
    uint32_t d ,       // 固定为12
    uint32_t N,
    uint32_t topk,
    uint32_t topp)
{
    /************ 第一阶段：粗筛 topp 个候选 ************/
    
    // Step 1: 构建距离表 [8][Ks]
    //加载一次uery的m
    float** distance_table = new float*[SUBSPACE_NUM];
    for (uint32_t m = 0; m < SUBSPACE_NUM; ++m) {
        distance_table[m] = new float[Ks];
        const float* query_sub = query + m * SUBSPACE_DIM;

        // NEON优化子空间内积计算（手动展开循环）
        //我们知道codebook'的形状为（m，ks*d)
        //计算每个子空间的内积
        for (uint32_t k = 0; k < Ks; ++k) {
            const float* code_sub = codebooks[m] + k * SUBSPACE_DIM;
            //float32x4_t 是一个 128 位的寄存器，它可以容纳 4 个 32 位浮点数
            float32x4_t sum = vdupq_n_f32(0.0f);
            
            // 手动展开3次循环处理12维
            float32x4_t q0 = vld1q_f32(query_sub);
            float32x4_t c0 = vld1q_f32(code_sub);
            sum = vmlaq_f32(sum, q0, c0);
            //vmlaq_f32 是一个 NEON 指令，表示 "vector multiply-accumulate"（向量乘法加法）。

            q0 = vld1q_f32(query_sub + 4);
            c0 = vld1q_f32(code_sub + 4);
            sum = vmlaq_f32(sum, q0, c0);
            
            q0 = vld1q_f32(query_sub + 8);
            c0 = vld1q_f32(code_sub + 8);
            sum = vmlaq_f32(sum, q0, c0);
            
            // 水平求和
            //果 sum = [a, b, c, d]，那么 vaddvq_f32(sum) 会执行以下操作：result=a+b+c+d
            //返回一个标量浮点数，表示 sum 向量所有元素的总和。

            distance_table[m][k] = 1.0f - vaddvq_f32(sum);
        }
    }

    // Step 2: ADC距离计算（手动展开8个子空间）
    //主要是对求和的优化
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;
    
    // 初始化dis数组，并确保内存对齐
    float* dis = static_cast<float*>(aligned_alloc(16, N * sizeof(float)));
    memset(dis, 0, N * sizeof(float));

    // 外层循环遍历每个m
    for (uint32_t m = 0; m < M; ++m) {
        const float* dt_m = distance_table[m]; // 当前m对应的距离表

        // 每次处理4个i
        uint32_t i = 0;
        for (; i + 3 < N; i += 4) {
            // 获取四个i对应的code
            uint32_t c0 = pq_codes[i][m];
            uint32_t c1 = pq_codes[i+1][m];
            uint32_t c2 = pq_codes[i+2][m];
            uint32_t c3 = pq_codes[i+3][m];

            // 加载四个distance值
            float d0 = dt_m[c0];
            float d1 = dt_m[c1];
            float d2 = dt_m[c2];
            float d3 = dt_m[c3];

            // 组合成NEON向量
            float32x4_t dv = {d0, d1, d2, d3};

            // 加载并累加当前dis
            float32x4_t disv = vld1q_f32(dis + i);
            disv = vaddq_f32(disv, dv);
            vst1q_f32(dis + i, disv);
        }

    }
        
        // 构建优先队列
        for (uint32_t i = 0; i < N; ++i) {
            if (coarse_heap.size() < topp) {
                coarse_heap.emplace(dis[i], i);
            } else if (dis[i] < coarse_heap.top().first) {
                coarse_heap.pop();
                coarse_heap.emplace(dis[i], i);
            }
        }
        free(dis); // 释放内存
    
    // 清理距离表
    for (uint32_t m = 0; m < SUBSPACE_NUM; ++m) delete[] distance_table[m];
    delete[] distance_table;

    /************ 第二阶段：精筛 topk 个结果 ************/
    
    // 提取候选索引
    const uint32_t candidate_count = coarse_heap.size();
    uint32_t* candidates = new uint32_t[candidate_count];
    for (int i = candidate_count-1; i >= 0; --i) {
        candidates[i] = coarse_heap.top().second;
        coarse_heap.pop();
    }

    // Step 3: 精确搜索（NEON优化）
    std::priority_queue<std::pair<float, uint32_t>> fine_heap;
    constexpr size_t block_size = 16; // 每次处理16个元素，分成6块

    // 外层循环：遍历候选样本数
    for (size_t i = 0; i < candidate_count; ++i) {
        const uint32_t idx = candidates[i];
        // 初始化 NEON 向量
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);
    
        // 获取当前候选样本的起始地址
        const float* base_ptr = original_vectors + idx * D;
    
        // 内层循环：遍历每 16 个元素
        for (size_t d = 0; d < D; d += block_size) {
            // 从查询向量加载 16 个元素
            float32x4_t q1 = vld1q_f32(query + d);
            float32x4_t q2 = vld1q_f32(query + d + 4);
            float32x4_t q3 = vld1q_f32(query + d + 8);
            float32x4_t q4 = vld1q_f32(query + d + 12);
    
            // 从样本向量加载 16 个元素
            float32x4_t b1 = vld1q_f32(base_ptr + d);
            float32x4_t b2 = vld1q_f32(base_ptr + d + 4);
            float32x4_t b3 = vld1q_f32(base_ptr + d + 8);
            float32x4_t b4 = vld1q_f32(base_ptr + d + 12);
    
            // 使用 NEON 指令进行逐元素内积计算
            sum1 = vmlaq_f32(sum1, b1, q1);
            sum2 = vmlaq_f32(sum2, b2, q2);
            sum3 = vmlaq_f32(sum3, b3, q3);
            sum4 = vmlaq_f32(sum4, b4, q4);
        }
    
        // 合并结果：水平加法
        sum1 = vaddq_f32(sum1, sum2);
        sum3 = vaddq_f32(sum3, sum4);
        sum1 = vaddq_f32(sum1, sum3);
    
        // 水平相加 sum1 中的四个元素
        float32x2_t sum_low = vget_low_f32(sum1);
        float32x2_t sum_high = vget_high_f32(sum1);
        sum_low = vadd_f32(sum_low, sum_high);
        
        // 获取最终内积值并计算距离
        float exact_dot = vget_lane_f32(vpadd_f32(sum_low, sum_low), 0);
        exact_dot = 1.0f - exact_dot; // 计算距离（1 - 内积）
        
        // 维护堆
        if (fine_heap.size() < topk) {
            fine_heap.emplace(exact_dot, candidates[i]);
        } else if (exact_dot < fine_heap.top().first) {
            fine_heap.pop();
            fine_heap.emplace(exact_dot, candidates[i]);
        }
    }
    
    delete[] candidates;
    return fine_heap;
}