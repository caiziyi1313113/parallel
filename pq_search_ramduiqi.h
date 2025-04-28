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
    //外层遍历每一个样本
    for (uint32_t i = 0; i < N; ++i) {
        const uint8_t* codes = pq_codes[i];//得到每一个样本的code_index
        //循环展开计算八个子空间的ADC距离
        //？是否还可以优化？同时处理多个向量？
        float dis = 
            distance_table[0][codes[0]] +
            distance_table[1][codes[1]] +
            distance_table[2][codes[2]] +
            distance_table[3][codes[3]] +
            distance_table[4][codes[4]] +
            distance_table[5][codes[5]] +
            distance_table[6][codes[6]] +
            distance_table[7][codes[7]];
        
        if (coarse_heap.size() < topp) {
            coarse_heap.emplace(dis, i);
        } else if (dis < coarse_heap.top().first) {
            coarse_heap.pop();
            coarse_heap.emplace(dis, i);
        }
    }

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
    struct alignas(64) QueryBlocks {  // 64字节对齐
        float32x4_t block[6][4];      // 96维 = 6块×16元素
    };
    
    // 预加载查询向量到寄存器块
    QueryBlocks q_blocks;
    for (uint32_t b = 0; b < 6; ++b) {
        const float* q_ptr = query + b*16;
        q_blocks.block[b][0] = vld1q_f32(q_ptr);
        q_blocks.block[b][1] = vld1q_f32(q_ptr+4);
        q_blocks.block[b][2] = vld1q_f32(q_ptr+8);
        q_blocks.block[b][3] = vld1q_f32(q_ptr+12);
    }

    std::priority_queue<std::pair<float, uint32_t>> fine_heap;
    for (uint32_t i = 0; i < candidate_count; ++i) {
        const float* vec = original_vectors + candidates[i] * TOTAL_DIM;
        float32x4_t sum = vdupq_n_f32(0.0f);
        
        // 处理6个数据块
        for (uint32_t b = 0; b < 6; ++b) {
            const float* v_ptr = vec + b*16;
            
            // 加载向量块
            const float32x4_t v0 = vld1q_f32(v_ptr);
            const float32x4_t v1 = vld1q_f32(v_ptr+4);
            const float32x4_t v2 = vld1q_f32(v_ptr+8);
            const float32x4_t v3 = vld1q_f32(v_ptr+12);
            
            // SIMD乘加计算
            const float32x4_t p0 = vmulq_f32(v0, q_blocks.block[b][0]);
            const float32x4_t p1 = vmulq_f32(v1, q_blocks.block[b][1]);
            const float32x4_t p2 = vmulq_f32(v2, q_blocks.block[b][2]);
            const float32x4_t p3 = vmulq_f32(v3, q_blocks.block[b][3]);
            
            // 累加结果
            sum = vaddq_f32(sum, vaddq_f32(
                vaddq_f32(p0, p1),
                vaddq_f32(p2, p3)
            ));
        }
        
        // 最终计算结果
        const float exact_dot = 1.0f - vaddvq_f32(sum);
        
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