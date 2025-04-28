#pragma once
#include <queue>
#include <arm_neon.h>

std::priority_queue<std::pair<float, uint32_t> > flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    constexpr int vec_blocks = 96 / 4;  // 每个block 4维，一共24组
    float32x4_t query_vec[vec_blocks];
    //我们知道数据类型为96维，4*32(32 个 128 位的 SIMD 寄存器)

    // 一次性加载 query 向量
    for (int j = 0; j < vec_blocks; ++j) {
        query_vec[j] = vld1q_f32(query + j * 4);//4个为一组存入寄存器，本来只需要96，而不需要128（注意对齐问题！后面考虑）
    }

    for (size_t i = 0; i < base_number; ++i) {
        float32x4_t acc = vdupq_n_f32(0.0f);  // 初始化累加器

        for (int j = 0; j < vec_blocks; ++j) {
            float32x4_t base_vec = vld1q_f32(base + i * 96 + j * 4);
            acc = vmlaq_f32(acc, base_vec, query_vec[j]);//该步骤是版本1和版本2的区别
        }

        // 累加 acc 向量为标量
        float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        float32x2_t sum1 = vpadd_f32(sum2, sum2);
        float dot = vget_lane_f32(sum1, 0);

        float dis = 1.0f - dot;

        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }

    return q;
}
