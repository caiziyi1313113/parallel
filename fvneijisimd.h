#pragma once
#include <queue>
#include <arm_neon.h>

std::priority_queue<std::pair<float, uint32_t> > flat_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for (size_t i = 0; i < base_number; ++i) {
        float32x4_t acc = vdupq_n_f32(0.0f);  // 累加器初始化为0
        //将一个标量 float 数值复制（duplicate）到一个 128-bit 的 float32x4_t 向量的所有 4 个元素中。

        // 每次处理4维，循环次数 = 96 / 4 = 24
        //已经知道数据维度，可以通过已有知识减少分支判断
        for (int d = 0; d < 96; d += 4) {
            //从内存中加载 4 个连续的 32-bit 浮点数（float）到一个 128-bit 的 NEON 向量（float32x4_t）中。
            float32x4_t vbase = vld1q_f32(base + i * 96 + d);
            float32x4_t vquery = vld1q_f32(query + d);
            acc = vmlaq_f32(acc, vbase, vquery);
            //vmlaq_f32 是 ARM NEON 的一个高效 向量乘加
            //执行两个 float 向量的逐元素乘法，并把结果加到另一个向量上：a + b * c
        }

        // 将acc中的4个值相加得到最终内积
        //对两个 64-bit 向量（包含两个 float32）进行逐元素相加。
        float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        float32x2_t sum1 = vpadd_f32(sum2, sum2);
        float dot = vget_lane_f32(sum1, 0);
        //从一个 float32x2_t 向量 vec 中，提取第 lane 个元素（lane只能是 0 或 1），返回对应的 float 标量。

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
