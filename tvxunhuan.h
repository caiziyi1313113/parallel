#include <arm_neon.h>
#include <queue>
#include <algorithm>


std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {

    std::priority_queue<std::pair<float, uint32_t>> q;

    // 确保vecdim是96，并且是16的倍数，这里假设vecdim=96

    constexpr size_t block_size = 16; 
    //外层循环样本个数，内层循环是对单个96维的向量进行处理
    for (size_t i = 0; i < base_number; ++i) {
    // 每次处理16个元素，分成6块（96/16=6）
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);

    float* base_ptr = base + i * vecdim;
    //循环展开，处理16维的四个寄存器
    for (size_t d = 0; d < vecdim; d += block_size) {

    // 每次处理16个元素，分成4个4元素块
    //获取测试数据的对应板块
        float32x4_t q1 = vld1q_f32(query + d);
        float32x4_t q2 = vld1q_f32(query + d + 4);
        float32x4_t q3 = vld1q_f32(query + d + 8);
        float32x4_t q4 = vld1q_f32(query + d + 12);
    //获取样例数据的对应板块
        float32x4_t b1 = vld1q_f32(base_ptr + d);
        float32x4_t b2 = vld1q_f32(base_ptr + d +4);
        float32x4_t b3 = vld1q_f32(base_ptr + d +8);
        float32x4_t b4 = vld1q_f32(base_ptr + d +12);


        sum1 = vmlaq_f32(sum1, b1, q1);
        sum2 = vmlaq_f32(sum2, b2, q2);
        sum3 = vmlaq_f32(sum3, b3, q3);
        sum4 = vmlaq_f32(sum4, b4, q4);
    }   
// 合并四个sum的结果
    sum1 = vaddq_f32(sum1, sum2);
    sum3 = vaddq_f32(sum3, sum4);
    sum1 = vaddq_f32(sum1, sum3);
// 水平相加sum1中的四个元素
    float32x2_t sum_low = vget_low_f32(sum1);
    float32x2_t sum_high = vget_high_f32(sum1);
    sum_low = vadd_f32(sum_low, sum_high);
    float dis = vget_lane_f32(vpadd_f32(sum_low, sum_low), 0);
    dis = 1.0f - dis;

    // 维护优先队列

    if (q.size() < k) {

        q.emplace(dis, i);

    } else if (dis < q.top().first) {
        q.emplace(dis, i);
        q.pop();
    }
    }

return q;

}