#pragma once

void free_pq_index(float** codebooks, uint8_t** codes, uint32_t M, uint32_t N) {
    for (uint32_t m = 0; m < M; ++m) {
        delete[] codebooks[m];
    }
    delete[] codebooks;

    for (uint32_t n = 0; n < N; ++n) {
        delete[] codes[n];
    }
    delete[] codes;
}