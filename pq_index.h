#pragma once

#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>

void load_pq_index(const std::string& filepath,
                   float**& codebooks,  // [M][Ks * d]
                   uint8_t**& codes,    // [N][M]
                   uint32_t& M,
                   uint32_t& Ks,
                   uint32_t& d,
                   uint32_t& N) {
    std::cout << "Loading PQ index from " << filepath << std::endl;
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return;
    }

    // 元信息
    in.read((char*)&M, sizeof(uint32_t));
    in.read((char*)&Ks, sizeof(uint32_t));
    in.read((char*)&d, sizeof(uint32_t));
    in.read((char*)&N, sizeof(uint32_t));

    // 分配码本内存
    codebooks = new float*[M];
    for (uint32_t m = 0; m < M; ++m) {
        codebooks[m] = new float[Ks * d];
        in.read(reinterpret_cast<char*>(codebooks[m]), Ks * d * sizeof(float));
    }
    std::cout << "Loaded " << M << " codebooks of size " << Ks << "x" << d << std::endl;

    // 分配编码内存
    codes = new uint8_t*[N];
    for (uint32_t n = 0; n < N; ++n) {
        codes[n] = new uint8_t[M];
        in.read(reinterpret_cast<char*>(codes[n]), M * sizeof(uint8_t));
    }
    std::cout << "Loaded " << N << " codes of size " << M << std::endl;

    in.close();
    std::cout << "Successfully loaded PQ index from " << filepath << std::endl;
}
