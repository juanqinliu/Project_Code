#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <vector>
#include <memory>

namespace samplesCommon
{
    // Calculate tensor volume (number of elements)
    inline int volume(const nvinfer1::Dims& d)
{
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>());
}

    // Create CUDA stream
    inline cudaStream_t makeCudaStream()
{
    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    {
        return nullptr;
    }
    return stream;
}

    // Read file to buffer
    inline std::vector<uint8_t> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file)
        {
            return {};
        }
        
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        
        return buffer;
    }
}

#endif // COMMON_H 