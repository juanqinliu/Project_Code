#ifndef BUFFERS_H
#define BUFFERS_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace samplesCommon
{
    // CUDA memory management template class
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
        GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            : mType(type), mBuffer(nullptr), mSize(0) {}

    GenericBuffer(size_t size, nvinfer1::DataType type)
            : mType(type), mBuffer(nullptr), mSize(size)
    {
            if (size)
        {
                allocate(size);
        }
    }

        ~GenericBuffer()
        {
            deallocate();
        }

        void allocate(size_t size)
        {
            if (mSize)
            {
                deallocate();
            }
            mSize = size;
            size_t nbBytes = sizeInBytes();
            if (nbBytes)
            {
                mBuffer = static_cast<void*>(AllocFunc()(nbBytes));
    }
        }

        void deallocate()
    {
            if (mBuffer)
        {
                FreeFunc()(mBuffer);
                mBuffer = nullptr;
                mSize = 0;
            }
        }

        size_t sizeInBytes() const
        {
            return mSize * elementSize();
    }

        size_t size() const
    {
            return mSize;
    }

    void* data()
    {
        return mBuffer;
    }

    const void* data() const
    {
        return mBuffer;
    }

        nvinfer1::DataType type() const
    {
            return mType;
    }

    private:
        size_t elementSize() const
    {
            switch (mType)
            {
            case nvinfer1::DataType::kFLOAT: return sizeof(float);
            case nvinfer1::DataType::kHALF: return sizeof(int16_t);
            case nvinfer1::DataType::kINT8: return sizeof(int8_t);
            case nvinfer1::DataType::kINT32: return sizeof(int32_t);
            case nvinfer1::DataType::kBOOL: return sizeof(bool);
    }
            return 0;
        }

    nvinfer1::DataType mType;
    void* mBuffer;
        size_t mSize;
};

    // Host memory allocator
    struct HostAllocator
{
        void* operator()(size_t size) const
    {
            return std::malloc(size);
    }
};

    // Host memory deallocator
    struct HostFree
{
    void operator()(void* ptr) const
    {
            std::free(ptr);
    }
};

    // Host memory buffer
    class HostBuffer : public GenericBuffer<HostAllocator, HostFree>
{
public:
        HostBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            : GenericBuffer<HostAllocator, HostFree>(type) {}

        HostBuffer(size_t size, nvinfer1::DataType type)
            : GenericBuffer<HostAllocator, HostFree>(size, type) {}
    };

    // CUDA device memory allocator
    struct DeviceAllocator
    {
        void* operator()(size_t size) const
    {
            void* ptr = nullptr;
            cudaMalloc(&ptr, size);
            return ptr;
    }
};

    // CUDA device memory deallocator
    struct DeviceFree
{
    void operator()(void* ptr) const
    {
            if (ptr)
            {
                cudaFree(ptr);
            }
    }
};

    // CUDA device memory buffer
    class DeviceBuffer : public GenericBuffer<DeviceAllocator, DeviceFree>
    {
    public:
        DeviceBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
            : GenericBuffer<DeviceAllocator, DeviceFree>(type) {}

        DeviceBuffer(size_t size, nvinfer1::DataType type)
            : GenericBuffer<DeviceAllocator, DeviceFree>(size, type) {}
    };

    // Buffer management class
    class BufferManager
    {
    public:
        BufferManager(nvinfer1::ICudaEngine* engine, const int batchSize = 1)
            : mEngine(engine), mBatchSize(batchSize)
        {
            // Allocate memory for all input and output buffers
            for (int i = 0; i < mEngine->getNbBindings(); ++i)
            {
                auto dims = mEngine->getBindingDimensions(i);
                size_t vol = volume(dims) * batchSize;
                nvinfer1::DataType type = mEngine->getBindingDataType(i);

                // Allocate memory on device
                mDeviceBindings.emplace_back(vol, type);

                if (mEngine->bindingIsInput(i))
                {
                    // Also allocate host memory for inputs
                    mHostBindings.emplace_back(vol, type);
                }
                else
                {
                    // Allocate host memory for outputs
                    mHostBindings.emplace_back(vol, type);
                }
            }
        }

        // Get device memory binding pointers
        std::vector<void*> getDeviceBindings() const
        {
            std::vector<void*> deviceBindings;
            for (const auto& buffer : mDeviceBindings)
            {
                deviceBindings.push_back(const_cast<void*>(buffer.data()));
            }
            return deviceBindings;
        }

        // Get input buffer
        template <typename T>
        T* getHostInput(const int index)
        {
            return static_cast<T*>(mHostBindings[index].data());
        }

        // Get output buffer
        template <typename T>
        T* getHostOutput(const int index)
        {
            return static_cast<T*>(mHostBindings[mEngine->getNbBindings() - 1 + index].data());
        }

        // Copy from host to device
        void copyInputToDevice()
        {
            for (int i = 0; i < mEngine->getNbBindings(); ++i)
            {
                if (mEngine->bindingIsInput(i))
                {
                    size_t size = mHostBindings[i].sizeInBytes();
                    cudaMemcpy(mDeviceBindings[i].data(), mHostBindings[i].data(), size, cudaMemcpyHostToDevice);
                }
            }
        }

        // Copy from device to host
        void copyOutputToHost()
        {
            for (int i = 0; i < mEngine->getNbBindings(); ++i)
            {
                if (!mEngine->bindingIsInput(i))
                {
                    size_t size = mDeviceBindings[i].sizeInBytes();
                    cudaMemcpy(mHostBindings[i].data(), mDeviceBindings[i].data(), size, cudaMemcpyDeviceToHost);
                }
            }
        }

        // Calculate total volume
        static size_t volume(const nvinfer1::Dims& d)
        {
            size_t vol = 1;
            for (int i = 0; i < d.nbDims; ++i)
            {
                vol *= d.d[i];
            }
            return vol;
        }

    private:
        std::vector<DeviceBuffer> mDeviceBindings;
        std::vector<HostBuffer> mHostBindings;
        nvinfer1::ICudaEngine* mEngine{nullptr};
        int mBatchSize{1};
    };
}

#endif // BUFFERS_H 