#include "nvtt_gpu.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA objects must outlive the kernel that reads them, including on error paths.
struct DecodeResources
{
    cudaArray_t blocks = nullptr;
    cudaTextureObject_t texture = 0;
    ~DecodeResources()
    {
        if (texture) cudaDestroyTextureObject(texture);
        if (blocks) cudaFreeArray(blocks);
    }
};

bool CheckCuda(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
        return true;
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

// NVTT uses planar float channels. Keep float precision, including BC6H HDR values.
__global__ void CopyDecodedPixels(cudaTextureObject_t texture, float* output, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const float4 pixel = tex2D<float4>(texture, x + 0.5f, y + 0.5f);
    const size_t index = size_t(y) * width + x;
    const size_t plane = size_t(width) * height;
    output[index] = pixel.x;
    output[plane + index] = pixel.y;
    output[2 * plane + index] = pixel.z;
    output[3 * plane + index] = pixel.w;
}

// Follows nvtt_samples/cuda_input: BC array -> texture view -> texel fetch kernel.
// This is a CUDA hardware decoder, not an NVTT GPU setImage2D overload.
bool DecompressGpu(nvtt::Format format, int width, int height,
                   const void* blocks, nvtt::Surface& image)
{
    cudaResourceViewDesc view{};
    size_t block_bytes = 16;
    switch (format)
    {
    case nvtt::Format_BC1: case nvtt::Format_BC1a:
        view.format = cudaResViewFormatUnsignedBlockCompressed1; block_bytes = 8; break;
    case nvtt::Format_BC2: view.format = cudaResViewFormatUnsignedBlockCompressed2; break;
    case nvtt::Format_BC3: view.format = cudaResViewFormatUnsignedBlockCompressed3; break;
    case nvtt::Format_BC4:
        view.format = cudaResViewFormatUnsignedBlockCompressed4; block_bytes = 8; break;
    case nvtt::Format_BC5: view.format = cudaResViewFormatUnsignedBlockCompressed5; break;
    case nvtt::Format_BC6U: view.format = cudaResViewFormatUnsignedBlockCompressed6H; break;
    case nvtt::Format_BC6S: view.format = cudaResViewFormatSignedBlockCompressed6H; break;
    case nvtt::Format_BC7: view.format = cudaResViewFormatUnsignedBlockCompressed7; break;
    default:
        std::cerr << "Unsupported GPU decode format.\n";
        return false;
    }

    // Allocate the working Surface without decoding or uploading CPU float pixels.
    if (!image.setImage(width, height, 1))
        return false;
    image.ToGPU(nullptr, false);
    float* output = image.gpuDataMutable();
    if (!output)
    {
        std::cerr << "NVTT could not allocate the GPU decode surface.\n";
        return false;
    }

    DecodeResources resources;
    const size_t columns = (width + 3) / 4;
    const size_t rows = (height + 3) / 4;
    cudaChannelFormatDesc channels{};
    channels.x = channels.y = 32;
    if (block_bytes == 16) channels.z = channels.w = 32;
    channels.f = cudaChannelFormatKindUnsigned;
    if (!CheckCuda(cudaMallocArray(&resources.blocks, &channels, columns, rows), "BC array allocation"))
        return false;
    const size_t pitch = columns * block_bytes;
    if (!CheckCuda(cudaMemcpy2DToArray(resources.blocks, 0, 0, blocks, pitch,
                                      pitch, rows, cudaMemcpyHostToDevice), "BC block upload"))
        return false;

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = resources.blocks;
    cudaTextureDesc texture{};
    texture.addressMode[0] = texture.addressMode[1] = cudaAddressModeClamp;
    texture.filterMode = cudaFilterModePoint;
    texture.readMode = cudaReadModeElementType;
    // Use raw channel values even for sRGB-tagged DDS, matching the current CPU baseline.
    view.width = columns * 4;
    view.height = rows * 4;
    if (!CheckCuda(cudaCreateTextureObject(&resources.texture, &resource, &texture, &view), "BC texture view"))
        return false;

    const dim3 threads(16, 16);
    const dim3 grid((width + 15) / 16, (height + 15) / 16);
    CopyDecodedPixels<<<grid, threads>>>(resources.texture, output, width, height);
    if (!CheckCuda(cudaGetLastError(), "Decode kernel launch"))
        return false;
    // Include actual completion in decode time and keep resources alive until it finishes.
    return CheckCuda(cudaDeviceSynchronize(), "GPU decode completion");
}
