// Compressonator benchmark using its own DDS I/O, mip generator, and codecs.
// CPU decoding and Box filtering are used for both CPU and DXC compression runs.
// Only lower mips are compressed; Mip 0 is copied without changing its blocks.
// IMPORTANT: ProcessFile currently has its SaveDDS call commented out.
// A successful run processes the image but does not create an output DDS.

#include <compressonator.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "common/timer.h"

// The SDK merges sRGB and UNORM formats, so retain the optional DX10 metadata.
struct DDSMetadata
{
    bool dx10 = false;
    std::array<uint32_t, 5> extension{};
};

// Read only enough DDS metadata to validate the supported layout and retain DX10 tags.
// The SDK still loads the actual image. This small read does not replace its DDS loader.
// Offsets below include the four-byte DDS magic: 128 legacy bytes + 20 DX10 bytes.
bool ReadDDSMetadata(const std::filesystem::path &path, DDSMetadata &metadata)
{
    std::ifstream file(path, std::ios::binary);
    // Word 0 is the DDS magic, 3/4 are height/width, 7 is mip count,
    // 19 is pixel-format size, 21 is FourCC, and 28 contains secondary caps.
    std::array<uint32_t, 32> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!file || header[0] != 0x20534444 || header[1] != 124 || header[19] != 32)
        return false;
    // Validate the input layout before the SDK can flatten unsupported arrays.
    if (header[3] == 0 || header[4] == 0
        || header[4] > std::numeric_limits<CMP_DWORD>::max() / 16 / header[3]
        || ((header[2] & 0x20000) && header[7] != 1)
        || (header[28] & 0x200200)) // Volume or cubemap.
        return false;
    metadata.dx10 = header[21] == CMP_MAKEFOURCC('D', 'X', '1', '0');
    if (!metadata.dx10)
        return true;
    file.read(reinterpret_cast<char*>(metadata.extension.data()), sizeof(metadata.extension));
    if (!file || metadata.extension[1] != 3 || metadata.extension[2] & 4 || metadata.extension[3] != 1)
        return false;
    // Typeless BC formats have no defined numeric interpretation for this benchmark.
    const auto format = metadata.extension[0];
    return format != 70 && format != 73 && format != 76 && format != 79
        && format != 82 && format != 94 && format != 97;
}

// The SDK owns these buffers and must also release them on early returns.
// This stack owner calls the matching SDK free function, not delete or free.
// Use these owners locally; copying them would create two owners of the same pointers.
struct OwnedMipSet
{
    CMP_MipSet data{};
    ~OwnedMipSet() { CMP_FreeMipSet(&data); }
};

// Keep the GPU device cached across files; release it when the program exits.
struct ComputeCleanup
{
    ~ComputeCleanup() { CMP_DestroyComputeLibrary(true); }
};

// Limit the pipeline to the BC family being compared. An SDK-supported image format
// is not automatically a supported input for this particular benchmark.
bool IsBC(CMP_FORMAT format)
{
    switch (format)
    {
    case CMP_FORMAT_BC1:
    case CMP_FORMAT_BC2:
    case CMP_FORMAT_BC3:
    case CMP_FORMAT_BC4:
    case CMP_FORMAT_BC4_S:
    case CMP_FORMAT_BC5:
    case CMP_FORMAT_BC5_S:
    case CMP_FORMAT_BC6H:
    case CMP_FORMAT_BC6H_SF:
    case CMP_FORMAT_BC7: return true;
    default: return false;
    }
}

// Build an SDK texture view without copying its pixels.
// CMP_Texture is a non-owning description of one mip inside a CMP_MipSet.
// The returned pData remains valid only while that mip set owns the allocation.
// Uncompressed pitch is bytes per row; compressed storage uses block sizes instead.
CMP_Texture GetTexture(CMP_MipSet &mips, int mip)
{
    CMP_MipLevel* level = nullptr;
    CMP_GetMipLevel(&level, &mips, mip, 0);
    CMP_Texture texture{};
    texture.dwSize = sizeof(texture);
    texture.format = mips.m_format;
    texture.nBlockWidth = texture.nBlockHeight = 4;
    texture.nBlockDepth = 1;
    if (!level)
        return texture;
    texture.dwWidth = level->m_nWidth;
    texture.dwHeight = level->m_nHeight;
    texture.dwDataSize = level->m_dwLinearSize;
    texture.pData = level->m_pbData;
    if (!IsBC(mips.m_format))
    {
        const int bytes = mips.m_ChannelFormat == CF_Float32 ? 16
            : mips.m_ChannelFormat == CF_Float16 ? 8 : 4;
        texture.dwPitch = texture.dwWidth * bytes;
    }
    return texture;
}

// Keep the SDK error code and the stage name together so a failed run can be traced.
bool CheckStatus(CMP_ERROR status, const char* stage)
{
    if (status == CMP_OK)
        return true;
    std::cerr << stage << " failed: " << status << '\n';
    return false;
}

// Allocate the decoded destination before starting the conversion timer.
// Float channels keep HDR range and signed BC4/BC5 values that RGBA8 would discard.
// gpu selects the intermediate representation required by DXC, not GPU decoding.
bool Decompress(CMP_MipSet &source, CMP_MipSet &decoded,
                CMP_CompressOptions &options, bool gpu, double &milliseconds)
{
    const auto codec = source.m_format;
    const bool hdr = codec == CMP_FORMAT_BC6H || codec == CMP_FORMAT_BC6H_SF;
    const bool signed_channels = codec == CMP_FORMAT_BC4_S || codec == CMP_FORMAT_BC5_S;
    // The DXC pipeline expects half-float HDR pixels; keep the CPU baseline unchanged.
    const auto channels = hdr && gpu ? CF_Float16 : hdr || signed_channels ? CF_Float32 : CF_8bit;
    const auto format = channels == CF_Float16 ? CMP_FORMAT_RGBA_16F
        : channels == CF_Float32 ? CMP_FORMAT_RGBA_32F : CMP_FORMAT_RGBA_8888;
    if (!CheckStatus(CMP_CreateMipSet(&decoded, source.m_nWidth, source.m_nHeight,
                                    1, channels, TT_2D), "Decode allocation"))
        return false;
    decoded.m_format = format;
    auto input = GetTexture(source, 0);
    auto output = GetTexture(decoded, 0);
    const auto start = GetTime();
    const auto status = CMP_ConvertTexture(&input, &output, &options, nullptr);
    milliseconds = (GetTime() - start) / 1000.0;
    return CheckStatus(status, "Decompress");
}

// Build a complete compressed mip set outside the compression timer.
// decoded supplies the mip dimensions; source supplies the original compressed format.
// Each lower mip needs its own SDK allocation before an encoder can write into it.
bool AllocateOutput(CMP_MipSet &source, CMP_MipSet &decoded, CMP_MipSet &result, bool dx10)
{
    if (!CheckStatus(CMP_CreateCompressMipSet(&result, &decoded), "Output allocation"))
        return false;
    result.m_format = source.m_format;
    result.m_compressed = true;
    result.m_Flags = source.m_Flags;
    result.m_nMipLevels = decoded.m_nMipLevels;
    CMP_Format2FourCC(result.m_format, &result);
    if (dx10)
        result.m_dwFourCC = CMP_MAKEFOURCC('D', 'X', '1', '0');

    // The SDK allocates Mip 0 first. Allocate each lower level and transfer ownership.
    for (int mip = 1; mip < result.m_nMipLevels; ++mip)
    {
        const auto pixels = GetTexture(decoded, mip);
        CMP_MipSet dimensions{};
        dimensions.m_nWidth = pixels.dwWidth;
        dimensions.m_nHeight = pixels.dwHeight;
        dimensions.m_nDepth = 1;
        dimensions.m_TextureType = TT_2D;
        dimensions.m_nMaxMipLevels = CMP_CalcMaxMipLevel(pixels.dwHeight, pixels.dwWidth, false);
        OwnedMipSet allocation;
        if (!CheckStatus(CMP_CreateCompressMipSet(&allocation.data, &dimensions), "Mip allocation"))
            return false;
        CMP_MipLevel* from = nullptr;
        CMP_MipLevel* to = nullptr;
        CMP_GetMipLevel(&from, &allocation.data, 0, 0);
        CMP_GetMipLevel(&to, &result, mip, 0);
        if (!from || !from->m_pbData || !to)
            return false;
        // Move the allocated level into result by exchanging the level descriptors.
        // The temporary owner then frees its remaining storage, while result owns
        // the transferred buffer. Copying the pointer without swapping would dangle.
        std::swap(*from, *to);
        auto texture = GetTexture(result, mip);
        to->m_dwLinearSize = CMP_CalculateBufferSize(&texture);
    }
    const auto original = GetTexture(source, 0);
    const auto output = GetTexture(result, 0);
    if (!original.pData || !output.pData || output.dwDataSize < original.dwDataSize)
        return false;
    // Preserve the original block bytes exactly; decoding Mip 0 was only needed
    // to generate lower levels. This copy is outside all three processing timers.
    std::memcpy(output.pData, original.pData, original.dwDataSize);
    CMP_MipLevel* base = nullptr;
    CMP_GetMipLevel(&base, &result, 0, 0);
    base->m_dwLinearSize = original.dwDataSize;
    return true;
}

// DXC requires whole 4x4 blocks. Repeat edge pixels without changing the mip's logical size.
// Store the logical mip dimensions in decoded/result and use a separate padded input.
// Repeating the last valid row/column avoids introducing a black border into edge blocks.
// Allocation and padding are preparation costs excluded from the compression timer.
bool PadGpuInput(CMP_MipSet &decoded, int mip, CMP_MipSet &padded)
{
    const auto input = GetTexture(decoded, mip);
    const unsigned int width = (input.dwWidth + 3) & ~3u;
    const unsigned int height = (input.dwHeight + 3) & ~3u;
    if (!CheckStatus(CMP_CreateMipSet(&padded, width, height, 1,
                                    decoded.m_ChannelFormat, TT_2D), "GPU input allocation"))
        return false;
    padded.m_format = decoded.m_format;
    const auto output = GetTexture(padded, 0);
    const auto bytes = input.dwPitch / input.dwWidth;
    for (unsigned int y = 0; y < height; ++y)
    {
        const auto* row = input.pData + std::min(y, input.dwHeight - 1) * input.dwPitch;
        auto* destination = output.pData + y * output.dwPitch;
        std::memcpy(destination, row, input.dwPitch);
        for (unsigned int x = input.dwWidth; x < width; ++x)
            std::memcpy(destination + x * bytes, row + (input.dwWidth - 1) * bytes, bytes);
    }
    padded.dwWidth = width;
    padded.dwHeight = height;
    padded.dwDataSize = output.dwDataSize;
    return true;
}

// The GPU plugin consumes a padded mip and writes into result's existing block buffer.
// Setup API calls below are outside the timer; work deferred by the SDK into
// CMP_CompressTexture is inside it. This is API wall time, not GPU kernel-only time.
bool CompressGpuMip(CMP_MipSet &decoded, CMP_MipSet &result, int mip,
                    CMP_CompressOptions &options, double &milliseconds)
{
    OwnedMipSet padded;
    if (!PadGpuInput(decoded, mip, padded.data))
        return false;

    KernelOptions kernel{};
    kernel.encodeWith = CMP_GPU_DXC;
    kernel.format = result.m_format;
    kernel.fquality = options.fquality;
    kernel.bc15.useAlphaThreshold = true;
    kernel.bc15.alphaThreshold = options.nAlphaThreshold;
    if (!CheckStatus(CMP_CreateComputeLibrary(&padded.data, &kernel, nullptr), "GPU pipeline setup"))
    {
        std::cerr << "GPU needs EncodeWith_DXC.dll next to the executable and plugins/compute shaders.\n";
        return false;
    }
    ComputeOptions compute{};
    if (!CheckStatus(CMP_SetComputeOptions(&compute), "GPU options"))
        return false;

    const auto output = GetTexture(result, mip);
    // This shallow copy is a borrowed view, not an OwnedMipSet. Its pointer fields
    // refer to result, so freeing destination would also invalidate result's storage.
    CMP_MipSet destination = result; // Borrow this mip's output buffer; do not free the view.
    destination.dwWidth = padded.data.dwWidth;
    destination.dwHeight = padded.data.dwHeight;
    destination.pData = output.pData;
    destination.dwDataSize = output.dwDataSize;

    // Includes upload, GPU work, readback, and lazy device/shader setup on the first call.
    const auto start = GetTime();
    const auto status = CMP_CompressTexture(&kernel, padded.data, destination, nullptr);
    milliseconds += (GetTime() - start) / 1000.0;
    return CheckStatus(status, "GPU compress");
}

// Start at mip 1: mip 0 already contains the unchanged source blocks.
// Accumulate one call per mip. A 1x1 source has no lower mip and takes zero encode time.
bool CompressMips(CMP_MipSet &decoded, CMP_MipSet &result,
                  CMP_CompressOptions &options, bool gpu, double &milliseconds)
{
    for (int mip = 1; mip < result.m_nMipLevels; ++mip)
    {
        if (gpu)
        {
            if (!CompressGpuMip(decoded, result, mip, options, milliseconds))
                return false;
            continue;
        }
        auto input = GetTexture(decoded, mip);
        auto output = GetTexture(result, mip);
        // shortcut: SDK 4.6.0 corrupts tiny BC1 mips with auto threads; retest after upgrading.
        options.dwnumThreads = input.dwWidth < 4 || input.dwHeight < 4 ? 1 : 0;
        const auto start = GetTime();
        const auto status = CMP_ConvertTexture(&input, &output, &options, nullptr);
        milliseconds += (GetTime() - start) / 1000.0;
        if (!CheckStatus(status, "Compress"))
            return false;
    }
    if (gpu && result.m_nMipLevels > 1)
    {
        KernelDeviceInfo device{};
        if (!CheckStatus(CMP_GetDeviceInfo(&device), "GPU device info"))
            return false;
        // The SDK may use the reference rasterizer if hardware creation fails.
        if (std::strstr(device.m_deviceName, "Reference") || std::strstr(device.m_deviceName, "Basic Render"))
        {
            std::cerr << "Software rasterizers are excluded from the GPU benchmark.\n";
            return false;
        }
        std::cout << "GPU device: " << device.m_deviceName << '\n';
    }
    return true;
}

// Let the SDK write all DDS pixels, then restore tags that its format enum merges.
// This helper is currently not called by ProcessFile because saving is disabled there.
// Reopening the file and patching the DX10 extension are both outside processing time.
bool SaveDDS(const std::filesystem::path &output, CMP_MipSet &result, const DDSMetadata &metadata)
{
    if (!CheckStatus(CMP_SaveTexture(output.string().c_str(), &result), "DDS save"))
        return false;
    if (!metadata.dx10)
        return true;
    // Restore the original sRGB and alpha metadata; the SDK wrote all pixel data.
    std::fstream file(output, std::ios::binary | std::ios::in | std::ios::out);
    uint32_t fourcc = 0;
    file.seekg(84);
    file.read(reinterpret_cast<char*>(&fourcc), sizeof(fourcc));
    if (!file || fourcc != CMP_MAKEFOURCC('D', 'X', '1', '0'))
    {
        std::cerr << "SDK did not save the requested DX10 header.\n";
        return false;
    }
    file.seekp(128);
    file.write(reinterpret_cast<const char*>(metadata.extension.data()), sizeof(metadata.extension));
    file.flush();
    if (!file)
        std::cerr << "Cannot preserve DX10 metadata.\n";
    return static_cast<bool>(file);
}

// Keep source, decoded, and compressed storage independent so Mip 0 remains intact.
// Stack owners release all three mip sets if any stage returns an error.
int ProcessFile(const std::filesystem::path &input, const std::filesystem::path &output,
                float quality, bool gpu)
{
    DDSMetadata metadata;
    OwnedMipSet source, decoded, result;
    if (!ReadDDSMetadata(input, metadata))
    {
        std::cerr << "Invalid DDS header or unsupported texture layout.\n";
        return 1;
    }
    if (!CheckStatus(CMP_LoadTexture(input.string().c_str(), &source.data), "DDS load"))
        return 1;
    const auto codec = source.data.m_format;
    if (!IsBC(codec) || source.data.m_nMipLevels != 1
        || source.data.m_TextureType != TT_2D || source.data.m_nDepth != 1)
    {
        std::cerr << "Input must be a typed BC1-BC7 single 2D texture with only Mip 0.\n";
        return 1;
    }
    if (gpu && (codec == CMP_FORMAT_BC4_S || codec == CMP_FORMAT_BC5_S))
    {
        std::cerr << "The DXC GPU pipeline does not support signed BC4/BC5.\n";
        return 1;
    }
    const auto base = GetTexture(source.data, 0);
    if (!base.pData || !base.dwWidth || !base.dwHeight
        || base.dwWidth > std::numeric_limits<CMP_DWORD>::max() / 16 / base.dwHeight)
    {
        std::cerr << "Input exceeds SDK buffer limits.\n";
        return 1;
    }

    // Initialize the SDK structure size and select the requested encoder quality.
    // BC1 alpha uses a fixed 128 cutoff; this is an encoder policy, not a timing option.
    CMP_CompressOptions options{};
    options.dwSize = sizeof(options);
    options.fquality = quality;
    options.bDXT1UseAlpha = true;
    options.nAlphaThreshold = 128;
    double decode_ms = 0, mip_ms = 0, encode_ms = 0;

    // Only these three processing steps contribute to the reported total.
    if (!Decompress(source.data, decoded.data, options, gpu, decode_ms))
        return 1;
    const auto start = GetTime();
    const int mip_status = CMP_GenerateMIPLevels(&decoded.data, 1); // CPU Box, down to 1x1.
    mip_ms = (GetTime() - start) / 1000.0;
    if (mip_status != 0)
    {
        std::cerr << "GenerateMIPLevels failed: " << mip_status << '\n';
        return 1;
    }
    if (!AllocateOutput(source.data, decoded.data, result.data, metadata.dx10))
        return 1;
    if (!CompressMips(decoded.data, result.data, options, gpu, encode_ms))
        return 1;
    // Saving is deliberately left in its current disabled state for this comment pass.
    // The printed success count does not imply that an output file was written.
    // if (!SaveDDS(output, result.data, metadata))
    //     return 1;

    std::cout << "Compression backend: " << (gpu ? "GPU DXC" : "CPU (auto threads; tiny mips: 1)") << '\n'
              << "Quality: " << quality << '\n'
              << "Mip levels: " << result.data.m_nMipLevels << '\n'
              << "Decompress: " << decode_ms << " ms\n"
              << "Downsampling: " << mip_ms << " ms\n"
              << "Compression: " << encode_ms << " ms\n"
              << "Total generation time: " << decode_ms + mip_ms + encode_ms << " ms\n";
    return 0;
}

// Extension matching is case-insensitive, so .DDS and .dds are treated alike.
// unsigned char keeps the argument in the valid range required by std::tolower.
char Lowercase(unsigned char c)
{
    return static_cast<char>(std::tolower(c));
}

// Batch handling is outside the processing timers. Only this directory is scanned.
// Sort before processing to make the input order repeatable between runs.
// Files are processed one at a time; codec-internal threads may still run in parallel.
// Continue after a file fails, then return failure if any file was skipped or failed.
int ProcessBatch(const std::filesystem::path &folder, const std::filesystem::path &output, float quality, bool gpu)
{
    // Read this folder only and keep a stable order between runs.
    std::error_code error;
    std::vector<std::filesystem::path> files;
    std::filesystem::directory_iterator entry(folder, error), end;
    
    for (; !error && entry != end; entry.increment(error))
    {
        const bool regular = entry->is_regular_file(error);
        if (error)
            break;
        auto extension = entry->path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), Lowercase);
        if (regular && extension == ".dds")
            files.push_back(entry->path());
    }
    
    if (error || files.empty())
    {
        std::cerr << "Cannot collect DDS files: " << (error ? error.message() : "No DDS files found.") << '\n';
        return 1;
    }
    
    std::sort(files.begin(), files.end());
    std::filesystem::create_directories(output, error);
    
    if (error)
    {
        std::cerr << "Cannot create output folder: " << error.message() << '\n';
        return 1;
    }

    size_t succeeded = 0;
    for (const auto &file : files)
    {
        std::cout << "Processing: " << file << '\n';
        const auto destination = output / file.filename();
        const bool exists = std::filesystem::exists(destination, error);
        if (error || exists)
        {
            std::cerr << "Cannot write output: " << destination << '\n';
            continue;
        }
        if (ProcessFile(file, destination, quality, gpu) == 0)
            ++succeeded;
    }
    std::cout << "Succeeded: " << succeeded << ", failed: " << files.size() - succeeded << '\n';
    return succeeded == files.size() ? 0 : 1;
}

// Parse the trailing GPU switch first, leaving an optional numeric quality argument.
// Framework initialization and final GPU cleanup are outside per-image timers.
int main(int argc, char** argv)
{
    bool gpu = false;
    float quality = 0.05f;
    
    if (argc > 3 && std::strcmp(argv[argc - 1], "--gpu") == 0)
    {
        gpu = true;
        --argc;
    }
    
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: compressnator <input.dds|folder|folder/*.dds> <output.dds|folder> [quality: 0..1] [--gpu]\n";
        return 1;
    }
    
    if (argc == 4)
    {
        const char* end = argv[3] + std::strlen(argv[3]);
        // Require the entire argument to be a finite number in [0, 1].
        // from_chars reports an error code and does not need exception handling.
        const auto parsed = std::from_chars(argv[3], end, quality);
        if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(quality) || quality < 0 || quality > 1)
        {
            std::cerr << "Quality must be a number from 0 to 1.\n";
            return 1;
        }
    }
    
    CMP_InitFramework();
    ComputeCleanup compute_cleanup;
    const std::filesystem::path input(argv[1]), output(argv[2]);
    if (input.filename() == "*.dds")
        return ProcessBatch(input.has_parent_path() ? input.parent_path() : ".", output, quality, gpu);
    
    // The error_code overload reports filesystem errors without a try/catch block.
    // An ordinary file goes to ProcessFile; a directory goes to ProcessBatch.
    std::error_code error;
    const bool directory = std::filesystem::is_directory(input, error);
    if (error)
    {
        std::cerr << "Cannot access input: " << error.message() << '\n';
        return 1;
    }
    
    if (directory)
        return ProcessBatch(input, output, quality, gpu);
    
    // Never overwrite an input or an existing output in single-file mode either.
    const bool exists = std::filesystem::exists(output, error);
    if (error || exists)
    {
        std::cerr << "Cannot write output: " << output << '\n';
        return 1;
    }
    
    return ProcessFile(input, output, quality, gpu);
}
