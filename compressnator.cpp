// Compressonator benchmark using its own DDS I/O, mip generator, and codecs.
// CPU decoding and Box filtering are used for both CPU and DXC compression runs.
// Only lower mips are compressed; Mip 0 is copied without changing its blocks.

#include <compressonator.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <iostream>
#include <limits>
#include <vector>

#include "common/timer.h"

// Explicit 256-entry sRGB to Linear table used by the CPU paths.
static const float c_srgb_to_linear[256] = {
    0.0f,         0.000303527f, 0.000607054f, 0.000910581f, 0.001214108f, 0.001517635f, 0.001821162f, 0.002124689f,
    0.002428216f, 0.002731743f, 0.003035270f, 0.003346536f, 0.003676507f, 0.004024717f, 0.004391442f, 0.004776953f,
    0.005181517f, 0.005605392f, 0.006048833f, 0.006512091f, 0.006995410f, 0.007499032f, 0.008023193f, 0.008568126f,
    0.009134059f, 0.009721217f, 0.010329820f, 0.010960090f, 0.011612250f, 0.012286490f, 0.012983030f, 0.013702080f,
    0.014443840f, 0.015208510f, 0.015996290f, 0.016807380f, 0.017641950f, 0.018500220f, 0.019382360f, 0.020288560f,
    0.021219010f, 0.022173880f, 0.023153370f, 0.024157630f, 0.025186860f, 0.026241220f, 0.027320890f, 0.028426040f,
    0.029556830f, 0.030713440f, 0.031896030f, 0.033104770f, 0.034339810f, 0.035601310f, 0.036889450f, 0.038204370f,
    0.039546240f, 0.040915200f, 0.042311410f, 0.043735030f, 0.045186200f, 0.046665090f, 0.048171820f, 0.049706570f,
    0.051269460f, 0.052860650f, 0.054480280f, 0.056128490f, 0.057805430f, 0.059511240f, 0.061246050f, 0.063010020f,
    0.064803270f, 0.066625940f, 0.068478170f, 0.070360100f, 0.072271850f, 0.074213570f, 0.076185380f, 0.078187420f,
    0.080219820f, 0.082282710f, 0.084376210f, 0.086500460f, 0.088655590f, 0.090841710f, 0.093058960f, 0.095307470f,
    0.097587350f, 0.099898730f, 0.102241700f, 0.104616500f, 0.107023100f, 0.109461700f, 0.111932400f, 0.114435400f,
    0.116970700f, 0.119538400f, 0.122138800f, 0.124771800f, 0.127437700f, 0.130136500f, 0.132868300f, 0.135633300f,
    0.138431600f, 0.141263300f, 0.144128500f, 0.147027300f, 0.149959800f, 0.152926200f, 0.155926500f, 0.158960800f,
    0.162029400f, 0.165132200f, 0.168269400f, 0.171441100f, 0.174647400f, 0.177888400f, 0.181164200f, 0.184475000f,
    0.187820800f, 0.191201700f, 0.194617800f, 0.198069300f, 0.201556300f, 0.205078700f, 0.208636900f, 0.212230800f,
    0.215860500f, 0.219526200f, 0.223228000f, 0.226965900f, 0.230740000f, 0.234550600f, 0.238397600f, 0.242281100f,
    0.246201300f, 0.250158300f, 0.254152100f, 0.258182900f, 0.262250700f, 0.266355600f, 0.270497800f, 0.274677300f,
    0.278894300f, 0.283148700f, 0.287440800f, 0.291770600f, 0.296138300f, 0.300543800f, 0.304987300f, 0.309468900f,
    0.313988700f, 0.318546800f, 0.323143200f, 0.327778100f, 0.332451500f, 0.337163600f, 0.341914400f, 0.346704100f,
    0.351532600f, 0.356400100f, 0.361306800f, 0.366252600f, 0.371237700f, 0.376262100f, 0.381326000f, 0.386429400f,
    0.391572500f, 0.396755200f, 0.401977800f, 0.407240200f, 0.412542600f, 0.417885100f, 0.423267700f, 0.428690500f,
    0.434153600f, 0.439657200f, 0.445201200f, 0.450785800f, 0.456411000f, 0.462077000f, 0.467783800f, 0.473531500f,
    0.479320200f, 0.485149900f, 0.491020800f, 0.496933000f, 0.502886500f, 0.508881300f, 0.514917700f, 0.520995600f,
    0.527115100f, 0.533276400f, 0.539479500f, 0.545724500f, 0.552011400f, 0.558340400f, 0.564711500f, 0.571124800f,
    0.577580400f, 0.584078400f, 0.590618800f, 0.597201800f, 0.603827300f, 0.610495600f, 0.617206600f, 0.623960400f,
    0.630757100f, 0.637596900f, 0.644479700f, 0.651405600f, 0.658374800f, 0.665387300f, 0.672443200f, 0.679542500f,
    0.686685300f, 0.693871800f, 0.701101900f, 0.708375800f, 0.715693500f, 0.723055100f, 0.730460700f, 0.737910400f,
    0.745404200f, 0.752942200f, 0.760524500f, 0.768151100f, 0.775822200f, 0.783537800f, 0.791297900f, 0.799102700f,
    0.806952300f, 0.814846600f, 0.822785800f, 0.830769900f, 0.838799000f, 0.846873200f, 0.854992600f, 0.863157200f,
    0.871367100f, 0.879622400f, 0.887923100f, 0.896269400f, 0.904661200f, 0.913098700f, 0.921581900f, 0.930110900f,
    0.938685700f, 0.947306500f, 0.955973400f, 0.964686200f, 0.973445300f, 0.982250600f, 0.991102100f, 1.0f};

float SrgbToLinear(float value)
{
    const float index = std::clamp(value, 0.0f, 1.0f) * 255.0f;
    const auto low = static_cast<unsigned int>(index);
    const auto high = std::min(low + 1, 255u);
    return c_srgb_to_linear[low]
        + (c_srgb_to_linear[high] - c_srgb_to_linear[low]) * (index - low);
}

const std::array<CMP_BYTE, 65536>& LinearToSrgbTable()
{
    static const auto table = [] {
        std::array<CMP_BYTE, 65536> values{};
        for (size_t i = 0; i < values.size(); ++i)
        {
            const double linear = double(i) / (values.size() - 1);
            const double srgb = linear <= 0.0031308 ? 12.92 * linear
                : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
            values[i] = static_cast<CMP_BYTE>(srgb * 255.0 + 0.5);
        }
        return values;
    }();
    return table;
}

CMP_BYTE LinearToSrgb(float value)
{
    // shortcut: 16-bit linear lookup can differ by one output byte; use exact conversion if required.
    const auto index = static_cast<unsigned int>(std::clamp(value, 0.0f, 1.0f) * 65535.0f + 0.5f);
    return LinearToSrgbTable()[index];
}

// The SDK merges sRGB and UNORM formats, so retain the optional DX10 metadata.
struct DDSMetadata
{
    bool dx10 = false;
    std::array<uint32_t, 5> extension{};
};

// Validate the layout and retain DX10 tags before the SDK loads the pixels.
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

// Release SDK allocations on early returns. Do not copy these owners.
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

// Non-owning view of one mip; pitch is set only for uncompressed pixels.
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

bool CheckStatus(CMP_ERROR status, const char* stage)
{
    if (status == CMP_OK)
        return true;
    std::cerr << stage << " failed: " << status << '\n';
    return false;
}

// Allocate the decoded destination before starting the conversion timer.
// Float channels retain HDR/signed values and precision for linear-light sRGB filtering.
bool Decompress(CMP_MipSet &source, CMP_MipSet &decoded, CMP_CompressOptions &options, bool srgb, double &milliseconds)
{
    const auto codec = source.m_format;
    const bool hdr = codec == CMP_FORMAT_BC6H || codec == CMP_FORMAT_BC6H_SF;
    const bool signed_channels = codec == CMP_FORMAT_BC4_S || codec == CMP_FORMAT_BC5_S;
    
    // Keep Box filtering in float32; SDK 4.6 half-float filtering fails the Box reference check.
    const auto channels = hdr || signed_channels || srgb ? CF_Float32 : CF_8bit;
    const auto format = channels == CF_Float32 ? CMP_FORMAT_RGBA_32F : CMP_FORMAT_RGBA_8888;

    if (!CheckStatus(CMP_CreateMipSet(&decoded, source.m_nWidth, source.m_nHeight, 1, channels, TT_2D), "Decode allocation"))
        return false;
    
    decoded.m_format = format;
    
    auto input = GetTexture(source, 0);
    auto output = GetTexture(decoded, 0);
    
    const auto start = GetTime();
    const auto status = CMP_ConvertTexture(&input, &output, &options, nullptr);
    milliseconds = (GetTime() - start) / 1000.0;

    return CheckStatus(status, "Decompress");
}

// Filter RGB in linear light; alpha is not gamma converted.
bool GenerateMips(CMP_MipSet &decoded, bool srgb, double &milliseconds)
{
    const auto start = GetTime();

    if (srgb)
    {
        const auto base = GetTexture(decoded, 0);
        auto* pixels = reinterpret_cast<float*>(base.pData);

#pragma omp parallel for schedule(static) if(base.dwWidth * base.dwHeight >= 65536)
        for (int y = 0; y < static_cast<int>(base.dwHeight); ++y)
        {
            auto* row = pixels + size_t(y) * base.dwWidth * 4;
            for (unsigned int x = 0; x < base.dwWidth; ++x)
            {
                auto* pixel = row + size_t(x) * 4;
                pixel[0] = SrgbToLinear(pixel[0]);
                pixel[1] = SrgbToLinear(pixel[1]);
                pixel[2] = SrgbToLinear(pixel[2]);
            }
        }
    }

    const int mip_status = CMP_GenerateMIPLevels(&decoded, 1); // CPU Box, down to 1x1.

    milliseconds = (GetTime() - start) / 1000.0;
    if (mip_status != 0)
    {
        std::cerr << "GenerateMIPLevels failed: " << mip_status << '\n';
        return false;
    }

    return true;
}

// Allocate the full output and preserve Mip 0 outside processing timers.
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
        // Transfer buffer ownership to result; the temporary frees the old descriptor.
        std::swap(*from, *to);
        auto texture = GetTexture(result, mip);
        to->m_dwLinearSize = CMP_CalculateBufferSize(&texture);
    }
    const auto original = GetTexture(source, 0);
    const auto output = GetTexture(result, 0);
    if (!original.pData || !output.pData || output.dwDataSize < original.dwDataSize)
        return false;
    std::memcpy(output.pData, original.pData, original.dwDataSize);
    CMP_MipLevel* base = nullptr;
    CMP_GetMipLevel(&base, &result, 0, 0);
    base->m_dwLinearSize = original.dwDataSize;
    return true;
}

// Convert linear RGB back to sRGB when needed and repeat edges to fill BC blocks.
bool PrepareEncoderInput(CMP_MipSet &decoded, int mip, bool srgb, CMP_MipSet &encoderInput)
{
    const auto input = GetTexture(decoded, mip);
    const auto width = (input.dwWidth + 3) & ~3u;
    const auto height = (input.dwHeight + 3) & ~3u;

    if (!CheckStatus(CMP_CreateMipSet(&encoderInput, width, height, 1, srgb ? CF_8bit : decoded.m_ChannelFormat, TT_2D), "Encoder input allocation"))
    {
        return false;
    }

    encoderInput.m_format = srgb ? CMP_FORMAT_RGBA_8888 : decoded.m_format;
    
    const auto prepared = GetTexture(encoderInput, 0);
    const auto bytes = input.dwPitch / input.dwWidth;
    
#pragma omp parallel for schedule(static) if(width * height >= 65536)
    for (int y = 0; y < static_cast<int>(height); ++y)
    {
        const auto sourceY = std::min(static_cast<unsigned int>(y), input.dwHeight - 1);
        const auto* sourceRow = input.pData + size_t(sourceY) * input.dwPitch;

        auto* outputRow = prepared.pData + size_t(y) * prepared.dwPitch;
        if (srgb)
        {
            const auto* pixels = reinterpret_cast<const float*>(sourceRow);
            for (unsigned int x = 0; x < width; ++x)
            {
                const auto* pixel = pixels + size_t(std::min(x, input.dwWidth - 1)) * 4;
                auto* destination = outputRow + size_t(x) * 4;
                destination[0] = LinearToSrgb(pixel[0]);
                destination[1] = LinearToSrgb(pixel[1]);
                destination[2] = LinearToSrgb(pixel[2]);
                destination[3] = static_cast<CMP_BYTE>(std::clamp(pixel[3], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        else
        {
            std::memcpy(outputRow, sourceRow, input.dwPitch);
            for (unsigned int x = input.dwWidth; x < width; ++x)
            {
                std::memcpy(outputRow + size_t(x) * bytes, sourceRow + size_t(input.dwWidth - 1) * bytes, bytes);
            }
        }
    }
    return true;
}

bool CompressMipCPU(CMP_MipSet &encoderInput, CMP_MipSet &result,
                    const CMP_Texture &output, KernelOptions &kernel, CMP_MipSet &encoded)
{
    if (!CheckStatus(CMP_ProcessTexture(&encoderInput, &encoded, kernel, nullptr), "Framework HPC compression"))
    {
        return false;
    }
    const auto compressed = GetTexture(encoded, 0);
    if (encoded.m_format != result.m_format || !compressed.pData || compressed.dwDataSize != output.dwDataSize)
    {
        std::cerr << "Framework output format or block size mismatch.\n";
        return false;
    }
    std::memcpy(output.pData, compressed.pData, output.dwDataSize);
    return true;
}

bool CompressMipGPU(CMP_MipSet &encoderInput, CMP_MipSet &result,
                    const CMP_Texture &output, KernelOptions &kernel)
{
    const auto prepared = GetTexture(encoderInput, 0);
    encoderInput.dwWidth = prepared.dwWidth;
    encoderInput.dwHeight = prepared.dwHeight;
    encoderInput.dwDataSize = prepared.dwDataSize;
    encoderInput.pData = prepared.pData;
    
    // Keep the compute device alive so its hardware identity can be checked.
    if (!CheckStatus(CMP_CreateComputeLibrary(&encoderInput, &kernel, nullptr), "GPU pipeline setup"))
    {
        return false;
    }
    
    ComputeOptions compute{};
    if (!CheckStatus(CMP_SetComputeOptions(&compute), "GPU options"))
    {
        return false;
    }
    
    CMP_MipSet destination = result; // Borrow the output; never free this view.
    destination.dwWidth = prepared.dwWidth;
    destination.dwHeight = prepared.dwHeight;
    destination.pData = output.pData;
    destination.dwDataSize = output.dwDataSize;
    if (!CheckStatus(CMP_CompressTexture(&kernel, encoderInput, destination, nullptr), "GPU compression"))
    {
        return false;
    }

    return true;
}

// Input preparation counts as downsampling; SDK encoding and its output copy count as compression.
bool CompressMips(CMP_MipSet &decoded, CMP_MipSet &result,
                  CMP_CompressOptions &options, bool gpu, bool srgb,
                  double &compression_ms, double &downsampling_ms)
{
    for (int mip = 1; mip < result.m_nMipLevels; ++mip)
    {
        const auto prepare_start = GetTime();
        OwnedMipSet encoderInput;
        if (!PrepareEncoderInput(decoded, mip, srgb, encoderInput.data))
        {
            return false;
        }
        downsampling_ms += (GetTime() - prepare_start) / 1000.0;

        KernelOptions kernel{};
        kernel.encodeWith = gpu ? CMP_GPU_DXC : CMP_HPC; // kernel setting 
        kernel.format = result.m_format;
        kernel.srcformat = encoderInput.data.m_format;
        kernel.fquality = options.fquality;
        kernel.threads = 0; // SDK automatic threading.
        kernel.bc15.useAlphaThreshold = options.bDXT1UseAlpha;
        kernel.bc15.alphaThreshold = options.nAlphaThreshold;

        OwnedMipSet encoded;
        const auto encode_start = GetTime();
        const auto output = GetTexture(result, mip);
        const bool success = gpu ? CompressMipGPU(encoderInput.data, result, output, kernel) : CompressMipCPU(encoderInput.data, result, output, kernel, encoded.data);
        if (!success)
        {
            return false;
        }
        compression_ms += (GetTime() - encode_start) / 1000.0;
    }
    if (gpu && result.m_nMipLevels > 1)
    {
        KernelDeviceInfo device{};
        if (!CheckStatus(CMP_GetDeviceInfo(&device), "GPU device info"))
        {
            return false;
        }
        if (std::strstr(device.m_deviceName, "Reference") || std::strstr(device.m_deviceName, "Basic Render"))
        {
            std::cerr << "Software rasterizers are excluded from the GPU benchmark.\n";
            return false;
        }
    }
    return true;
}

// Let the SDK write all DDS pixels, then restore tags that its format enum merges.
// Reopening the file and patching the DX10 extension are both outside processing time.
bool SaveDDS(const std::filesystem::path &output, CMP_MipSet &result, const DDSMetadata &metadata)
{
    if (!CheckStatus(CMP_SaveTexture(output.string().c_str(), &result), "DDS save"))
        return false;
    std::fstream file(output, std::ios::binary | std::ios::in | std::ios::out);
    std::array<uint32_t, 32> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!file || header[0] != 0x20534444 || header[1] != 124
        || (metadata.dx10 && header[21] != CMP_MAKEFOURCC('D', 'X', '1', '0')))
    {
        std::cerr << "SDK did not save the requested DDS header.\n";
        return false;
    }
    header[2] = (header[2] & ~0x8u) | 0x81007u; // CAPS, HEIGHT, WIDTH, PIXELFORMAT, LINEARSIZE.
    header[5] = GetTexture(result, 0).dwDataSize;
    header[27] |= 0x1000u; // DDSCAPS_TEXTURE.
    if (result.m_nMipLevels > 1)
    {
        header[2] |= 0x20000u;
        header[27] |= 0x400008u; // MIPMAP and COMPLEX.
    }
    file.seekp(0);
    file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    if (metadata.dx10)
    {
        file.seekp(128);
        file.write(reinterpret_cast<const char*>(metadata.extension.data()), sizeof(metadata.extension));
    }
    file.flush();
    if (!file)
    {
        std::cerr << "Cannot preserve DDS metadata.\n";
    }
    return static_cast<bool>(file);
}

// Processing times in milliseconds; file I/O and warm-up are excluded.
struct BatchTimes
{
    double decompress = 0;
    double downsample = 0;
    double compress = 0;
};

void WriteCsvRow(std::ostream& csv, const std::string& name, const BatchTimes& times)
{
    csv << '"';
    for (char c : name)
    {
        if (c == '"')
            csv << '"';
        csv << c;
    }
    csv << "\"," << times.decompress << ',' << times.downsample << ',' << times.compress
        << ',' << times.decompress + times.downsample + times.compress << '\n';
}

bool OpenCsv(std::ofstream& csv, const std::filesystem::path& path)
{
    csv.open(path);
    csv.imbue(std::locale::classic());
    csv << std::fixed << std::setprecision(6) << "file name,decompression time,downsampling time,compression time,total generation time\n";
    if (!csv)
    {
        std::cerr << "Cannot create CSV: " << path << '\n';
        return false;
    }

    return true;
}

int ProcessFile(const std::filesystem::path &input, const std::filesystem::path &output,
                float quality, bool gpu, BatchTimes* total = nullptr, bool warmup = false)
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
    const auto dxgi = metadata.extension[0];
    const bool srgb = metadata.dx10 && (dxgi == 72 || dxgi == 75 || dxgi == 78 || dxgi == 99);

    if (!IsBC(codec) || source.data.m_nMipLevels != 1 || source.data.m_TextureType != TT_2D || source.data.m_nDepth != 1)
    {
        std::cerr << "Input must be a typed BC1-BC7 single 2D texture with only Mip 0.\n";
        return 1;
    }

    if (gpu && (codec == CMP_FORMAT_BC4_S || codec == CMP_FORMAT_BC5_S))
    {
        std::cerr << "The DXC GPU pipeline does not support signed BC4/BC5.\n";
        return 1;
    }

    // shortcut: bundled DXC BC6 shader returns zero blocks; enable only after runtime validation.
    if (gpu && (codec == CMP_FORMAT_BC6H || codec == CMP_FORMAT_BC6H_SF))
    {
        std::cerr << "N/A: BC6H GPU compression is disabled for the bundled SDK 4.6 shader.\n";
        return 1;
    }

    const auto base = GetTexture(source.data, 0);
    if (!base.pData || !base.dwWidth || !base.dwHeight || base.dwWidth > std::numeric_limits<CMP_DWORD>::max() / 16 / base.dwHeight)
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
    if (!Decompress(source.data, decoded.data, options, srgb, decode_ms))
        return 1;

    if (!GenerateMips(decoded.data, srgb, mip_ms))
    {
        return 1;
    }

    if (!AllocateOutput(source.data, decoded.data, result.data, metadata.dx10))
        return 1;

    if (!CompressMips(decoded.data, result.data, options, gpu, srgb, encode_ms, mip_ms))
        return 1;

    // Warm-up completes the same processing path without saving or reporting.
    if (warmup)
        return 0;

    if (!SaveDDS(output, result.data, metadata))
        return 1;

    if (total)
    {
        total->decompress = decode_ms;
        total->downsample = mip_ms;
        total->compress = encode_ms;
    }
    return 0;
}

// unsigned char keeps std::tolower valid for non-ASCII filename bytes.
char Lowercase(unsigned char c)
{
    return static_cast<char>(std::tolower(c));
}

// Process files sequentially in sorted order; SDK/OpenMP parallelism is within each file.
int ProcessBatch(const std::filesystem::path &folder, const std::filesystem::path &output, float quality, bool gpu)
{
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

    std::ofstream csv;
    if (!OpenCsv(csv, output / "timings.csv"))
        return 1;

    BatchTimes total;
    size_t succeeded = 0;
    bool warmed_up = false;
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
        if (!warmed_up)
        {
            if (ProcessFile(file, destination, quality, gpu, nullptr, true) != 0)
                continue;
            warmed_up = true;
            std::cout << "Warm-up completed (excluded from averages).\n";
        }
        BatchTimes times;
        if (ProcessFile(file, destination, quality, gpu, &times) == 0)
        {
            WriteCsvRow(csv, file.filename().u8string(), times);
            total.decompress += times.decompress;
            total.downsample += times.downsample;
            total.compress += times.compress;
            ++succeeded;
        }
    }
    std::cout << "Succeeded: " << succeeded << ", failed: " << files.size() - succeeded << '\n';
    if (succeeded > 0)
    {
        WriteCsvRow(csv, "average", {total.decompress / succeeded, total.downsample / succeeded, total.compress / succeeded});
    }
    else
        std::cout << "Average times unavailable: no images succeeded.\n";
    csv.close();
    if (!csv)
    {
        std::cerr << "CSV write failed.\n";
        return 1;
    }
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
        // Reject partial parses, non-finite values, and out-of-range quality.
        const auto parsed = std::from_chars(argv[3], end, quality);
        if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(quality) || quality < 0 || quality > 1)
        {
            std::cerr << "Quality must be a number from 0 to 1.\n";
            return 1;
        }
    }

    LinearToSrgbTable(); // Initialize once, outside all per-file processing timers.
    CMP_InitFramework();

    ComputeCleanup compute_cleanup;
    const std::filesystem::path input(argv[1]), output(argv[2]);
    if (input.filename() == "*.dds")
        return ProcessBatch(input.has_parent_path() ? input.parent_path() : ".", output, quality, gpu);

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

    if (ProcessFile(input, output, quality, gpu, nullptr, true) != 0)
        return 1;

    std::cout << "Warm-up completed (excluded from averages).\n";

    std::ofstream csv;
    auto csv_path = output;
    csv_path.replace_extension(".timings.csv");
    if (!OpenCsv(csv, csv_path))
        return 1;

    BatchTimes times;
    const int result = ProcessFile(input, output, quality, gpu, &times);
    if (result == 0)
    {
        WriteCsvRow(csv, input.filename().u8string(), times);
        WriteCsvRow(csv, "average", times);
    }

    csv.close();
    if (!csv)
    {
        std::cerr << "CSV write failed.\n";
        return 1;
    }

    return result;
}
