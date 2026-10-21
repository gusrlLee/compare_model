// NVTT benchmark using raw BC blocks and the Surface/Context APIs.
// A small DDS reader retains original Mip 0; NVTT handles decoding and lower mip encoding.
// Output is collected in memory so filesystem writes never enter the encode timer.
// The installed SDK is 3.2.5; unsupported signed BC4/BC5 are rejected explicitly.

#include <nvtt/nvtt.h>
#include <cuda.h>
#include "nvtt_gpu.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <iostream>
#include <limits>
#include <vector>

#include "common/timer.h"

// Only DDS metadata and original Mip 0 bytes are kept outside NVTT.
// header has 32 words for legacy DDS plus 5 words for the DX10 extension.
// blocks owns only the compressed base level; Surface owns the decoded working image.
struct DDSImage
{
    std::array<uint32_t, 37> header{};
    nvtt::Format format = nvtt::Format_Unset;
    int width = 0, height = 0;
    bool dx10 = false, srgb = false;
    std::vector<uint8_t> blocks;
};

// Check DDS magic, structure sizes, FourCC, texture layout, and payload length.
// Word offsets include the magic: [3]/[4] are height/width, [7] is mip count,
// [21] is FourCC, and [32..36] are DX10 format/dimension/flags/array/alpha metadata.
// Native integer reads assume the little-endian platform targeted by this project.
bool ReadDDS(const std::filesystem::path &path, DDSImage &image)
{
    std::ifstream file(path, std::ios::binary);
    auto &header = image.header;
    file.read(reinterpret_cast<char*>(header.data()), 128);
    if (!file || header[0] != 0x20534444 || header[1] != 124 || header[19] != 32
        || !(header[20] & 4) || (header[28] & 0x200200)
        || ((header[2] & 0x20000) && header[7] != 1))
        return false;
    image.dx10 = header[21] == 0x30315844;
    if (image.dx10)
    {
        file.read(reinterpret_cast<char*>(header.data() + 32), 20);
        if (!file || header[33] != 3 || (header[34] & 4) || header[35] != 1)
            return false;
        // These numbers are typed DXGI formats. sRGB shares the same block layout
        // as UNORM but must retain its original format tag when the DDS is saved.
        // BC1a allows the encoder to represent BC1 one-bit transparency.
        switch (header[32])
        {
        case 71: case 72: image.format = nvtt::Format_BC1; break;
        case 74: case 75: image.format = nvtt::Format_BC2; break;
        case 77: case 78: image.format = nvtt::Format_BC3; break;
        case 80: image.format = nvtt::Format_BC4; break;
        case 81: image.format = nvtt::Format_BC4S; break;
        case 83: image.format = nvtt::Format_BC5; break;
        case 84: image.format = nvtt::Format_BC5S; break;
        case 95: image.format = nvtt::Format_BC6U; break;
        case 96: image.format = nvtt::Format_BC6S; break;
        case 98: case 99: image.format = nvtt::Format_BC7; break;
        default: return false;
        }
        image.srgb = header[32] == 72 || header[32] == 75 || header[32] == 78 || header[32] == 99;
    }
    else
    {
        switch (header[21])
        {
        case 0x31545844: image.format = nvtt::Format_BC1a; break; // DXT1
        case 0x33545844: image.format = nvtt::Format_BC2; break; // DXT3
        case 0x35545844: image.format = nvtt::Format_BC3; break; // DXT5
        case 0x31495441: case 0x55344342: image.format = nvtt::Format_BC4; break;
        case 0x53344342: image.format = nvtt::Format_BC4S; break;
        case 0x32495441: case 0x55354342: image.format = nvtt::Format_BC5; break;
        case 0x53354342: image.format = nvtt::Format_BC5S; break;
        default: return false;
        }
    }
    // Validate dimensions using wide arithmetic before narrowing to NVTT's int API.
    // The conservative pixel limit leaves room for a four-channel float working image.
    const uint64_t width = header[4], height = header[3];
    if (!width || !height || width > INT32_MAX || height > INT32_MAX
        || width > INT32_MAX / 16 / height)
        return false;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);

    const bool eight_bytes = image.format == nvtt::Format_BC1a 
        || image.format == nvtt::Format_BC1
        || image.format == nvtt::Format_BC4 
        || image.format == nvtt::Format_BC4S;
    
    const size_t size = ((width + 3) / 4) * ((height + 3) / 4) * (eight_bytes ? 8 : 16);
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || file_size < (image.dx10 ? 148 : 128) + size)
        return false;
    image.blocks.resize(size);
    file.read(reinterpret_cast<char*>(image.blocks.data()), size);
    return static_cast<bool>(file);
}

// NVTT's output interface keeps compression in memory so disk writes are not timed.
// NVTT calls this interface while encoding. The callbacks borrow NVTT's temporary
// data, so writeData must copy it into owned storage before returning.
// Image boundaries need no extra records: DDS stores mip payloads consecutively.
struct MemoryOutput : nvtt::OutputHandler
{
    std::vector<uint8_t> bytes;
    void beginImage(int, int, int, int, int, int) override {}
    void endImage() override {}
    bool writeData(const void* data, int size) override
    {
        // ProcessFile reserves the complete output size before compression starts.
        // Reject unexpected output growth rather than reallocating in the timed callback.
        if (size < 0 || bytes.size() + size > bytes.capacity())
            return false;
        if (size)
        {
            const auto* first = static_cast<const uint8_t*>(data);
            bytes.insert(bytes.end(), first, first + size);
        }
        return true;
    }
};

// source retains the original blocks while image is progressively downsampled.
// Context is borrowed from main and reused across files; Surface/output live per file.
struct BatchTimes
{
    double decompress = 0;
    double downsample = 0;
    double compress = 0;
    double pipeline = 0;
};

void WriteCsvRow(std::ostream& csv, const std::string& name, const BatchTimes& times, bool combined)
{
    csv << '"';
    for (char c : name)
    {
        if (c == '"')
            csv << '"';
        csv << c;
    }
    csv << "\"," << times.decompress << ',';
    if (combined)
        csv << times.pipeline << ',' << times.decompress + times.pipeline << '\n';
    else
        csv << times.downsample << ',' << times.compress << ','
            << times.decompress + times.downsample + times.compress << '\n';
}

bool OpenCsv(std::ofstream& csv, const std::filesystem::path& path, bool combined)
{
    csv.open(path);
    csv.imbue(std::locale::classic());
    csv << std::fixed << std::setprecision(6);
    csv << (combined
        ? "file name,decompression time,downsampling + compression time,total generation time\n"
        : "file name,decompression time,downsampling time,compression time,total generation time\n");
    if (!csv)
    {
        std::cerr << "Cannot create CSV: " << path << '\n';
        return false;
    }

    return true;
}

int ProcessFile(const std::filesystem::path &input, const std::filesystem::path &output,
                nvtt::Context &context, nvtt::Quality quality, BatchTimes* total = nullptr, bool warmup = false)
{
    DDSImage source;
    if (!ReadDDS(input, source))
    {
        std::cerr << "Input must be a typed BC1-BC7 single 2D DDS with only Mip 0.\n";
        return 1;
    }
    
    // shortcut: reject signed BC4/BC5 until a newer SDK preserves negative values.
    if (source.format == nvtt::Format_BC4S || source.format == nvtt::Format_BC5S)
    {
        std::cerr << "Signed BC4/BC5 are excluded: NVTT 3.2.5 does not preserve negative values in this pipeline.\n";
        return 1;
    }

    // Keep both paths: NVTT CPU decoding, or CUDA texture hardware decoding.
    // GPU decode uploads compressed blocks and leaves float pixels resident on the GPU.
    const bool gpu = context.isCudaAccelerationEnabled();
    
    nvtt::Surface image;
    const auto decode_start = GetTime();
    
    const bool decoded = gpu
        ? DecompressGpu(source.format, source.width, source.height, source.blocks.data(), image)
        : image.setImage2D(source.format, source.width, source.height, source.blocks.data());
    
    const double decode_ms = (GetTime() - decode_start) / 1000.0;
    if (!decoded)
    {
        std::cerr << "NVTT decompress failed.\n";
        return 1;
    }

    nvtt::CompressionOptions compression;
    compression.setFormat(source.format);
    compression.setQuality(quality);
    
    // Match the sample's numeric interpretation for HDR encoders.
    if (source.format == nvtt::Format_BC6U)
        compression.setPixelType(nvtt::PixelType_UnsignedFloat);
    
    else if (source.format == nvtt::Format_BC6S)
        compression.setPixelType(nvtt::PixelType_Float);
    
    // countMipmaps includes Mip 0. estimateSize covers all compressed level payloads;
    // add the 148-byte DDS10 header separately when reserving output storage.
    const int levels = image.countMipmaps();
    const int size = context.estimateSize(image, levels, compression);
    if (size <= 0)
    {
        std::cerr << "Invalid estimated output size.\n";
        return 1;
    }
    
    MemoryOutput memory;
    memory.bytes.reserve(148 + static_cast<size_t>(size));
    
    nvtt::OutputOptions options;
    options.setContainer(nvtt::Container_DDS10);
    options.setSrgbFlag(source.srgb);
    options.setOutputHandler(&memory);
    // Let NVTT write the DDS header, then insert the original compressed Mip 0.
    if (!context.outputHeader(image, levels, compression, options) || memory.bytes.size() != 148)
    {
        std::cerr << "NVTT DDS header failed.\n";
        return 1;
    }
    // NVTT generated dimensions and mip count; restore the original DX10 tags only.
    // This preserves sRGB and alpha-mode metadata without replacing the new mip count.
    if (source.dx10)
        std::memcpy(memory.bytes.data() + 128, source.header.data() + 32, 20);
    if (!memory.writeData(source.blocks.data(), static_cast<int>(source.blocks.size())))
        return 1;

    // Decoded pixels are already GPU-resident. Measure filtering, encoding and completion.
    // Per-mip synchronization would prevent the library from overlapping GPU work.
    const auto pipeline_start = GetTime();

    // Mip 0 is already in memory as its exact original blocks, so begin at level 1.
    // Each buildNextMipmap replaces the current Surface with its next smaller image.
    // This avoids keeping an entire decoded chain resident at once.
    double mip_ms = 0, encode_ms = 0;
    if (source.srgb && levels > 1)
    {
        const auto start = GetTime();
        image.toLinearFromSrgb();
        mip_ms += (GetTime() - start) / 1000.0;
    }
    for (int mip = 1; mip < levels; ++mip)
    {
        // Keep the working mip linear; only the encoder input returns to sRGB.
        const auto mip_start = GetTime();
        const bool downsampled = image.buildNextMipmap(nvtt::MipmapFilter_Box);
    
        nvtt::Surface encodeImage = source.srgb ? image.clone() : image;
        if (source.srgb)
            encodeImage.toSrgb();
        mip_ms += (GetTime() - mip_start) / 1000.0;
        if (!downsampled)
        {
            std::cerr << "NVTT downsampling failed at mip " << mip << '\n';
            return 1;
        }
    
        const auto encode_start = GetTime();
        // Face 0 is the single 2D image; mip identifies this level in the output stream.
        // Compression includes the callback copies and any library-internal transfers.
        const bool compressed = context.compress(encodeImage, 0, mip, compression, options);
        encode_ms += (GetTime() - encode_start) / 1000.0;
        if (!compressed)
        {
            std::cerr << "NVTT compression failed at mip " << mip << '\n';
            return 1;
        }
    }

    if (gpu && levels > 1)
    {
        // Wait without downloading the final uncompressed mip just for timing.
        // The NVTT operations above leave their CUDA context current on this thread.
        const CUresult status = cuCtxSynchronize();
        if (status != CUDA_SUCCESS)
        {
            std::cerr << "CUDA completion failed: " << static_cast<int>(status) << '\n';
            return 1;
        }
    }
    const double pipeline_ms = levels > 1 ? (GetTime() - pipeline_start) / 1000.0 : 0;

    // Warm-up completes the same processing path without saving or reporting.
    if (warmup)
        return 0;

    // Write the completed NVTT output only after all timed calls have finished.
    std::ofstream file(output, std::ios::binary);
    file.write(reinterpret_cast<const char*>(memory.bytes.data()), memory.bytes.size());
    file.flush();
    if (!file)
    {
        std::cerr << "DDS save failed: " << output << '\n';
        return 1;
    }
    
    if (total)
    {
        total->decompress = decode_ms;
        total->downsample = mip_ms;
        total->compress = encode_ms;
        total->pipeline = pipeline_ms;
    }
    
    return 0;
}

// Extension matching is case-insensitive, so .DDS and .dds are treated alike.
// unsigned char keeps the argument in the valid range required by std::tolower.
char Lowercase(unsigned char c)
{
    return static_cast<char>(std::tolower(c));
}

// Refuse an existing path to protect previous results and the original input.
// This is a preflight check, not an atomic reservation against other processes.
bool CanWrite(const std::filesystem::path &path)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error || exists)
    {
        std::cerr << "Cannot write output: " << path << '\n';
        return false;
    }
    return true;
}

// Batch handling is outside the processing timers. Only this directory is scanned.
// Sort before processing to make the input order repeatable between runs.
// Files are processed one at a time; codec-internal threads may still run in parallel.
// Continue after a file fails, then return failure if any file was skipped or failed.
int ProcessBatch(const std::filesystem::path &folder, const std::filesystem::path &output,
                 nvtt::Context &context, nvtt::Quality quality)
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
    if (!OpenCsv(csv, output / "timings.csv", context.isCudaAccelerationEnabled()))
        return 1;

    BatchTimes total;
    size_t succeeded = 0;
    bool warmed_up = false;
    for (const auto &file : files)
    {
        std::cout << "Processing: " << file << '\n';
        const auto destination = output / file.filename();
        if (!CanWrite(destination))
            continue;
        if (!warmed_up)
        {
            if (ProcessFile(file, destination, context, quality, nullptr, true) != 0)
                continue;
            warmed_up = true;
            std::cout << "Warm-up completed (excluded from averages).\n";
        }
        BatchTimes times;
        if (ProcessFile(file, destination, context, quality, &times) == 0)
        {
            WriteCsvRow(csv, file.filename().u8string(), times, context.isCudaAccelerationEnabled());
            total.decompress += times.decompress;
            total.downsample += times.downsample;
            total.compress += times.compress;
            total.pipeline += times.pipeline;
            ++succeeded;
        }
    }
    std::cout << "Succeeded: " << succeeded << ", failed: " << files.size() - succeeded << '\n';
    if (succeeded > 0)
    {
        WriteCsvRow(csv, "average", {total.decompress / succeeded, total.downsample / succeeded, total.compress / succeeded, total.pipeline / succeeded}, context.isCudaAccelerationEnabled());
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

// Remove a trailing --gpu before parsing the optional quality name.
// Explicitly request CPU by default; do not let CUDA availability change that baseline.
int main(int argc, char** argv)
{
    bool gpu = argc > 3 && std::strcmp(argv[argc - 1], "--gpu") == 0;
    if (gpu)
        --argc;
    
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: nvtt <input.dds|folder|folder/*.dds> <output.dds|folder> [fastest|normal|production|highest] [--gpu]\n";
        return 1;
    }
    
    nvtt::Quality quality = nvtt::Quality_Normal;
    if (argc == 4)
    {
        if (std::strcmp(argv[3], "fastest") == 0) quality = nvtt::Quality_Fastest;
        else if (std::strcmp(argv[3], "normal") == 0) quality = nvtt::Quality_Normal;
        else if (std::strcmp(argv[3], "production") == 0) quality = nvtt::Quality_Production;
        else if (std::strcmp(argv[3], "highest") == 0) quality = nvtt::Quality_Highest;
        else
        {
            std::cerr << "Unknown quality setting.\n";
            return 1;
        }
    }
    
    // Reuse the library's own context and threading; initialize CUDA outside the timers.
    nvtt::Context context(gpu);
    if (gpu && !context.isCudaAccelerationEnabled())
    {
        std::cerr << "CUDA acceleration is unavailable.\n";
        return 1;
    }
    
    if (gpu)
    {
        // Resolve the delayed driver import and initialize it outside image timers.
        const CUresult status = cuInit(0);
        if (status != CUDA_SUCCESS)
        {
            std::cerr << "CUDA driver initialization failed: " << static_cast<int>(status) << '\n';
            return 1;
        }
    }
    
    const std::filesystem::path input(argv[1]), output(argv[2]);
    if (input.filename() == "*.dds")
        return ProcessBatch(input.has_parent_path() ? input.parent_path() : ".", output, context, quality);
    
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
        return ProcessBatch(input, output, context, quality);
    if (!CanWrite(output))
        return 1;
    if (ProcessFile(input, output, context, quality, nullptr, true) != 0)
        return 1;
    
    std::cout << "Warm-up completed (excluded from averages).\n";
    
    std::ofstream csv;
    auto csv_path = output;
    csv_path.replace_extension(".timings.csv");
    if (!OpenCsv(csv, csv_path, context.isCudaAccelerationEnabled()))
        return 1;
    
    BatchTimes times;
    const int result = ProcessFile(input, output, context, quality, &times);
    if (result == 0)
    {
        WriteCsvRow(csv, input.filename().u8string(), times, context.isCudaAccelerationEnabled());
        WriteCsvRow(csv, "average", times, context.isCudaAccelerationEnabled());
    }
    
    csv.close();
    if (!csv)
    {
        std::cerr << "CSV write failed.\n";
        return 1;
    }
    
    return result;
}
