// etcpak benchmark: use upstream filtering, encoders, and TaskDispatch directly.
// The small DDS reader/writer keeps original headers and compressed Mip 0 available.
// CPU AVX2 is required; this executable does not offer a GPU mode.
// Workarounds below describe limitations in the checked-out upstream revision.

#include "BitmapDownsampled.hpp"
#include "Decode.hpp"
#include "ProcessDxtc.hpp"
#include "System.hpp"
#include "TaskDispatch.hpp"
#include "TextureHeader.hpp"
#include "bc7enc.h"
#include "bcdec.h"
#include "common/timer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif

// Keep the DDS header unchanged except for mip count and mip-related flags.
// blocks stores compressed bytes in 64-bit words, matching etcpak's block interfaces.
// header includes space for the optional DX10 extension; header_size says how much to save.
struct DDSImage
{
    std::array<uint32_t, 37> header{};
    size_t header_size = 128;
    CodecType codec = Bc1;
    int width = 0;
    int height = 0;
    std::vector<uint64_t> blocks;
};

// Bitmap(size) leaves this field unset in upstream; initialize it without editing the library.
class InputBitmap : public Bitmap
{
public:
    explicit InputBitmap(const v2i &size) : Bitmap(size) { m_alpha = true; }
};

// width/height are the logical DDS dimensions. pixels may have extra rows/columns
// to satisfy 4x4 block processing. Keep these two sizes separate for tiny and odd mips.
// unique_ptr owns the Bitmap; blocks owns the newly encoded mip, except at Mip 0.
struct MipImage
{
    int width;
    int height;
    std::unique_ptr<Bitmap> pixels;
    std::vector<uint64_t> blocks;
};

// CPU support alone is insufficient: the OS must also preserve AVX register state.
// Check OSXSAVE and XCR0 before calling code compiled with AVX2 instructions.
bool HasAVX2()
{
#ifdef _MSC_VER
    int registers[4];
    __cpuid(registers, 0);
    if (registers[0] < 7)
        return false;
    __cpuidex(registers, 1, 0);
    const int avx_and_osxsave = (1 << 28) | (1 << 27);
    if ((registers[2] & avx_and_osxsave) != avx_and_osxsave || (_xgetbv(0) & 6) != 6)
        return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

// BC1/BC4 use 8 bytes per 4x4 block; BC3/BC5/BC7 use 16 bytes.
// Return 64-bit word counts because block buffers are vector<uint64_t>.
size_t BlockWords(CodecType codec)
{
    return codec == Bc1 || codec == Bc4 ? 1 : 2;
}

// Read one typed, block-compressed 2D image with no existing lower mips.
// Validate the layout and payload length before allocating or calling a decoder.
// Header indices include the magic word: [3]/[4] are height/width, [7] is mip count,
// [21] is FourCC, and [32..36] are the optional DX10 extension.
bool ReadDDS(const std::filesystem::path &path, DDSImage &image)
{
    std::ifstream file(path, std::ios::binary);
    auto &header = image.header;
    file.read(reinterpret_cast<char*>(header.data()), 128);
    if (!file || header[0] != 0x20534444 || header[1] != 124 || header[19] != 32
        || !(header[20] & 4) || (header[28] & 0x200200)
        || ((header[2] & 0x20000) && header[7] != 1))
        return false;

    if (header[21] == 0x30315844) // DX10: preserve format, color space, and alpha mode.
    {
        file.read(reinterpret_cast<char*>(header.data() + 32), 20);
        if (!file || header[33] != 3 || (header[34] & 4) || header[35] != 1)
            return false;
        image.header_size = 148;
        switch (header[32])
        {
        case 71: case 72: image.codec = Bc1; break;
        case 77: case 78: image.codec = Bc3; break;
        case 80: image.codec = Bc4; break;
        case 83: image.codec = Bc5; break;
        case 98: case 99: image.codec = Bc7; break;
        default: return false;
        }
    }
    else
    {
        switch (header[21])
        {
        case 0x31545844: image.codec = Bc1; break; // DXT1
        case 0x35545844: image.codec = Bc3; break; // DXT5
        case 0x31495441: case 0x55344342: image.codec = Bc4; break; // ATI1 / BC4U
        case 0x32495441: case 0x55354342: image.codec = Bc5; break; // ATI2 / BC5U
        default: return false;
        }
    }
    const uint64_t width = header[4], height = header[3];
    const uint64_t padded_pixels = ((width + 7) & ~7ull) * ((height + 7) & ~7ull);
    // Bitmap uses signed int products for allocation; reject overflowing dimensions.
    if (!width || !height || width > INT32_MAX - 8 || height > INT32_MAX - 8
        || padded_pixels > INT32_MAX / sizeof(uint32_t))
        return false;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    const size_t words = ((width + 3) / 4) * ((height + 3) / 4) * BlockWords(image.codec);
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error || file_size < image.header_size + words * sizeof(uint64_t))
        return false;
    image.blocks.resize(words);
    file.read(reinterpret_cast<char*>(image.blocks.data()), words * sizeof(uint64_t));
    return static_cast<bool>(file);
}

// Crop the unused odd edge for floor-half filtering, then fill any padding with edge pixels.
// source uses its own row stride; destination uses destination.Size().x.
// Clamp the source row and repeat the final pixel to initialize the whole padding.
// min(width, stride) also handles cropping an odd input edge for floor-half filtering.
// In-place use only refreshes padding; it skips memcpy when both rows are identical.
void CopyWithEdges(const Bitmap &source, int width, int height, Bitmap &destination)
{
    const auto* input = source.Data();
    auto* output = destination.Data();
    const int stride = destination.Size().x;
    const int copied_width = std::min(width, stride);
    for (int y = 0; y < destination.Size().y; ++y)
    {
        const auto* row = input + std::min(y, height - 1) * source.Size().x;
        auto* target = output + y * stride;
        if (target != row)
            std::memcpy(target, row, copied_width * sizeof(uint32_t));
        std::fill(target + copied_width, target + stride, row[copied_width - 1]);
    }
}

// The destination is rounded up to whole 4x4 blocks, so block decoders can write
// all 16 pixels without running past a partial final row or column.
// BC1 uses the bundled fallback; the other supported codecs use native etcpak decoders.
void Decompress(const DDSImage &source, Bitmap &pixels)
{
    const auto* blocks = source.blocks.data();
    auto* output = pixels.Data();
    const auto size = pixels.Size();
    switch (source.codec)
    {
    case Bc1:
        // shortcut: upstream DecodeBc1 writes only one row per block; use bundled bcdec until fixed.
        for (int y = 0; y < size.y; y += 4)
            for (int x = 0; x < size.x; x += 4)
                bcdec_bc1(blocks++, output + y * size.x + x, size.x * sizeof(uint32_t));
        break;
    case Bc3: DecodeBc3(blocks, output, size.x, size.y); break;
    case Bc4: DecodeBc4(blocks, output, size.x, size.y); break;
    case Bc5: DecodeBc5(blocks, output, size.x, size.y); break;
    case Bc7: DecodeBc7(blocks, output, size.x, size.y); break;
    default: break;
    }
}

// DDS dimensions halve with integer division and never become smaller than one.
// The upstream filter expects sufficiently large input, so provide a padded temporary
// at twice the block-rounded output size. Edge preparation is outside the timer.
// Keep the library's filtering work, including its async completion, inside the timer.
void GenerateMips(std::vector<MipImage> &mips, double &milliseconds)
{
    while (mips.back().width > 1 || mips.back().height > 1)
    {
        const auto &previous = mips.back();
        const int width = std::max(1, previous.width / 2);
        const int height = std::max(1, previous.height / 2);
        // Avoid upstream's zero-filled sub-4x4 branch; all filtering remains in etcpak.
        InputBitmap input(v2i(((width + 3) & ~3) * 2, ((height + 3) & ~3) * 2));
        CopyWithEdges(*previous.pixels, previous.width, previous.height, input);
        const auto start = GetTime();
        auto output = std::make_unique<BitmapDownsampled>(input, 1, false);
        output->Data(); // Wait for the library's async Box filter before stopping the timer.
        milliseconds += (GetTime() - start) / 1000.0;
        CopyWithEdges(*output, width, height, *output);
        mips.push_back({width, height, std::move(output), {}});
    }
}

// A queued task borrows a disjoint block range until TaskDispatch::Sync() completes.
// A plain callable object avoids a lambda while using the library's task interface.
// input/output are borrowed ranges; parameters points to shared read-only BC7 settings.
// Each task writes different blocks, so no output mutex is needed.
struct CompressTask
{
    CodecType codec;
    const uint32_t* input;
    uint64_t* output;
    uint32_t blocks;
    size_t width;
    const bc7enc_compress_block_params* parameters;

    void operator()() const
    {
        switch (codec)
        {
        case Bc1: CompressBc1(input, output, blocks, width); break;
        case Bc3: CompressBc3(input, output, blocks, width); break;
        case Bc4: CompressBc4(input, output, blocks, width); break;
        case Bc5: CompressBc5(input, output, blocks, width); break;
        case Bc7: CompressBc7(input, output, blocks, width, parameters); break;
        default: break;
        }
    }
};

// Allocate all lower-mip buffers first so vector growth cannot invalidate pointers
// held by queued tasks. Keep settings and pixel buffers alive until Sync returns.
// Timing includes task submission, encoding, and waiting for all workers to finish.
void CompressMips(std::vector<MipImage> &mips, CodecType codec, double &milliseconds)
{
    if (mips.size() == 1)
        return;
    // Allocate all output buffers before submitting tasks or starting the timer.
    for (size_t mip = 1; mip < mips.size(); ++mip)
    {
        auto &image = mips[mip];
        const auto size = image.pixels->Size();
        const uint32_t blocks = (size.x / 4) * (size.y / 4);
        image.blocks.resize(blocks * BlockWords(codec));
    }
    bc7enc_compress_block_params parameters;
    bc7enc_compress_block_params_init(&parameters);
    const auto start = GetTime();
    for (size_t mip = 1; mip < mips.size(); ++mip)
    {
        auto &image = mips[mip];
        const auto size = image.pixels->Size();
        const int blocks_per_row = size.x / 4;
        // Use the upstream CLI's base chunk size: 32 rows of 4x4 blocks per task.
        for (int row = 0; row < size.y / 4; row += 32)
        {
            const int rows = std::min(32, size.y / 4 - row);
            // row counts BLOCK rows, hence row * 4 pixel rows for the input offset.
            // Output offsets count blocks, then convert blocks to 64-bit words.
            TaskDispatch::Queue(CompressTask{
                codec,
                image.pixels->Data() + row * 4 * size.x,
                image.blocks.data() + row * blocks_per_row * BlockWords(codec),
                static_cast<uint32_t>(rows * blocks_per_row),
                static_cast<size_t>(size.x), &parameters});
        }
    }
    // Wait before reading output or releasing the borrowed buffers and BC7 parameters.
    TaskDispatch::Sync();
    milliseconds = (GetTime() - start) / 1000.0;
}

// Preserve source metadata, update mip-related fields, and write levels in DDS order.
// The base payload comes from source.blocks; only mip 1 onward comes from the encoder.
// All file writes and flushes happen after the processing timers have stopped.
bool SaveDDS(const std::filesystem::path &path, const DDSImage &source, const std::vector<MipImage> &mips)
{
    auto header = source.header;
    header[7] = static_cast<uint32_t>(mips.size());
    if (mips.size() > 1)
    {
        header[2] |= 0x20000; // DDSD_MIPMAPCOUNT
        header[27] |= 0x400008; // DDSCAPS_MIPMAP | DDSCAPS_COMPLEX
    }
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header.data()), source.header_size);
    file.write(reinterpret_cast<const char*>(source.blocks.data()), source.blocks.size() * sizeof(uint64_t));
    for (size_t mip = 1; mip < mips.size(); ++mip)
        file.write(reinterpret_cast<const char*>(mips[mip].blocks.data()), mips[mip].blocks.size() * sizeof(uint64_t));
    file.flush();
    return static_cast<bool>(file);
}

// Allocate a block-rounded decode target outside the decode timer.
// Moving pixels into mips transfers ownership; no decoded pixel copy is made.
// Retain all pixel levels until compression tasks have completed.
int ProcessFile(const std::filesystem::path &input, const std::filesystem::path &output)
{
    DDSImage source;
    if (!ReadDDS(input, source))
    {
        std::cerr << "Input must be a single 2D BC1/BC3/BC4/BC5/BC7 DDS with only Mip 0.\n";
        return 1;
    }
    auto pixels = std::make_unique<InputBitmap>(v2i((source.width + 3) & ~3, (source.height + 3) & ~3));
    const auto start = GetTime();
    Decompress(source, *pixels);
    const double decode_ms = (GetTime() - start) / 1000.0;
    std::vector<MipImage> mips;
    mips.push_back({source.width, source.height, std::move(pixels), {}});
    double mip_ms = 0, encode_ms = 0;
    GenerateMips(mips, mip_ms);
    CompressMips(mips, source.codec, encode_ms);
    if (!SaveDDS(output, source, mips))
    {
        std::cerr << "DDS save failed: " << output << '\n';
        return 1;
    }
    std::cout << "Compression backend: etcpak CPU AVX2 + TaskDispatch\n"
              << "Compression threads: " << System::CPUCores() << " (including caller)\n"
              << "Decompress implementation: " << (source.codec == Bc1 ? "bundled bcdec" : "etcpak") << '\n'
              << "BC7 settings: upstream defaults\n"
              << "Mip levels: " << mips.size() << '\n'
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
int ProcessBatch(const std::filesystem::path &folder, const std::filesystem::path &output)
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
    size_t succeeded = 0;
    for (const auto &file : files)
    {
        std::cout << "Processing: " << file << '\n';
        const auto destination = output / file.filename();
        if (CanWrite(destination) && ProcessFile(file, destination) == 0)
            ++succeeded;
    }
    std::cout << "Succeeded: " << succeeded << ", failed: " << files.size() - succeeded << '\n';
    return succeeded == files.size() ? 0 : 1;
}

// Initialize codec tables and one upstream worker pool before processing files.
// Their setup cost is excluded from the reported per-image processing times.
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: etcpak <input.dds|folder|folder/*.dds> <output.dds|folder>\n";
        return 1;
    }
    // The driver is compiled without AVX2 so this guard runs safely on older CPUs.
    if (!HasAVX2())
    {
        std::cerr << "This etcpak build requires CPU and OS support for AVX2.\n";
        return 1;
    }
    bc7enc_compress_block_init(); // Initialize lookup tables once, outside all timers.
    TaskDispatch dispatcher(System::CPUCores()); // Reuse the upstream worker pool across files.
    const std::filesystem::path input(argv[1]), output(argv[2]);
    if (input.filename() == "*.dds")
        return ProcessBatch(input.has_parent_path() ? input.parent_path() : ".", output);
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
        return ProcessBatch(input, output);
    return CanWrite(output) ? ProcessFile(input, output) : 1;
}
