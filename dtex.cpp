// DirectXTex benchmark: read DDS -> decode -> generate Box mips -> encode lower mips.
// Preserve the original compressed Mip 0 instead of recompressing it.
// Timers surround processing calls only; allocations inside those calls are included.
// IMPORTANT: the SaveToDDSFile call below is currently commented out.
// A successful return therefore means processing succeeded, not that a DDS was saved.

#include <d3d11.h>
#include <wrl/client.h>
#include <DirectXTex.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cctype>
#include <string>
#include <fstream>
#include <iomanip>
#include <locale>
#include <objbase.h>

#include "common/timer.h"

// Extension matching is case-insensitive, so .DDS and .dds are treated alike.
// unsigned char keeps the argument in the valid range required by std::tolower.
char Lowercase(unsigned char c)
{
    return static_cast<char>(std::tolower(c));
}

// A null device selects CPU compression; a valid device selects GPU compression.
// ScratchImage objects own their pixel buffers and release them on every return path.
// The device is borrowed from main and must remain alive for the whole call.
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
    csv << "\"," << times.decompress << ',' << times.downsample << ',' << times.compress << ',' << times.decompress + times.downsample + times.compress << '\n';
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

int ProcessFile(const std::filesystem::path &input_path, const std::filesystem::path &output_path,
                ID3D11Device* device, BatchTimes* total = nullptr, bool warmup = false)
{
    // 1. Load and validate the input. File I/O is outside all processing timers.
    // source owns the compressed pixels; metadata describes their layout.
    DirectX::TexMetadata metadata{};
    DirectX::ScratchImage source;

    const HRESULT hr = DirectX::LoadFromDDSFile(
        input_path.c_str(),
        DirectX::DDS_FLAGS_NONE,
        &metadata,
        source);

    if (FAILED(hr))
    {
        std::cerr << "DDS load failed: 0x" << std::hex << static_cast<unsigned long>(hr) << '\n';
        return 1;
    }

    // This benchmark accepts one BC-compressed 2D texture with only Mip 0.
    if (!DirectX::IsCompressed(metadata.format))
    {
        std::cerr << "Input must be BC compressed" << '\n';
        return 1;
    }

    if (metadata.mipLevels != 1)
    {
        std::cerr << "Input must contain only Mip 0." << '\n';
        return 1;
    }

    // Reject unsupported GPU formats instead of silently falling back to the CPU.
    if (device && metadata.format != DXGI_FORMAT_BC6H_UF16
        && metadata.format != DXGI_FORMAT_BC6H_SF16
        && metadata.format != DXGI_FORMAT_BC7_UNORM
        && metadata.format != DXGI_FORMAT_BC7_UNORM_SRGB)
    {
        std::cerr << "GPU compression requires BC6H or BC7.\n";
        return 1;
    }

    if (metadata.dimension != DirectX::TEX_DIMENSION_TEXTURE2D || metadata.arraySize != 1)
    {
        std::cerr << "Input must be a single 2D texture.\n";
        return 1;
    }

    // 2. Decompress on the CPU, including when GPU compression is selected.
    // UNKNOWN lets DirectXTex choose the decoded format, including HDR for BC6H.
    DirectX::ScratchImage decompressed;

    const auto decompress_start = GetTime();
    const HRESULT decompress_hr = DirectX::Decompress(
        source.GetImages(), source.GetImageCount(), source.GetMetadata(),
        DXGI_FORMAT_UNKNOWN, decompressed);
    const double decompress_ms = (GetTime() - decompress_start) / 1000.0;

    if (FAILED(decompress_hr))
    {
        std::cerr << "Decompress failed: 0x" << std::hex << static_cast<unsigned long>(decompress_hr) << '\n';
        return 1;
    }

    // 3. Generate the full Box mip chain on the CPU (0 requests all levels).
    // mip_chain owns the decoded Mip 0 and every lower mip.
    // Color-space handling is left to DirectXTex's flags and decoded format.
    DirectX::ScratchImage mip_chain;
    const auto downsample_start = GetTime();
    
    const HRESULT mip_hr = DirectX::GenerateMipMaps(
        decompressed.GetImages(), decompressed.GetImageCount(),
        decompressed.GetMetadata(),
        DirectX::TEX_FILTER_BOX, 0, mip_chain);

    const double downsample_ms = (GetTime() - downsample_start) / 1000.0;

    if (FAILED(mip_hr))
    {
        std::cerr << "GenerateMipMaps failed: 0x" << std::hex << static_cast<unsigned long>(mip_hr) << '\n';
        return 1;
    }

    // 4. Compress only the lower mips. Keep the original compressed Mip 0.
    DirectX::ScratchImage compressed;
    auto lower_metadata = mip_chain.GetMetadata();

    // A 1x1 input has no lower mip levels, so its compression time stays zero.
    double compress_ms = 0.0;

    if (lower_metadata.mipLevels > 1)
    {
        // Skip Mip 0 and describe Mip 1 as the base of the chain to compress.
        // Advance by one Image descriptor, then match metadata to the smaller base.
        const DirectX::Image *first_lower_mip = mip_chain.GetImages() + 1;
        lower_metadata.width = first_lower_mip->width;
        lower_metadata.height = first_lower_mip->height;
    
        --lower_metadata.mipLevels;

        // Use the library's CPU parallel implementation instead of adding file workers.
        // GPU compression uses the separate D3D11 overload and its own default flags.
        DirectX::TEX_COMPRESS_FLAGS flags = DirectX::TEX_COMPRESS_DEFAULT;
        if (!device)
            flags |= DirectX::TEX_COMPRESS_PARALLEL;

        if (metadata.format == DXGI_FORMAT_BC7_UNORM || metadata.format == DXGI_FORMAT_BC7_UNORM_SRGB)
        {
            // BC7 QUICK reduces the encoder search for speed. It changes the quality
            // tradeoff, so do not treat this as equal quality to another library's default.
            flags |= DirectX::TEX_COMPRESS_BC7_QUICK;
        }

        // Time the whole API call, including GPU transfers when using a device.
        const auto compress_start = GetTime();
        HRESULT compress_hr;
        if (device)
        {
            compress_hr = DirectX::Compress(
                device, first_lower_mip, lower_metadata.mipLevels, lower_metadata,
                metadata.format, flags,
                DirectX::TEX_ALPHA_WEIGHT_DEFAULT, compressed);
        }
        else
        {
            compress_hr = DirectX::Compress(
                first_lower_mip, lower_metadata.mipLevels, lower_metadata,
                metadata.format, flags,
                DirectX::TEX_THRESHOLD_DEFAULT, compressed);
        }
    
        compress_ms = (GetTime() - compress_start) / 1000.0;
        if (FAILED(compress_hr))
        {
            std::cerr << "Compress failed: 0x" << std::hex << static_cast<unsigned long>(compress_hr) << '\n';
            return 1;
        }
    }

    // 5. Assemble the output: original Mip 0 followed by newly compressed mips.
    // Image is a view: copying it copies pointers and pitches, not the pixel storage.
    // source and compressed must outlive this list and any future SaveToDDSFile call.
    std::vector<DirectX::Image> images{source.GetImages()[0]};
    if (compressed.GetImageCount() > 0)
    {
        images.insert(images.end(), compressed.GetImages(), compressed.GetImages() + compressed.GetImageCount());
    }

    auto output_metadata = metadata;
    output_metadata.mipLevels = images.size();

    if (warmup)
        return 0;

    const HRESULT save_hr = DirectX::SaveToDDSFile(
        images.data(), images.size(), output_metadata,
        DirectX::DDS_FLAGS_FORCE_DX10_EXT, output_path.c_str());

    if (FAILED(save_hr))
    {
        std::cerr << "Save failed: 0x" << std::hex << static_cast<unsigned long>(save_hr) << '\n';
        return 1;
    }

    if (total)
    {
        total->decompress = decompress_ms;
        total->downsample = downsample_ms;
        total->compress = compress_ms;
    }
    return 0;
}

// Batch handling is outside the processing timers. Only this directory is scanned.
// Sort before processing to make the input order repeatable between runs.
// Files are processed one at a time; codec-internal threads may still run in parallel.
// Continue after a file fails, then return failure if any file was skipped or failed.
int ProcessBatch(const std::filesystem::path &folder, const std::filesystem::path &output, ID3D11Device* device)
{
    // Collect first, then process. Keep directory traversal out of the image pipeline.
    std::error_code error;
    std::vector<std::filesystem::path> files;
    std::filesystem::directory_iterator entry(folder, error), end;

    for (; !error && entry != end; entry.increment(error))
    {
        const bool regular_file = entry->is_regular_file(error);
        if (error)
            break;
        
            // Match .dds regardless of letter case.
        auto extension = entry->path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), Lowercase);

        if (regular_file && extension == ".dds")
            files.push_back(entry->path());
    }

    if (error)
    {
        std::cerr << "Cannot read folder " << folder << ": " << error.message() << '\n';
        return 1;
    }

    if (files.empty())
    {
        std::cerr << "No DDS files found.\n";
        return 1;
    }

    std::sort(files.begin(), files.end());

    std::filesystem::create_directories(output, error);
    if (error)
    {
        std::cerr << "Cannot create output folder " << output << ": " << error.message() << '\n';
        return 1;
    }

    std::ofstream csv;
    if (!OpenCsv(csv, output / "timings.csv"))
        return 1;

    BatchTimes total;
    size_t succeeded = 0;
    bool warmed_up = false;
    
    // main processing loop 
    for (const auto &file : files)
    {
        const auto destination = output / file.filename();
        std::cout << std::dec << "Processing: " << file << '\n';
        // Protect existing outputs, including inputs when both folders are the same.
        const bool exists = std::filesystem::exists(destination, error);
        if (error)
        {
            std::cerr << "Cannot check output " << destination << ": " << error.message() << '\n';
            continue;
        }
        if (exists)
        {
            std::cerr << "Output already exists: " << destination << '\n';
            continue;
        }
        if (!warmed_up)
        {
            if (ProcessFile(file, destination, device, nullptr, true) != 0)
                continue;
            warmed_up = true;
            std::cout << "Warm-up completed (excluded from averages).\n";
        }
        BatchTimes times;
        if (ProcessFile(file, destination, device, &times) == 0)
        {
            WriteCsvRow(csv, file.filename().u8string(), times);
            total.decompress += times.decompress;
            total.downsample += times.downsample;
            total.compress += times.compress;
            ++succeeded;
        }
    }

    std::cout << std::dec << "Succeeded: " << succeeded << ", failed: " << files.size() - succeeded << '\n';

    // Skipped files also count as failures in the batch exit status.
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

// Validate arguments, initialize shared runtime state, then choose file or batch mode.
// Device creation and COM initialization happen once and are not benchmarked.
int main(int argc, char **argv)
{
    if ((argc != 3 && argc != 4) || (argc == 4 && std::string(argv[3]) != "--gpu"))
    {
        std::cerr << "Usage: dtex <input.dds> <output.dds> [--gpu]\n"
                  << "       dtex <input-folder|input-folder/*.dds> <output-folder> [--gpu]\n";
        return 1;
    }

    // 1. Initialize COM once for library paths that use WIC.
    // COINIT_MULTITHREADED is an apartment setting, not a worker-thread count.
    const HRESULT com_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(com_hr))
    {
        std::cerr << "COM initialization failed: 0x" << std::hex << static_cast<unsigned long>(com_hr) << '\n';
        return 1;
    }

    // Balance CoInitializeEx even when a later error returns early.
    struct ComCleanup
    {
        ~ComCleanup() { CoUninitialize(); }
    } com_cleanup;

    // 2. Create a shared GPU device only when --gpu is requested.
    // Device creation happens outside the measured calls.
    // ComPtr releases the device automatically when main returns.
    // if we use GPU encoder, 
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    if (argc == 4)
    {
        const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
        const HRESULT device_hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            levels, 1, D3D11_SDK_VERSION, device.GetAddressOf(), nullptr, nullptr);
        
        if (FAILED(device_hr))
        {
            std::cerr << "GPU device creation failed: 0x" << std::hex
                      << static_cast<unsigned long>(device_hr) << '\n';
            return 1;
        }
    }

    // 3. Route a wildcard or folder to batch mode; otherwise process one file.
    // Input and output path define 
    const std::filesystem::path input(argv[1]);
    const std::filesystem::path output(argv[2]);

    // Handle the supported wildcard explicitly; no general glob matching is used.
    if (input.filename() == "*.dds")
    {
        const auto folder = input.has_parent_path() ? input.parent_path() : ".";
        return ProcessBatch(folder, output, device.Get());
    }

    // The error_code overload reports filesystem errors without a try/catch block.
    // An ordinary file goes to ProcessFile; a directory goes to ProcessBatch.
    std::error_code error;
    const bool directory = std::filesystem::is_directory(input, error);
    if (error)
    {
        std::cerr << "Cannot access input " << input << ": " << error.message() << '\n';
        return 1;
    }

    // A folder runs a batch; any other input is treated as one DDS file.
    if (directory)
        return ProcessBatch(input, output, device.Get());

    if (ProcessFile(input, output, device.Get(), nullptr, true) != 0)
        return 1;
    
    std::cout << "Warm-up completed (excluded from averages).\n";
    std::ofstream csv;
    auto csv_path = output;
    csv_path.replace_extension(".timings.csv");
    if (!OpenCsv(csv, csv_path))
        return 1;
    
    BatchTimes times;
    const int result = ProcessFile(input, output, device.Get(), &times);
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
