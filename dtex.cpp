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
int ProcessFile(const std::filesystem::path &input_path, const std::filesystem::path &output_path,
                ID3D11Device* device)
{
    // Metadata describes the layout and format; source owns the compressed bytes.
    // Loading must finish before the decode timer starts to exclude disk access.
    DirectX::TexMetadata meta_data{};
    DirectX::ScratchImage source;

    // Load the DDS before timing so disk I/O is excluded.
    const HRESULT hr = DirectX::LoadFromDDSFile(
        input_path.c_str(),
        DirectX::DDS_FLAGS_NONE,
        &meta_data,
        source);

    // HRESULT encodes success/failure. Do not use a partially loaded ScratchImage.
    if (FAILED(hr))
    {
        std::cerr << "DDS load failed: 0x" << std::hex << static_cast<unsigned long>(hr) << '\n';
        return 1;
    }

    // This benchmark accepts one BC-compressed 2D texture with only Mip 0.
    if (!DirectX::IsCompressed(meta_data.format))
    {
        std::cerr << "Input must be BC compressed" << std::endl;
        return 1;
    }

    if (meta_data.mipLevels != 1)
    {
        std::cerr << "Input must contain only Mip 0." << std::endl;
        return 1;
    }

    // Reject unsupported GPU formats instead of silently falling back to the CPU.
    if (device && meta_data.format != DXGI_FORMAT_BC6H_UF16
        && meta_data.format != DXGI_FORMAT_BC6H_SF16
        && meta_data.format != DXGI_FORMAT_BC7_UNORM
        && meta_data.format != DXGI_FORMAT_BC7_UNORM_SRGB)
    {
        std::cerr << "GPU compression requires BC6H or BC7.\n";
        return 1;
    }

    if (meta_data.dimension != DirectX::TEX_DIMENSION_TEXTURE2D || meta_data.arraySize != 1)
    {
        std::cerr << "Input must be a single 2D texture.\n";
        return 1;
    }

    // Decode on the CPU; UNKNOWN lets DirectXTex choose the output pixel format.
    // UNKNOWN requests the library's default decoded format for the source codec.
    // This lets HDR BC6H use an appropriate representation rather than forcing RGBA8.
    // Decompression is a CPU operation even when GPU compression is selected.
    DirectX::ScratchImage decompressed;
    const auto decompress_start = GetTime();
    const HRESULT decompress_hr = DirectX::Decompress(source.GetImages(), source.GetImageCount(), source.GetMetadata(), DXGI_FORMAT_UNKNOWN, decompressed);
    const double decompress_ms = (GetTime() - decompress_start) / 1000.0;

    if (FAILED(decompress_hr))
    {
        std::cerr << "Decompress failed: 0x" << std::hex << static_cast<unsigned long>(decompress_hr) << std::endl;
        return 1;
    }

    // Generate the full mip chain on the CPU using the Box filter (0 = all levels).
    // The generated chain includes an uncompressed copy of Mip 0 plus lower levels.
    // Keep decompressed alive while the library reads it; mip_chain owns its output.
    // Box selects the filter. Color-space handling is left to DirectXTex's flags/format.
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

    std::cout << "Mip levels: " << mip_chain.GetMetadata().mipLevels << '\n';

    DirectX::ScratchImage compressed;
    auto lower_meta = mip_chain.GetMetadata();

    // A 1x1 input has no lower mip levels, so its compression time stays zero.
    double compress_ms = 0.0;

    if (lower_meta.mipLevels > 1)
    {
        // Skip Mip 0 and describe Mip 1 as the base of the chain to compress.
        // Image pointer arithmetic advances by one image descriptor, not one pixel.
        // Passing the original full-size metadata here would describe the wrong chain.
        const DirectX::Image *first = mip_chain.GetImages() + 1;
        lower_meta.width = first->width;
        lower_meta.height = first->height;
        --lower_meta.mipLevels;

        // Use the library's CPU parallel implementation instead of adding file workers.
        // GPU compression uses the separate D3D11 overload and its own default flags.
        DirectX::TEX_COMPRESS_FLAGS flags = device ? DirectX::TEX_COMPRESS_DEFAULT : DirectX::TEX_COMPRESS_PARALLEL;

        if (meta_data.format == DXGI_FORMAT_BC7_UNORM || meta_data.format == DXGI_FORMAT_BC7_UNORM_SRGB)
        {
            // BC7 QUICK reduces the encoder search for speed. It changes the quality
            // tradeoff, so do not treat this as equal quality to another library's default.
            flags |= DirectX::TEX_COMPRESS_BC7_QUICK;
        }

        // Time the whole API call, including GPU transfers when using a device.
        const auto compress_start = GetTime();
        HRESULT compress_hr;
        if (device) // gpu 
        {
            compress_hr = DirectX::Compress(
                device, first, lower_meta.mipLevels, lower_meta,
                meta_data.format, flags,
                DirectX::TEX_ALPHA_WEIGHT_DEFAULT, compressed);
        }
        else // cpu
        {
            compress_hr = DirectX::Compress(
                first, lower_meta.mipLevels, lower_meta,
                meta_data.format,
                flags, DirectX::TEX_THRESHOLD_DEFAULT,
                compressed);
        }
        compress_ms = (GetTime() - compress_start) / 1000.0;
        if (FAILED(compress_hr))
        {
            std::cerr << "Compress failed: 0x" << std::hex << static_cast<unsigned long>(compress_hr) << '\n';
            return 1;
        }
    }

    // Keep the original compressed Mip 0 and append the newly compressed levels.
    // These Image objects borrow pixels owned by source and compressed above.
    // Image is a view: copying it copies pointers and pitches, not the pixel storage.
    // source and compressed must outlive this list and any future SaveToDDSFile call.
    std::vector<DirectX::Image> images{source.GetImages()[0]};
    if (compressed.GetImageCount() > 0)
    {
        images.insert(images.end(), compressed.GetImages(), compressed.GetImages() + compressed.GetImageCount());
    }

    // Prepare the output metadata; DDS saving is currently disabled below.
    auto output_meta = meta_data;
    output_meta.mipLevels = images.size();

    // Saving would be outside all timers. Keep the existing disabled-save behavior
    // while reviewing the benchmark; output_path currently produces no DDS file.
    // const HRESULT save_hr = DirectX::SaveToDDSFile(
    //     images.data(), images.size(), output_meta,
    //     DirectX::DDS_FLAGS_FORCE_DX10_EXT, output_path.c_str());

    // if (FAILED(save_hr))
    // {
    //     std::cerr << "Save failed: 0x" << std::hex << static_cast<unsigned long>(save_hr) << '\n';
    //     return 1;
    // }

    // GetTime returns microseconds; each timed difference above is converted to ms.
    // Sum only the three processing calls, excluding file I/O and setup.
    std::cout << "Compression backend: " << (device ? "GPU" : "CPU OpenMP") << '\n'
              << "Decompress: " << decompress_ms << " ms\n"
              << "Downsampling: " << downsample_ms << " ms\n"
              << "Compression: " << compress_ms << " ms\n"
              << "Total generation time : " << decompress_ms + downsample_ms + compress_ms << " ms" << std::endl;
    return 0;
}

// Batch handling is outside the processing timers. Only this directory is scanned.
// Sort before processing to make the input order repeatable between runs.
// Files are processed one at a time; codec-internal threads may still run in parallel.
// Continue after a file fails, then return failure if any file was skipped or failed.
int ProcessBatch(const std::filesystem::path &folder, const std::filesystem::path &output,
                 ID3D11Device* device)
{
    // Collect DDS files in this folder only; report filesystem errors without exceptions.
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
    // Use the same file order on each run.
    std::sort(files.begin(), files.end());

    std::filesystem::create_directories(output, error);
    if (error)
    {
        std::cerr << "Cannot create output folder " << output << ": " << error.message() << '\n';
        return 1;
    }

    // Process files sequentially; a failed file does not stop the remaining files.
    size_t succeeded = 0;
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
        if (ProcessFile(file, destination, device) == 0)
            ++succeeded;
    }

    std::cout << std::dec << "Succeeded: " << succeeded
              << ", failed: " << files.size() - succeeded << '\n';
    // Skipped files also count as failures in the batch exit status.
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

    // WIC-based mip generation needs COM; this flag does not parallelize the benchmark.
    const HRESULT com_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(com_hr))
    {
        std::cerr << "COM initialization failed: 0x" << std::hex
                  << static_cast<unsigned long>(com_hr) << '\n';
        return 1;
    }

    // Balance COM initialization on every return path.
    // A stack destructor balances CoInitializeEx even when a later error returns early.
    // COINIT_MULTITHREADED is a COM apartment setting, not a worker-thread count.
    struct ComCleanup
    {
        ~ComCleanup() { CoUninitialize(); }
    } com_cleanup;


    // Create one hardware device for the whole run, outside the measured calls.
    // ComPtr releases the device automatically when main returns.
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

    return ProcessFile(input, output, device.Get());
}
