#include <DirectXTex.h>
#include <iostream>
#include <filesystem>
#include <vector>

#include "common/timer.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: dtex <input.dds> <output.dds>\n";
        return 1;
    }

    DirectX::TexMetadata meta_data{};
    DirectX::ScratchImage source;

    // file loading
    const std::filesystem::path input_path(argv[1]);
    const HRESULT hr = DirectX::LoadFromDDSFile(
        input_path.c_str(),
        DirectX::DDS_FLAGS_NONE,
        &meta_data,
        source);

    if (FAILED(hr))
    {
        std::cerr << "DDS load failed: 0x" << std::hex << static_cast<unsigned long>(hr) << '\n';
        return 1;
    }

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

    if (meta_data.dimension != DirectX::TEX_DIMENSION_TEXTURE2D || meta_data.arraySize != 1)
    {
        std::cerr << "Input must be a single 2D texture.\n";
        return 1;
    }

    // Decompressed
    DirectX::ScratchImage decompressed;
    const auto decompress_start = GetTime();
    const HRESULT decompress_hr = DirectX::Decompress(source.GetImages(), source.GetImageCount(), source.GetMetadata(), DXGI_FORMAT_UNKNOWN, decompressed);
    const double decompress_ms = (GetTime() - decompress_start) / 1000.0;
    
    if (FAILED(decompress_hr))
    {
        std::cerr << "Decompress failed: 0x" << std::hex << static_cast<unsigned long>(decompress_hr) << std::endl;
        return 1;
    }

    // Downsampling
    DirectX::ScratchImage mip_chain;
    const HRESULT mip_hr = DirectX::GenerateMipMaps(
        decompressed.GetImages(), decompressed.GetImageCount(),
        decompressed.GetMetadata(),
        DirectX::TEX_FILTER_BOX, 0, mip_chain);

    if (FAILED(mip_hr))
    {
        std::cerr << "GenerateMipMaps failed: 0x" << std::hex << static_cast<unsigned long>(mip_hr) << '\n';
        return 1;
    }

    std::cout << "Mip levels: " << mip_chain.GetMetadata().mipLevels << '\n';

    DirectX::ScratchImage compressed;
    auto lower_meta = mip_chain.GetMetadata();

    // compression 
    if (lower_meta.mipLevels > 1)
    {
        const DirectX::Image *first = mip_chain.GetImages() + 1;
        lower_meta.width = first->width;
        lower_meta.height = first->height;
        --lower_meta.mipLevels;

        const HRESULT compress_hr = DirectX::Compress(
            first, lower_meta.mipLevels, lower_meta,
            meta_data.format,
            DirectX::TEX_COMPRESS_PARALLEL, DirectX::TEX_THRESHOLD_DEFAULT,
            compressed);

        if (FAILED(compress_hr))
        {
            std::cerr << "Compress failed: 0x" << std::hex << static_cast<unsigned long>(compress_hr) << '\n';
            return 1;
        }
    }

    std::vector<DirectX::Image> images{source.GetImages()[0]};
    if (compressed.GetImageCount() > 0)
    {
        images.insert(images.end(), compressed.GetImages(), compressed.GetImages() + compressed.GetImageCount());
    }

    // dds file save
    auto output_meta = meta_data;
    output_meta.mipLevels = images.size();

    const std::filesystem::path output_path(argv[2]);
    const HRESULT save_hr = DirectX::SaveToDDSFile(
        images.data(), images.size(), output_meta,
        DirectX::DDS_FLAGS_FORCE_DX10_EXT, output_path.c_str());

    if (FAILED(save_hr))
    {
        std::cerr << "Save failed: 0x" << std::hex << static_cast<unsigned long>(save_hr) << '\n';
        return 1;
    }

    std::cout << "Saved " << images.size() << " mip levels.\n";

    return 0;
}
