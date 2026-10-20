#include <stdio.h>
#include <memory>
#include <string.h>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <assert.h>

#include "bc7enc.h"
#include "Bitmap.hpp"
#include "BlockData.hpp"
#include "DataProvider.hpp"
#include "System.hpp"
#include "TaskDispatch.hpp"
#include "common/timer.h"
#include "TextureHeader.hpp"
#include <utility>

namespace fs = std::filesystem;

void Usage()
{
    fprintf( stderr, "Usage: etcpak [-c bc1|bc3|bc4|bc5|bc7] [--linear] <input.dds|\"folder/*.dds\"> <output-directory>\n" );
    fprintf( stderr, "Default codec: bc7. --linear disables sRGB conversion for mip generation.\n" );
    fprintf( stderr, "The selected codec must match the input codec to preserve Mip 0.\n" );
}

// Read the whole file before we start timing.
static std::vector<uint8_t> LoadFileToMemory( const fs::path& filename )
{
    std::ifstream file( filename, std::ios::binary | std::ios::ate );
    if( !file )
    {
        fprintf( stderr, "Failed to open input file: %s\n", filename.string().c_str() );
        return {};
    }

    const auto fileSize = file.tellg();
    if( fileSize <= 0 )
    {
        fprintf( stderr, "Failed to determine size or input file is empty: %s\n", filename.string().c_str() );
        return {};
    }

    std::vector<uint8_t> data( static_cast<size_t>( fileSize ) );

    file.seekg( 0, std::ios::beg );
    if(!file.read( reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
    {
        fprintf( stderr, "Failed to read input file: %s\n", filename.string().c_str() );
        return {};
    }

    return data;
}

// Store both times in microseconds.
struct DdsBenchmarkResult
{
    uint64_t decompress = 0;
    uint64_t mipmapCompression = 0;
};


static bool ProcessDdsFile(const fs::path& input, const fs::path& output, DdsBenchmarkResult& result,
    unsigned int workers, CodecType codec, bool linearize )
{
    // Step 1: Load the file. Do not time file reading.
    auto ddsData = LoadFileToMemory( input );
    if( ddsData.empty() ) return false;

    try
    {
        // source uses these bytes, so keep ddsData alive.
        BlockData source( ddsData.data(), ddsData.size() );
        if( source.Codec() != codec )
        {
            fprintf( stderr, "Output codec must match input to preserve compressed Mip 0.\n" );
            return false;
        }

        // Step 2: Time decoding and the wait for all decode jobs.
        const auto decompressStart = GetTime();
        auto decoded = source.DecodeMultithreaded( workers );
        const auto decompressEnd = GetTime();

        if( !decoded )
        {
            fprintf( stderr, "Failed to decode DDS: %s\n", input.string().c_str() );
            return false;
        }


        result.decompress = decompressEnd - decompressStart;
        const auto size = decoded->Size();

        if( ( size.x & ( size.x - 1 ) ) != 0 || ( size.y & ( size.y - 1 ) ) != 0 )
        {
            throw std::invalid_argument( "This mipmap benchmark currently requires power-of-two dimensions" );
        }

        const bool rgba = codec == CodecType::Bc3 || codec == CodecType::Bc7;
        // Make room for all mip levels in the chosen output codec.
        auto compressed = std::make_shared<BlockData>( size, true, codec );
        // Preserve the exact original blocks instead of encoding Mip 0 again.
        // This copy is outside the processing timer.
        const size_t baseBlocks = size_t( size.x / 4 ) * ( size.y / 4 );
        const size_t blockBytes = codec == CodecType::Bc1 || codec == CodecType::Bc4 ? 8 : 16;
        memcpy( compressed->Data(), source.Data(), baseBlocks * blockBytes );

        bc7enc_compress_block_params bc7params{};
        if( codec == CodecType::Bc7 )
        {
            bc7enc_compress_block_params_init( &bc7params );
        }

        // Give the image to the old pipeline. dp now owns its memory.
        DataProvider dp( std::move( decoded ), true, linearize );
        const auto num = dp.NumberOfParts();
        size_t offset = 0;

        // Step 3: Time downsampling and encoding together.
        const auto processingStart = GetTime();
        try
        {
            for( unsigned int i = 0; i < num; i++ )
            {
                // Get ready pixels. The old code also starts the next mip when needed.
                const auto part = dp.NextPart();
                const auto blocks = part.width / 4 * part.lines;
                assert( blocks > 0 && part.offset == offset );
                // Consume base-level parts so NextPart can start the first mip filter.
                // Only lower levels enter the encoder; their offsets still include Mip 0.
                if( offset >= baseBlocks )
                {
                    TaskDispatch::Queue( [compressed, part, blocks, offset, rgba, &bc7params] {
                        if( rgba )
                            compressed->ProcessRGBA( part.src, blocks, offset, part.width, true, &bc7params );
                        else
                            compressed->Process( part.src, blocks, offset, part.width, false, true );
                    } );
                }
                // Small mips still use a full block. Count these blocks here.
                offset += blocks;
            }
        }
        catch( ... )
        {
            TaskDispatch::Sync();
            // On error, wait for mip jobs before the image memory is freed.
            dp.WaitForBenchmark();
            throw;
        }
        TaskDispatch::Sync();
        // Encoding is done. Also wait for all downsampling jobs.
        dp.WaitForBenchmark();
        result.mipmapCompression = GetTime() - processingStart;

        // Step 4: Save the result after timing has stopped.
        if( !compressed->WriteForBenchmark( output.string().c_str(), BlockData::Dds, source ) ) return false;
        printf( "%s (%d x %d): decompress %.3f ms, mipmap + compression %.3f ms\n",
            input.string().c_str(), size.x, size.y,
            result.decompress / 1000.0, result.mipmapCompression / 1000.0 );
        printf( "Saved: %s\n", output.string().c_str() );
    }
    catch( const std::invalid_argument& error )
    {
        fprintf( stderr, "Invalid DDS: %s (%s)\n", input.string().c_str(), error.what() );
        return false;
    }

    return true;
}

static bool RunDdsMipmapBenchmark(
    const fs::path& input, const fs::path& outputDir, CodecType codec, bool linearize )
{
    if( !fs::is_directory( outputDir ) )
    {
        fprintf( stderr, "Output directory does not exist: %s\n", outputDir.string().c_str() );
        return false;
    }

    std::vector<fs::path> files;

    // Put one file or all matching files into the same list.
    if( input.filename() == "*.dds" )
    {
        const auto directory = input.has_parent_path() ? input.parent_path() : fs::path( "." );

        for( const auto& entry : fs::directory_iterator( directory ) )
        {
            if( entry.is_regular_file() && entry.path().extension() == ".dds" )
            {
                files.emplace_back( entry.path() );
            }
        }
    }
    else if( fs::is_regular_file( input ) && input.extension() == ".dds" )
    {
        files.emplace_back( input );
    }

    if( files.empty() )
    {
        fprintf( stderr, "No DDS files found: %s\n", input.string().c_str() );
        return false;
    }

    std::sort( files.begin(), files.end() );
    const auto cpus = System::CPUCores();
    TaskDispatch taskDispatch( cpus );

    // Use the same thread pool for the whole batch.
    DdsBenchmarkResult total;

    if( codec == CodecType::Bc7 )
    {
        bc7enc_compress_block_init();
    }

    for( const auto& file : files )
    {
        DdsBenchmarkResult result;

        auto filename = file.filename();
        filename.replace_extension( ".dds" );
        const auto output = outputDir / filename;
        // Do not replace the input file with its own result.
        if( fs::exists( output ) && fs::equivalent( file, output ) )
        {
            fprintf( stderr, "Output must differ from input: %s\n", file.string().c_str() );
            return false;
        }
        if( !ProcessDdsFile( file, output, result, cpus, codec, linearize ) )
        {
            return false;
        }

        // Add this file's times to the batch totals.
        total.decompress += result.decompress;
        total.mipmapCompression += result.mipmapCompression;
    }

    printf( "Total decompress time: %.3f ms\n", total.decompress / 1000.0 );
    printf( "Total mipmap + compression time: %.3f ms\n", total.mipmapCompression / 1000.0 );

    return true;
}

// Only the DDS mipmap experiment remains; no separate mode flag is needed.
int main( int argc, char** argv )
{
    auto codec = CodecType::Bc7;
    bool linearize = true;
    int argument = 1;
    while( argument < argc )
    {
        if( strcmp( argv[argument], "--linear" ) == 0 )
        {
            linearize = false;
            ++argument;
        }
        else if( strcmp( argv[argument], "-c" ) == 0 && argument + 1 < argc )
        {
            const char* name = argv[argument + 1];
            if( strcmp( name, "bc1" ) == 0 ) codec = CodecType::Bc1;
            else if( strcmp( name, "bc3" ) == 0 ) codec = CodecType::Bc3;
            else if( strcmp( name, "bc4" ) == 0 ) codec = CodecType::Bc4;
            else if( strcmp( name, "bc5" ) == 0 ) codec = CodecType::Bc5;
            else if( strcmp( name, "bc7" ) == 0 ) codec = CodecType::Bc7;
            else
            {
                fprintf( stderr, "Unknown BC codec: %s\n", name );
                return 1;
            }
            argument += 2;
        }
        else
            break;
    }
    if( argc - argument != 2 || argv[argument][0] == '-' )
    {
        Usage();
        return 1;
    }
    try
    {
        return RunDdsMipmapBenchmark( argv[argument], argv[argument + 1], codec, linearize ) ? 0 : 1;
    }
    catch( const fs::filesystem_error& error )
    {
        fprintf( stderr, "DDS file enumeration failed: %s\n", error.what() );
        return 1;
    }
}
