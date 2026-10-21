#pragma once
#include <nvtt/nvtt.h>

// Decode BC blocks with CUDA texture hardware directly into an NVTT GPU Surface.
bool DecompressGpu(nvtt::Format format, int width, int height,
                   const void* blocks, nvtt::Surface& image);
