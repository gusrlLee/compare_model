# Run from the project directory so executable paths resolve correctly.
Set-Location -LiteralPath $PSScriptRoot

# Build current sources before measuring; stop rather than run stale executables.
cmake --build build --config Release --target dtex nvtt etcpak compressnator
if ($LASTEXITCODE -ne 0) { throw "Release build failed." }

# Each executable warms up once internally; warm-up is excluded from averages.
# All commands are listed explicitly; each combination has its own output folder.
# Input folders: bc1, bc2, bc3, bc4, bc5, bc6h, bc7.
# CPU/GPU here selects each driver's available backend, not identical stage implementations.
# DirectXTex BC7 QUICK is fixed in the current executable; no other quality CLI is exposed.
# NVTT GPU compression may fall back to the CPU for some formats/qualities.
# Compressonator quality is fixed to 0.05; DDS saving is currently disabled in its C++ driver.
# Existing outputs are not cleared. Use fresh output folders for another complete run.

# DirectXTex - CPU (7 commands)
./build/Release/dtex.exe "D:/space/input/bc1/*.dds" "D:/space/output/directxtex/cpu/bc1"
./build/Release/dtex.exe "D:/space/input/bc2/*.dds" "D:/space/output/directxtex/cpu/bc2"
./build/Release/dtex.exe "D:/space/input/bc3/*.dds" "D:/space/output/directxtex/cpu/bc3"
./build/Release/dtex.exe "D:/space/input/bc4/*.dds" "D:/space/output/directxtex/cpu/bc4"
./build/Release/dtex.exe "D:/space/input/bc5/*.dds" "D:/space/output/directxtex/cpu/bc5"
./build/Release/dtex.exe "D:/space/input/bc6h/*.dds" "D:/space/output/directxtex/cpu/bc6h"
./build/Release/dtex.exe "D:/space/input/bc7/*.dds" "D:/space/output/directxtex/cpu/bc7"

# DirectXTex - GPU: BC6H and BC7 only (2 commands)
./build/Release/dtex.exe "D:/space/input/bc6h/*.dds" "D:/space/output/directxtex/gpu/bc6h" --gpu
./build/Release/dtex.exe "D:/space/input/bc7/*.dds" "D:/space/output/directxtex/gpu/bc7" --gpu

# NVTT - CPU/GPU, fastest quality only (14 commands)

# NVTT - CPU, fastest
./build/Release/nvtt.exe "D:/space/input/bc1/*.dds" "D:/space/output/nvtt/cpu/fastest/bc1" fastest
./build/Release/nvtt.exe "D:/space/input/bc2/*.dds" "D:/space/output/nvtt/cpu/fastest/bc2" fastest
./build/Release/nvtt.exe "D:/space/input/bc3/*.dds" "D:/space/output/nvtt/cpu/fastest/bc3" fastest
./build/Release/nvtt.exe "D:/space/input/bc4/*.dds" "D:/space/output/nvtt/cpu/fastest/bc4" fastest
./build/Release/nvtt.exe "D:/space/input/bc5/*.dds" "D:/space/output/nvtt/cpu/fastest/bc5" fastest
./build/Release/nvtt.exe "D:/space/input/bc6h/*.dds" "D:/space/output/nvtt/cpu/fastest/bc6h" fastest
./build/Release/nvtt.exe "D:/space/input/bc7/*.dds" "D:/space/output/nvtt/cpu/fastest/bc7" fastest

# NVTT - GPU, fastest
./build/Release/nvtt.exe "D:/space/input/bc1/*.dds" "D:/space/output/nvtt/gpu/fastest/bc1" fastest --gpu
./build/Release/nvtt.exe "D:/space/input/bc2/*.dds" "D:/space/output/nvtt/gpu/fastest/bc2" fastest --gpu
./build/Release/nvtt.exe "D:/space/input/bc3/*.dds" "D:/space/output/nvtt/gpu/fastest/bc3" fastest --gpu
./build/Release/nvtt.exe "D:/space/input/bc4/*.dds" "D:/space/output/nvtt/gpu/fastest/bc4" fastest --gpu
./build/Release/nvtt.exe "D:/space/input/bc5/*.dds" "D:/space/output/nvtt/gpu/fastest/bc5" fastest --gpu
./build/Release/nvtt.exe "D:/space/input/bc6h/*.dds" "D:/space/output/nvtt/gpu/fastest/bc6h" fastest --gpu
./build/Release/nvtt.exe "D:/space/input/bc7/*.dds" "D:/space/output/nvtt/gpu/fastest/bc7" fastest --gpu

# etcpak - CPU only; sRGB color codecs and linear BC4/BC5 (5 commands).
# Use raw filtering for BC4/BC5 channel data; they also bypass sRGB conversion internally.
# Unlike the other drivers, etcpak requires the output directory to exist.

# etcpak - srgb-filter
New-Item -ItemType Directory -Force -Path "D:/space/output/etcpak/cpu/srgb-filter/bc1" | Out-Null
New-Item -ItemType Directory -Force -Path "D:/space/output/etcpak/cpu/srgb-filter/bc3" | Out-Null
New-Item -ItemType Directory -Force -Path "D:/space/output/etcpak/cpu/srgb-filter/bc7" | Out-Null
./build/Release/etcpak.exe -c bc1 "D:/space/input/bc1/*.dds" "D:/space/output/etcpak/cpu/srgb-filter/bc1"
./build/Release/etcpak.exe -c bc3 "D:/space/input/bc3/*.dds" "D:/space/output/etcpak/cpu/srgb-filter/bc3"
./build/Release/etcpak.exe -c bc7 "D:/space/input/bc7/*.dds" "D:/space/output/etcpak/cpu/srgb-filter/bc7"

# etcpak - linear channel data
New-Item -ItemType Directory -Force -Path "D:/space/output/etcpak/cpu/raw-filter/bc4" | Out-Null
New-Item -ItemType Directory -Force -Path "D:/space/output/etcpak/cpu/raw-filter/bc5" | Out-Null
./build/Release/etcpak.exe -c bc4 --linear "D:/space/input/bc4/*.dds" "D:/space/output/etcpak/cpu/raw-filter/bc4"
./build/Release/etcpak.exe -c bc5 --linear "D:/space/input/bc5/*.dds" "D:/space/output/etcpak/cpu/raw-filter/bc5"

# Compressonator - quality 0.05, CPU/GPU (14 commands)

# Compressonator - CPU, quality 0.05
./build/Release/compressnator.exe "D:/space/input/bc1/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc1" 0.05
./build/Release/compressnator.exe "D:/space/input/bc2/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc2" 0.05
./build/Release/compressnator.exe "D:/space/input/bc3/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc3" 0.05
./build/Release/compressnator.exe "D:/space/input/bc4/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc4" 0.05
./build/Release/compressnator.exe "D:/space/input/bc5/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc5" 0.05
./build/Release/compressnator.exe "D:/space/input/bc6h/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc6h" 0.05
./build/Release/compressnator.exe "D:/space/input/bc7/*.dds" "D:/space/output/compressonator/cpu/q0.05/bc7" 0.05

# Compressonator - GPU, quality 0.05
./build/Release/compressnator.exe "D:/space/input/bc1/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc1" 0.05 --gpu
./build/Release/compressnator.exe "D:/space/input/bc2/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc2" 0.05 --gpu
./build/Release/compressnator.exe "D:/space/input/bc3/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc3" 0.05 --gpu
./build/Release/compressnator.exe "D:/space/input/bc4/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc4" 0.05 --gpu
./build/Release/compressnator.exe "D:/space/input/bc5/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc5" 0.05 --gpu
./build/Release/compressnator.exe "D:/space/input/bc6h/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc6h" 0.05 --gpu
./build/Release/compressnator.exe "D:/space/input/bc7/*.dds" "D:/space/output/compressonator/gpu/q0.05/bc7" 0.05 --gpu
