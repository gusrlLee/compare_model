cmake_minimum_required(VERSION 3.30)

project(compare_models LANGUAGES C CXX)


# DirectXTex 
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(directxtex CONFIG REQUIRED)

add_executable(dtex dtex.cpp)
target_compile_features(dtex PRIVATE cxx_std_17)
target_link_libraries(dtex PRIVATE Microsoft::DirectXTex OpenMP::OpenMP_CXX d3d11)

# Use the installed x64 SDK and copy its runtime next to the executable.
set(COMPRESSONATOR_ROOT "C:/Compressonator_4.6.0" CACHE PATH "Compressonator SDK directory")
find_path(COMPRESSONATOR_INCLUDE_DIR compressonator.h HINTS "${COMPRESSONATOR_ROOT}/include" REQUIRED)
find_library(COMPRESSONATOR_LIBRARY Compressonator_MD_DLL HINTS "${COMPRESSONATOR_ROOT}/lib/bin/x64" REQUIRED)
find_file(COMPRESSONATOR_DLL Compressonator_MD_DLL.dll HINTS "${COMPRESSONATOR_ROOT}/lib/bin/x64" REQUIRED)
add_library(CompressonatorSDK SHARED IMPORTED)
set_target_properties(CompressonatorSDK PROPERTIES
    IMPORTED_IMPLIB "${COMPRESSONATOR_LIBRARY}"
    IMPORTED_LOCATION "${COMPRESSONATOR_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${COMPRESSONATOR_INCLUDE_DIR}")

add_executable(compressnator compressnator.cpp)
target_compile_features(compressnator PRIVATE cxx_std_17)
target_link_libraries(compressnator PRIVATE CompressonatorSDK)
add_custom_command(TARGET compressnator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:CompressonatorSDK>" "$<TARGET_FILE_DIR:compressnator>")

# GPU files come from the official 4.6.0 CLI zip; the CPU build needs only the SDK.
set(COMPRESSONATOR_GPU_RUNTIME "${CMAKE_SOURCE_DIR}/build/compressonator-runtime/compressonatorcli-4.6.0-win64"
    CACHE PATH "Extracted Compressonator CLI runtime directory")
if(EXISTS "${COMPRESSONATOR_GPU_RUNTIME}/plugins/EncodeWith_DXC.dll")
    add_custom_command(TARGET compressnator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:compressnator>/plugins"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${COMPRESSONATOR_GPU_RUNTIME}/plugins/EncodeWith_DXC.dll" "$<TARGET_FILE_DIR:compressnator>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${COMPRESSONATOR_GPU_RUNTIME}/plugins/compute" "$<TARGET_FILE_DIR:compressnator>/plugins/compute"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${COMPRESSONATOR_GPU_RUNTIME}/license" "$<TARGET_FILE_DIR:compressnator>/license")
else()
    message(STATUS "Compressonator GPU runtime not found; set COMPRESSONATOR_GPU_RUNTIME to the extracted official CLI package for --gpu")
endif()

# Build only etcpak's codec and mip code, without its CLI or Tracy dependency.
find_package(PNG REQUIRED)
set(ETCPAK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/etcpak")
add_library(etcpak_codec STATIC
    "${ETCPAK_DIR}/ProcessDxtc.cpp"
    "${ETCPAK_DIR}/Dither.cpp"
    "${ETCPAK_DIR}/Tables.cpp"
    "${ETCPAK_DIR}/bc7enc.cpp"
    "${ETCPAK_DIR}/bcdec.c"
    "${ETCPAK_DIR}/Decode.cpp"
    "${ETCPAK_DIR}/Bitmap.cpp"
    "${ETCPAK_DIR}/BitmapDownsampled.cpp"
    "${ETCPAK_DIR}/TaskDispatch.cpp"
    "${ETCPAK_DIR}/System.cpp"
    "${ETCPAK_DIR}/Debug.cpp")
target_include_directories(etcpak_codec PUBLIC "${ETCPAK_DIR}")
target_compile_features(etcpak_codec PUBLIC cxx_std_20)
target_compile_definitions(etcpak_codec PRIVATE NOMINMAX)
target_link_libraries(etcpak_codec PRIVATE PNG::PNG)
if(MSVC)
    target_compile_options(etcpak_codec PRIVATE /arch:AVX2)
else()
    target_compile_options(etcpak_codec PRIVATE -mavx2 -msse4.1)
endif()

add_executable(etcpak etcpak.cpp)
target_link_libraries(etcpak PRIVATE etcpak_codec)

# Installed NVTT SDK; no nvpro GUI framework is needed for this console benchmark.
set(NVTT_ROOT "C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools" CACHE PATH "NVTT SDK directory")
find_path(NVTT_INCLUDE_DIR nvtt/nvtt.h HINTS "${NVTT_ROOT}/include" REQUIRED)
find_library(NVTT_LIBRARY nvtt30205 HINTS "${NVTT_ROOT}/lib/x64-v142" REQUIRED)
find_file(NVTT_DLL nvtt30205.dll HINTS "${NVTT_ROOT}" REQUIRED)
add_library(NVTT::NVTT SHARED IMPORTED)
set_target_properties(NVTT::NVTT PROPERTIES
    IMPORTED_IMPLIB "${NVTT_LIBRARY}"
    IMPORTED_LOCATION "${NVTT_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${NVTT_INCLUDE_DIR}")
add_executable(nvtt nvtt.cpp)
target_compile_features(nvtt PRIVATE cxx_std_17)
target_link_libraries(nvtt PRIVATE NVTT::NVTT)
add_custom_command(TARGET nvtt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:NVTT::NVTT>" "$<TARGET_FILE_DIR:nvtt>")
