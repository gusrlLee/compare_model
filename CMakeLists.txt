cmake_minimum_required(VERSION 3.30)

project(compare_models LANGUAGES CXX)

find_package(directxtex CONFIG REQUIRED)

add_executable(dtex dtex.cpp)
target_compile_features(dtex PRIVATE cxx_std_17)
target_link_libraries(dtex PRIVATE Microsoft::DirectXTex)

