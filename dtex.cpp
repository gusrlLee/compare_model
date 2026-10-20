#include <DirectXTex.h>
#include <iostream>

int main()
{
    const bool compressed = DirectX::IsCompressed(DXGI_FORMAT_BC1_UNORM);
    std::cout << "BC1 is compressed: " << std::boolalpha << compressed << '\n';
    return compressed ? 0 : 1;
}
