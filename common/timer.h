#pragma once 

#include <cstdint>
#include <chrono>

uint64_t GetTime()
{
    return std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() ).time_since_epoch().count();
}