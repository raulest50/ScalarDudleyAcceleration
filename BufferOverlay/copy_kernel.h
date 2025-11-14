#pragma once
#include <complex>
#ifndef N
#define N 8192
#endif
using complex_t = std::complex<float>;
extern "C" void copy_kernel(const complex_t* in, complex_t* out);
