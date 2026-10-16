#include <cstdio>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <float.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#define MAXN 16384
#define RNG_MAX 64.0

// Include CPU algorithm from fft.c
#include "fft.c"
// Include GPU test function from fft_gpu_test.cu
#include "fft_gpu_test.cu"

int main(void) {
    printf("=== Unified entry-point test (N=%d) ===\n\n", MAXN);

    std::complex<double> *a = (std::complex<double>*)malloc(MAXN * sizeof(std::complex<double>));
    std::complex<double> *a_orig = (std::complex<double>*)malloc(MAXN * sizeof(std::complex<double>));
    srand(42);
    for (int i = 0; i < MAXN; i++) {
        a[i] = std::complex<double>(
            ((double)rand() / RAND_MAX - 0.5) * 2.0 * RNG_MAX,
            0.0
        );
        a_orig[i] = a[i];
    }

    /* --- CPU FFT+IFFT round-trip --- */
    printf("[CPU] FFT + IFFT (round-trip):\n");
    split_radix_fft_inplace(a, MAXN);
    split_radix_ifft_inplace(a, MAXN);
    double max_err_cpu = 0;
    for (int i = 0; i < MAXN; i++) {
        double e = hypot(a[i].real() - a_orig[i].real(), a[i].imag() - a_orig[i].imag());
        max_err_cpu = e > max_err_cpu ? e : max_err_cpu;
    }
    printf("  Max error: %.2e  ", max_err_cpu);
    printf("%s\n\n", max_err_cpu < 1e-10 ? "PASS" : "FAIL");

    /* --- GPU hipFFT + IFFT round-trip --- */
    printf("[GPU] hipFFT + IFFT (round-trip):\n");
    gpu_fft_test(MAXN);

    /* --- Cross-check CPU at N=256 --- */
    int N_small = 256;
    std::complex<double> *s = (std::complex<double>*)malloc(N_small * sizeof(std::complex<double>));
    std::complex<double> *s_orig = (std::complex<double>*)malloc(N_small * sizeof(std::complex<double>));
    srand(42);
    for (int i = 0; i < N_small; i++) {
        s[i] = std::complex<double>(
            ((double)rand() / RAND_MAX - 0.5) * 2.0 * RNG_MAX,
            0.0
        );
        s_orig[i] = s[i];
    }
    split_radix_fft_inplace(s, N_small);
    split_radix_ifft_inplace(s, N_small);
    double max_err_256 = 0;
    for (int i = 0; i < N_small; i++) {
        double e = hypot(s[i].real() - s_orig[i].real(), s[i].imag() - s_orig[i].imag());
        max_err_256 = e > max_err_256 ? e : max_err_256;
    }
    printf("[CPU] N=256 round-trip max error: %.2e  %s\n", max_err_256, max_err_256 < 1e-12 ? "PASS" : "FAIL");

    free(a); free(a_orig); free(s); free(s_orig);
    return 0;
}
