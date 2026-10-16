#include "split_radix_fft_cpu.cpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <float.h>

#define MAXN 16384
#define RNG_MAX 64.0

/**
 * Power-of-2 sweep round-trip test for the CPU FFTs.
 *
 * For each n in {1, 2, 4, ..., 16384}, fills a random real signal and runs a
 * forward + inverse round-trip through both the exp and the sin/cos variants,
 * reporting the max reconstruction error and a PASS/FAIL against a 1e-8
 * threshold.
 *
 * @return 0 if every round-trip passes, 1 otherwise.
 */
int main(void) {
    printf("=== N sweep: split-radix + sincos round-trip ===\n\n");

    /* --- split_radix_exp_fft_inout + split_radix_exp_ifft_inout --- */
    printf("[split_radix_exp_fft_inout + split_radix_exp_ifft_inout]\n");
    int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n_sizes = 15;
    int failures = 0;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        std::complex<double> *orig = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
        std::complex<double> *copy = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
        srand(12345);
        for (int i = 0; i < n; i++) {
            orig[i] = std::complex<double>(
                    ((double)rand() / RAND_MAX - 0.5) * 2.0 * RNG_MAX,
                    0.0
                    );
            copy[i] = orig[i];
        }

        clock_t t0 = clock();
        split_radix_exp_fft_inout(copy, n);
        split_radix_exp_ifft_inout(copy, n);
        clock_t t1 = clock();

        double err = 0;
        for (int i = 0; i < n; i++) {
            double e = hypot(copy[i].real() - orig[i].real(), copy[i].imag() - orig[i].imag());
            err = e > err ? e : err;
        }

        double dt = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
        printf("  N=%6d: error=%.2e  time=%.3f ms  %s\n", n, err, dt, err < 1e-8 ? "PASS" : "FAIL");
        if (err >= 1e-8) failures++;

        free(orig);
        free(copy);
    }

    /* --- split_radix_sincos_fft_inout + split_radix_sincos_ifft_inout --- */
    printf("\n[sincos round-trip]\n");
    failures = 0;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        std::complex<double> *orig = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
        std::complex<double> *copy = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
        srand(12345);
        for (int i = 0; i < n; i++) {
            orig[i] = std::complex<double>(
                    ((double)rand() / RAND_MAX - 0.5) * 2.0 * RNG_MAX,
                    0.0
                    );
            copy[i] = orig[i];
        }

        clock_t t0 = clock();
        split_radix_sincos_fft_inout(copy, n);
        split_radix_sincos_ifft_inout(copy, n);
        clock_t t1 = clock();

        double err = 0;
        for (int i = 0; i < n; i++) {
            double e = hypot(copy[i].real() - orig[i].real(), copy[i].imag() - orig[i].imag());
            err = e > err ? e : err;
        }

        double dt = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
        printf("  N=%6d: error=%.2e  time=%.3f ms  %s\n", n, err, dt, err < 1e-8 ? "PASS" : "FAIL");
        if (err >= 1e-8) failures++;

        free(orig);
        free(copy);
    }

    return failures > 0 ? 1 : 0;
}
