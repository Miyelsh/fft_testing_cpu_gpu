#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <float.h>

#define RNG_MAX 64.0
#define MAXN 16384

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define HIPFFT_CHECK(call) do { \
    hipfftResult err = call; \
    if (err != HIPFFT_SUCCESS) { \
        fprintf(stderr, "hipFFT error %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/**
 * hipFFT forward + inverse round-trip test helper.
 *
 * Generates a random real signal on the host, copies it to the device, runs a
 * forward then inverse hipFFT, copies the result back, divides by size (hipFFT
 * does not scale the backward transform), and prints the max reconstruction
 * error and the transform time.
 *
 * @param[in] size  Transform length (number of samples).
 */
void gpu_fft_test(int size) {
    hipfftComplex *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, size * sizeof(hipfftComplex)));
    HIP_CHECK(hipMalloc(&d_out, size * sizeof(hipfftComplex)));

    srand(12345);
    hipfftComplex *h_in = (hipfftComplex*)malloc(size * sizeof(hipfftComplex));
    for (int i = 0; i < size; i++) {
        h_in[i].x = ((float)rand()/((float)RAND_MAX) - 0.5f) * 2.0f * RNG_MAX;
        h_in[i].y = 0.0f;
    }

    hipfftComplex *h_orig = (hipfftComplex*)malloc(size * sizeof(hipfftComplex));
    memcpy(h_orig, h_in, size * sizeof(hipfftComplex));
    HIP_CHECK(hipMemcpy(d_in, h_in, size * sizeof(hipfftComplex), hipMemcpyHostToDevice));

    hipfftHandle plan;
    HIPFFT_CHECK(hipfftPlan1d(&plan, size, HIPFFT_C2C, 1));

    clock_t t0 = clock();
    HIPFFT_CHECK(hipfftExecC2C(plan, d_in, d_out, HIPFFT_FORWARD));
    HIP_CHECK(hipDeviceSynchronize());
    HIPFFT_CHECK(hipfftExecC2C(plan, d_out, d_in, HIPFFT_BACKWARD));
    HIP_CHECK(hipDeviceSynchronize());
    clock_t t1 = clock();

    hipfftComplex *h_ifft = (hipfftComplex*)malloc(size * sizeof(hipfftComplex));
    HIP_CHECK(hipMemcpy(h_ifft, d_in, size * sizeof(hipfftComplex), hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
        h_ifft[i].x /= (float)size;
        h_ifft[i].y /= (float)size;
    }

    double max_err = 0;
    for (int i = 0; i < size; i++) {
        double e = hypot((double)h_ifft[i].x - (double)h_orig[i].x,
                (double)h_ifft[i].y - (double)h_orig[i].y);
        max_err = e > max_err ? e : max_err;
    }

    double dt = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  GPU size=%6d: error=%.2e  time=%.3f ms\n", size, max_err, dt);

    free(h_in);
    free(h_orig);
    free(h_ifft);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIPFFT_CHECK(hipfftDestroy(plan));
}

