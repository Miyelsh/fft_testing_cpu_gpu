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

void gpu_fft_test(int size) {
    hipfftComplex *d_in = nullptr, *d_out = nullptr;
    hipMalloc(&d_in, size * sizeof(hipfftComplex));
    hipMalloc(&d_out, size * sizeof(hipfftComplex));

    srand(12345);
    for (int i = 0; i < size; i++) {
        d_in[i].x = ((float)rand()/((float)RAND_MAX) - 0.5f) * 2.0f * RNG_MAX;
        d_in[i].y = 0.0f;
    }

    hipfftComplex *h_orig = (hipfftComplex*)malloc(size * sizeof(hipfftComplex));
    hipMemcpy(h_orig, d_in, size * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    hipfftHandle plan;
    hipfftPlan1d(&plan, size, HIPFFT_C2C, 1);

    clock_t t0 = clock();
    hipfftExecC2C(plan, d_in, d_out, HIPFFT_FORWARD);
    hipDeviceSynchronize();
    hipfftExecC2C(plan, d_out, d_in, HIPFFT_BACKWARD);
    hipDeviceSynchronize();
    clock_t t1 = clock();

    for (int i = 0; i < size; i++) {
        d_in[i].x /= (float)size;
        d_in[i].y /= (float)size;
    }

    hipfftComplex *h_ifft = (hipfftComplex*)malloc(size * sizeof(hipfftComplex));
    hipMemcpy(h_ifft, d_in, size * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    double max_err = 0;
    for (int i = 0; i < size; i++) {
        double e = hypot((double)h_ifft[i].x - (double)h_orig[i].x,
                         (double)h_ifft[i].y - (double)h_orig[i].y);
        max_err = e > max_err ? e : max_err;
    }

    double dt = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  GPU size=%6d: error=%.2e  time=%.3f ms\n", size, max_err, dt);

    free(h_orig);
    free(h_ifft);
    hipFree(d_in);
    hipFree(d_out);
    hipfftDestroy(plan);
}

