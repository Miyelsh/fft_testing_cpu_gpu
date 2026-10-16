#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <float.h>

#define RNG_MAX 64.0

void gpu_fft_test(int N) {
    hipfftComplex *d_in = nullptr, *d_out = nullptr;
    hipMalloc(&d_in,  N * sizeof(hipfftComplex));
    hipMalloc(&d_out, N * sizeof(hipfftComplex));

    srand(12345);
    for (int i = 0; i < N; i++) {
        d_in[i].x = ((float)rand()/((float)RAND_MAX) - 0.5f) * 2.0f * RNG_MAX;
        d_in[i].y = 0.0f;
    }

    hipfftComplex *h_orig = (hipfftComplex*)malloc(N * sizeof(hipfftComplex));
    hipMemcpy(h_orig, d_in, N * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    hipfftHandle plan;
    hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);

    clock_t t0 = clock();
    hipfftExecC2C(plan, d_in, d_out, HIPFFT_FORWARD);
    hipDeviceSynchronize();
    hipfftExecC2C(plan, d_in, d_out, HIPFFT_BACKWARD);
    hipDeviceSynchronize();
    clock_t t1 = clock();

    for (int i = 0; i < N; i++) {
        d_out[i].x /= (float)N;
        d_out[i].y /= (float)N;
    }

    hipfftComplex *h_ifft = (hipfftComplex*)malloc(N * sizeof(hipfftComplex));
    hipMemcpy(h_ifft, d_out, N * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double e = hypot((double)h_ifft[i].x - (double)h_orig[i].x,
                         (double)h_ifft[i].y - (double)h_orig[i].y);
        max_err = e > max_err ? e : max_err;
    }

    double dt = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  GPU N=%6d: error=%.2e  time=%.3f ms\n", N, max_err, dt);

    free(h_orig);
    free(h_ifft);
    hipFree(d_in);
    hipFree(d_out);
    hipfftDestroy(plan);
}

int main(void) {
    printf("ROCm hipFFT N=16384 comparison:\n");
    gpu_fft_test(16384);
    return 0;
}
