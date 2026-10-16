#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <float.h>


#define N 16384
#define RNG_MAX 64.0

/**
 * Convert a double-precision complex sample to a hipFFT float2.
 *
 * @param[in] r   Real part.
 * @param[in] im  Imaginary part.
 * @return The sample as a hipfftComplex (float precision).
 */
hipfftComplex to_hip(double r, double im) {
    hipfftComplex c;
    c.x = (float)r;
    c.y = (float)im;
    return c;
}

/**
 * Abort the program if a hipFFT call did not succeed.
 *
 * @param[in] e    hipFFT result code to check.
 * @param[in] msg  Label identifying the failing call, printed on failure.
 */
void check(hipfftResult_t e, const char *msg) {
    if (e != HIPFFT_SUCCESS) { fprintf(stderr, "%s: %d\n", msg, e); exit(1); }
}
/**
 * Abort the program if a HIP runtime call did not succeed.
 *
 * @param[in] e    HIP error code to check.
 * @param[in] msg  Label identifying the failing call, printed on failure.
 */
void check_hip(hipError_t e, const char *msg) {
    if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", msg, hipGetErrorString(e)); exit(1); }
}

/**
 * Standalone hipFFT round-trip test.
 *
 * Generates a random real signal, runs a forward then inverse hipFFT on the
 * GPU, divides the inverse by N (hipFFT does not scale the backward
 * transform), and reports the max error against the original.
 *
 * @return 0 on success.
 */
int main(void) {
    srand(12345);

    std::complex<double> *h_in  = (std::complex<double>*)malloc(N * sizeof(std::complex<double>));
    std::complex<double> *h_out = (std::complex<double>*)malloc(N * sizeof(std::complex<double>));
    for (int i = 0; i < N; i++)
        h_in[i] = std::complex<double>((double)rand()/RAND_MAX - 0.5, 0.0);

    hipfftComplex *d_in, *d_out;
    check_hip(hipMalloc(&d_in,  N * sizeof(hipfftComplex)), "hipMalloc d_in");
    check_hip(hipMalloc(&d_out, N * sizeof(hipfftComplex)), "hipMalloc d_out");

    hipfftComplex *h2d = (hipfftComplex*)malloc(N * sizeof(hipfftComplex));

    for (int i = 0; i < N; i++) {
        h2d[i] = to_hip(h_in[i].real(), h_in[i].imag());
    }

    check_hip(hipMemcpy(d_in, h2d, N * sizeof(hipfftComplex), hipMemcpyHostToDevice), "memcpy H2D");
    free(h2d);

    hipfftHandle plan;
    check(hipfftPlan1d(&plan, N, HIPFFT_C2C, 1), "hipfftPlan1d");

    check(hipfftExecC2C(plan, d_in, d_out, HIPFFT_FORWARD), "hipfftExec C2C fwd");
    check(hipfftExecC2C(plan, d_out, d_in, HIPFFT_BACKWARD), "hipfftExec C2C bwd");

    hipfftComplex *d2h = (hipfftComplex*)malloc(N * sizeof(hipfftComplex));
    check_hip(hipMemcpy(d2h, d_in, N * sizeof(hipfftComplex), hipMemcpyDeviceToHost), "memcpy D2H");
    for (int i = 0; i < N; i++) {
        double r = d2h[i].x / (float)N;
        double im = d2h[i].y / (float)N;
        h_out[i] = std::complex<double>(r, im);
    }
    free(d2h);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double e = hypot(h_out[i].real() - h_in[i].real(), h_out[i].imag() - h_in[i].imag());
        max_err = e > max_err ? e : max_err;
    }
    printf("GPU IFFT(FFT(x)) error: %.2e\n", max_err);

    hipfftDestroy(plan);
    check_hip(hipFree(d_in), "hipFree d_in");
    check_hip(hipFree(d_out), "hipFree d_out");
    free(h_in);
    free(h_out);
    return 0;
}
