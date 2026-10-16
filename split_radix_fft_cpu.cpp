#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define PI 3.14159265358979323846
static constexpr std::complex<double> I(0, 1);

/**
 * Forward split-radix DIT FFT using exponential (polar) twiddles.
 *
 * Computes the n-point DFT of the first n elements of in. The even-index
 * (N/2) and stride-4 odd (two N/4) sub-transforms are combined with the
 * standard split-radix butterfly.
 *
 * @param[in]  in   Input samples; the first n are read.
 * @param[out] out  Output buffer for the result; pass NULL to compute in-place in in.
 * @param[in]  n    Transform length; must be a power of 2.
 */
void split_radix_exp_fft(std::complex<double> *in, std::complex<double> *out, int n) {
    std::complex<double> *dest = out ? out : in;
    if (n <= 1) return;
    if (n == 2) {
        std::complex<double> t = in[0] + in[1];
        std::complex<double> u = in[0] - in[1];
        dest[0] = t; dest[1] = u;
        return;
    }
    std::complex<double> *e = (std::complex<double>*)malloc((n / 2) * sizeof(std::complex<double>));
    std::complex<double> *o1 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *o2 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *t1 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *t2 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *w = (std::complex<double>*)malloc(3 * n / 4 * sizeof(std::complex<double>));
    for (int i = 0, j = 0; j < n / 2; j++) {
        e[j] = in[i];
        i += 2;
    }
    split_radix_exp_fft(e, NULL, n / 2);
    for (int j = 0; j < n / 4; j++) {
        o1[j] = in[4 * j + 1];
        o2[j] = in[4 * j + 3];
    }
    split_radix_exp_fft(o1, NULL, n / 4);
    split_radix_exp_fft(o2, NULL, n / 4);
    for (int k = 0; k < n / 2; k++) {
        w[k] = std::polar(1.0, -2.0 * PI * k / n);
    }
    for (int k = 0; k < n / 4; k++) {
        w[k + n / 2] = std::polar(1.0, -4.0 * PI * k / n);
    }
    for (int k = 0; k < n / 4; k++) {
        std::complex<double> wk2 = w[k + n / 2];
        t1[k] = o1[k] + wk2 * o2[k];
        t2[k] = o1[k] - wk2 * o2[k];
    }
    for (int k = 0; k < n / 4; k++) {
        std::complex<double> wn = w[k];
        std::complex<double> wn4 = -I * wn;
        std::complex<double> wn6 = +I * wn;
        dest[k] = e[k] + wn * t1[k];
        dest[k + n / 2] = e[k] - wn * t1[k];
        dest[k + n / 4] = e[k + n / 4] + wn4 * t2[k];
        dest[k + 3 * n / 4] = e[k + n / 4] + wn6 * t2[k];
    }
    free(e); free(o1); free(o2); free(t1); free(t2); free(w);
}

/**
 * Forward split-radix DIT FFT using cos/sin twiddles.
 *
 * Identical algorithm to split_radix_exp_fft; the only difference is that
 * twiddle factors are built with cos/sin rather than std::polar. Provides
 * the "sin/cos" reference used by the exp-vs-sincos benchmark.
 *
 * @param[in]  in   Input samples; the first n are read.
 * @param[out] out  Output buffer for the result; pass NULL to compute in-place in in.
 * @param[in]  n    Transform length; must be a power of 2.
 */
void split_radix_sincos_fft(std::complex<double> *in, std::complex<double> *out, int n) {
    std::complex<double> *dest = out ? out : in;
    if (n <= 1) return;
    if (n == 2) {
        std::complex<double> t = in[0] + in[1];
        std::complex<double> u = in[0] - in[1];
        dest[0] = t; dest[1] = u;
        return;
    }
    std::complex<double> *e = (std::complex<double>*)malloc((n / 2) * sizeof(std::complex<double>));
    std::complex<double> *o1 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *o2 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *t1 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *t2 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *w = (std::complex<double>*)malloc(3 * n / 4 * sizeof(std::complex<double>));
    for (int i = 0, j = 0; j < n / 2; j++) {
        e[j] = in[i];
        i += 2;
    }
    split_radix_sincos_fft(e, NULL, n / 2);
    for (int j = 0; j < n / 4; j++) {
        o1[j] = in[4 * j + 1];
        o2[j] = in[4 * j + 3];
    }
    split_radix_sincos_fft(o1, NULL, n / 4);
    split_radix_sincos_fft(o2, NULL, n / 4);
    for (int k = 0; k < n / 2; k++) {
        double th = 2.0 * PI * k / n;
        w[k] = cos(th) - I * sin(th);
    }
    for (int k = 0; k < n / 4; k++) {
        double th = 4.0 * PI * k / n;
        w[k + n / 2] = cos(th) - I * sin(th);
    }
    for (int k = 0; k < n / 4; k++) {
        std::complex<double> wk2 = w[k + n / 2];
        t1[k] = o1[k] + wk2 * o2[k];
        t2[k] = o1[k] - wk2 * o2[k];
    }
    for (int k = 0; k < n / 4; k++) {
        std::complex<double> wn = w[k];
        std::complex<double> wn4 = -I * wn;
        std::complex<double> wn6 = +I * wn;
        dest[k] = e[k] + wn * t1[k];
        dest[k + n / 2] = e[k] - wn * t1[k];
        dest[k + n / 4] = e[k + n / 4] + wn4 * t2[k];
        dest[k + 3 * n / 4] = e[k + n / 4] + wn6 * t2[k];
    }
    free(e); free(o1); free(o2); free(t1); free(t2); free(w);
}


/**
 * In-place forward FFT (exp variant).
 *
 * Convenience wrapper that transforms inout in place via split_radix_exp_fft.
 *
 * @param[in,out] inout  Samples to transform; result is written back here.
 * @param[in]     n      Transform length; must be a power of 2.
 */
void split_radix_exp_fft_inout(std::complex<double> *inout, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    memcpy(b, inout, n * sizeof(std::complex<double>));
    split_radix_exp_fft(b, inout, n);
    free(b);
}

/**
 * In-place inverse FFT (exp variant).
 *
 * Implements the inverse via the conjugate identity IFFT(x) = conj(FFT(conj(x)))/n,
 * so it reuses split_radix_exp_fft rather than a dedicated backward kernel.
 *
 * @param[in,out] inout  Frequency samples; the time-domain inverse is written back here.
 * @param[in]     n      Transform length; must be a power of 2.
 */
void split_radix_exp_ifft_inout(std::complex<double> *inout, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    for (int i = 0; i < n; i++) {
        b[i] = std::conj(inout[i]);
    }
    split_radix_exp_fft(b, NULL, n);
    for (int i = 0; i < n; i++) {
        inout[i] = std::conj(b[i]) / (double)n;
    }
    free(b);
}

/**
 * In-place inverse FFT (sin/cos variant).
 *
 * Inverse via IFFT(x) = conj(FFT(conj(x)))/n, reusing split_radix_sincos_fft.
 *
 * @param[in,out] inout  Frequency samples; the time-domain inverse is written back here.
 * @param[in]     n      Transform length; must be a power of 2.
 */
void split_radix_sincos_ifft_inout(std::complex<double> *inout, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    for (int i = 0; i < n; i++) {
        b[i] = std::conj(inout[i]);
    }
    split_radix_sincos_fft(b, NULL, n);
    for (int i = 0; i < n; i++) {
        inout[i] = std::conj(b[i]) / (double)n;
    }
    free(b);
}

/**
 * In-place forward FFT (sin/cos variant).
 *
 * Convenience wrapper that transforms inout in place via split_radix_sincos_fft.
 *
 * @param[in,out] inout  Samples to transform; result is written back here.
 * @param[in]     n      Transform length; must be a power of 2.
 */
void split_radix_sincos_fft_inout(std::complex<double> *inout, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    memcpy(b, inout, n * sizeof(std::complex<double>));
    split_radix_sincos_fft(b, inout, n);
    free(b);
}
