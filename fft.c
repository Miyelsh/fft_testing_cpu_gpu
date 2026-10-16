#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define PI 3.14159265358979323846
static constexpr std::complex<double> I(0, 1);

void split_radix_fft_dit(std::complex<double> *a, std::complex<double> *b, int n) {
    if (n <= 1) return;
    if (n == 2) {
        std::complex<double> t = a[0] + a[1];
        std::complex<double> u = a[0] - a[1];
        a[0] = t; a[1] = u;
        return;
    }
    std::complex<double> *e = (std::complex<double>*)malloc((n / 2) * sizeof(std::complex<double>));
    std::complex<double> *o1 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *o2 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *t1 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *t2 = (std::complex<double>*)malloc((n / 4) * sizeof(std::complex<double>));
    std::complex<double> *w = (std::complex<double>*)malloc((n / 2) * sizeof(std::complex<double>));
    for (int i = 0, j = 0; j < n / 2; j++) { e[j] = a[i]; i += 2; }
    split_radix_fft_dit(e, NULL, n / 2);
    for (int j = 0; j < n / 4; j++) { o1[j] = a[4 * j + 1]; o2[j] = a[4 * j + 3]; }
    split_radix_fft_dit(o1, NULL, n / 4);
    split_radix_fft_dit(o2, NULL, n / 4);
    for (int k = 0; k < n / 2; k++) w[k] = std::polar(1.0, -2.0 * PI * k / n);
    for (int k = 0; k < n / 4; k++) w[k + n / 4] = std::polar(1.0, -4.0 * PI * k / n);
    for (int k = 0; k < n / 4; k++) {
        std::complex<double> wk2 = w[k + n / 4];
        t1[k] = o1[k] + wk2 * o2[k];
        t2[k] = o1[k] - wk2 * o2[k];
    }
    for (int k = 0; k < n / 4; k++) {
        std::complex<double> wn = w[k];
        std::complex<double> wn4 = -I * wn;
        std::complex<double> wn6 = +I * wn;
        a[k] = e[k] + wn * t1[k];
        a[k + n / 2] = e[k] - wn * t1[k];
        a[k + n / 4] = e[k + n / 4] + wn4 * t2[k];
        a[k + 3 * n / 4] = e[k + n / 4] + wn6 * t2[k];
    }
    free(e); free(o1); free(o2); free(t1); free(t2); free(w);
}

void split_radix_fft_inplace(std::complex<double> *a, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    memcpy(b, a, n * sizeof(std::complex<double>));
    split_radix_fft_dit(b, a, n);
    memcpy(a, b, n * sizeof(std::complex<double>));
    free(b);
}

void split_radix_ifft_inplace(std::complex<double> *a, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    for (int i = 0; i < n; i++) b[i] = std::conj(a[i]);
    split_radix_fft_dit(b, NULL, n);
    for (int i = 0; i < n; i++) a[i] = std::conj(b[i]) / (double)n;
    free(b);
}
