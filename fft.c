#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

void split_radix_fft_dit(double complex *a, double complex *b, int n) {
    if (n <= 1) return;
    if (n == 2) {
        double complex t = a[0] + a[1];
        double complex u = a[0] - a[1];
        a[0] = t; a[1] = u;
        return;
    }

    double complex *e = malloc((n / 2) * sizeof(double complex));
    double complex *o1 = malloc((n / 4) * sizeof(double complex));
    double complex *o2 = malloc((n / 4) * sizeof(double complex));
    double complex *t1 = malloc((n / 4) * sizeof(double complex));
    double complex *t2 = malloc((n / 4) * sizeof(double complex));
    double complex *w = malloc((n / 2) * sizeof(double complex));

    /* Even-indexed elements -> N/2 DFT */
    for (int i = 0, j = 0; j < n / 2; j++) {
        e[j] = a[i]; i += 2;
    }
    split_radix_fft_dit(e, NULL, n / 2);

    /* Odd-indexed: O1 (indices 1,5,9...) and O2 (indices 3,7,11...) */
    for (int j = 0; j < n / 4; j++) {
        o1[j] = a[4 * j + 1];
        o2[j] = a[4 * j + 3];
    }
    split_radix_fft_dit(o1, NULL, n / 4);
    split_radix_fft_dit(o2, NULL, n / 4);

    /* Twiddle factors W_N^k = exp(-2*pi*i*k/n) */
    for (int k = 0; k < n / 2; k++) {
        w[k] = cexp(-I * 2.0 * PI * k / n);
    }

    /* Twiddle for O1/O2 split: W_{N/2}^k = W_N^{2k} */
    for (int k = 0; k < n / 4; k++) {
        w[k + n / 4] = cexp(-I * 4.0 * PI * k / n);
    }

    /* Butterfly: t1 = O1 + W_{N/2}^k * O2, t2 = O1 - W_{N/2}^k * O2 */
    for (int k = 0; k < n / 4; k++) {
        double complex wk2 = w[k + n / 4];
        t1[k] = o1[k] + wk2 * o2[k];
        t2[k] = o1[k] - wk2 * o2[k];
    }

    /* X[k] = E[k] + ω_N^k * t1[k] */
    /* X[k+N/2] = E[k] - ω_N^k * t1[k] */
    /* X[k+N/4] = E[k+N/4] + ω_N^{k+N/4} * t2[k] */
    /* X[k+3N/4] = E[k+N/4] - ω_N^{k+3N/4} * t2[k] */
    for (int k = 0; k < n / 4; k++) {
        double complex wn = w[k];            // ω_N^k
        double complex wn4 = -I * wn;        // ω_N^{k+N/4} = ω_N^k * (-i)
        double complex wn6 = +I * wn;        // ω_N^{k+3N/4} = ω_N^k * (+i)
        a[k] = e[k] + wn * t1[k];
        a[k + n / 2] = e[k] - wn * t1[k];
        a[k + n / 4] = e[k + n / 4] + wn4 * t2[k];
        a[k + 3 * n / 4] = e[k + n / 4] + wn6 * t2[k];
    }

    free(e);
    free(o1);
    free(o2);
    free(t1);
    free(t2);
    free(w);
}

/* In-place version using a single output buffer */
void split_radix_fft_inplace(double complex *a, int n) {
    double complex *b = malloc(n * sizeof(double complex));
    memcpy(b, a, n * sizeof(double complex));
    split_radix_fft_dit(b, a, n);
    memcpy(a, b, n * sizeof(double complex));
    free(b);
}

/* IFFT via conjugate(FFT(conjugate(x)))/N */
void split_radix_ifft_inplace(double complex *a, int n) {
    double complex *b = malloc(n * sizeof(double complex));
    for (int i = 0; i < n; i++)
        b[i] = conj(a[i]);
    split_radix_fft_dit(b, NULL, n);
    for (int i = 0; i < n; i++)
        a[i] = conj(b[i]) / n;
    free(b);
}

int main(void) {
    int n = 8;
    double complex a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double complex original[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    printf("Split-Radix DIT FFT (n=%d):\n", n);
    split_radix_fft_inplace(a, n);

    for (int i = 0; i < n; i++)
        printf("  X[%d] = %.6f + %.6fi\n", i, creal(a[i]), cimag(a[i]));

    /* IFFT verification: ifft(fft(x)) == x */
    printf("\nIFFT verification (ifft(fft(x)) == x):\n");
    split_radix_ifft_inplace(a, n);

    double max_err = 0;
    for (int i = 0; i < n; i++) {
        double err = hypot(creal(a[i]) - creal(original[i]), cimag(a[i]) - cimag(original[i]));
        max_err = (err > max_err) ? err : max_err;
        printf("  x[%d] = %.6f + %.6fi  (orig: %.6f + %.6fi)\n", i, creal(a[i]), cimag(a[i]), creal(original[i]), cimag(original[i]));
    }
    printf("\n  Max error: %.2e\n", max_err);
    printf("  %s\n", max_err < 1e-10 ? "PASS" : "FAIL");

    return 0;
}
