#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>

#define PI 3.14159265358979323846
#define RNG_MAX 64.0

static double complex* gen_random(int n) {
    double complex *a = malloc(n * sizeof(double complex));
    for (int i = 0; i < n; i++) {
        double r = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        a[i] = r * RNG_MAX;
    }
    return a;
}

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

    for (int i = 0, j = 0; j < n / 2; j++) {
        e[j] = a[i]; i += 2;
    }
    split_radix_fft_dit(e, NULL, n / 2);

    for (int j = 0; j < n / 4; j++) {
        o1[j] = a[4 * j + 1];
        o2[j] = a[4 * j + 3];
    }
    split_radix_fft_dit(o1, NULL, n / 4);
    split_radix_fft_dit(o2, NULL, n / 4);

    for (int k = 0; k < n / 2; k++)
        w[k] = cexp(-I * 2.0 * PI * k / n);

    for (int k = 0; k < n / 4; k++)
        w[k + n / 4] = cexp(-I * 4.0 * PI * k / n);

    for (int k = 0; k < n / 4; k++) {
        double complex wk2 = w[k + n / 4];
        t1[k] = o1[k] + wk2 * o2[k];
        t2[k] = o1[k] - wk2 * o2[k];
    }

    for (int k = 0; k < n / 4; k++) {
        double complex wn = w[k];
        double complex wn4 = -I * wn;
        double complex wn6 = +I * wn;
        a[k] = e[k] + wn * t1[k];
        a[k + n / 2] = e[k] - wn * t1[k];
        a[k + n / 4] = e[k + n / 4] + wn4 * t2[k];
        a[k + 3 * n / 4] = e[k + n / 4] + wn6 * t2[k];
    }

    free(e); free(o1); free(o2); free(t1); free(t2); free(w);
}

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
    srand(12345);
    int max_n = 16384;
    int failures = 0;
    double max_err = 0;

    printf("%-6s %-12s %-15s %s\n", "N", "Max Error", "Time (ms)", "Status");
    printf("%-6s %-12s %-15s %s\n", "----", "------------", "---------------", "------");

    for (int n = 1; n <= max_n; n *= 2) {
        double complex *original = gen_random(n);
        double complex *copy = malloc(n * sizeof(double complex));
        memcpy(copy, original, n * sizeof(double complex));

        clock_t t0 = clock();
        split_radix_fft_dit(copy, NULL, n);
        split_radix_ifft_inplace(copy, n);
        clock_t t1 = clock();

        double err = 0;
        for (int i = 0; i < n; i++) {
            double e = hypot(creal(copy[i]) - creal(original[i]), cimag(copy[i]) - cimag(original[i]));
            err = e > err ? e : err;
        }

        double dt = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
        max_err = err > max_err ? err : max_err;

        const char *status = err < 1e-8 ? "PASS" : "FAIL";
        if (err >= 1e-8) failures++;

        printf("%-6d %-12.2e %-15.3f %s\n", n, err, dt, status);

        free(original);
        free(copy);
    }

    printf("\nOverall: %d failures, max error = %.2e\n", failures, max_err);
    return failures > 0 ? 1 : 0;
}
