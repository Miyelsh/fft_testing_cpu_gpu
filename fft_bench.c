#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>

#define PI 3.14159265358979323846
#define N_MAX 16384

void fft_cexp(double complex *a, int n) {
    if (n <= 1) return;
    if (n == 2) { a[0] += a[1]; a[1] = a[0] - 2*a[1]; return; }
    double complex *e=malloc((n/2)*sizeof(double complex));
    double complex *o1=malloc((n/4)*sizeof(double complex));
    double complex *o2=malloc((n/4)*sizeof(double complex));
    double complex *t1=malloc((n/4)*sizeof(double complex));
    double complex *t2=malloc((n/4)*sizeof(double complex));
    double complex *w=malloc((n/2)*sizeof(double complex));
    for(int i=0,j=0;j<n/2;j++){e[j]=a[i];i+=2;}
    fft_cexp(e,n/2);
    for(int j=0;j<n/4;j++){o1[j]=a[4*j+1];o2[j]=a[4*j+3];}
    fft_cexp(o1,n/4); fft_cexp(o2,n/4);
    for(int k=0;k<n/2;k++) w[k]=cexp(-I*2.0*PI*k/n);
    for(int k=0;k<n/4;k++) w[k+n/4]=cexp(-I*4.0*PI*k/n);
    for(int k=0;k<n/4;k++){double complex wk2=w[k+n/4];t1[k]=o1[k]+wk2*o2[k];t2[k]=o1[k]-wk2*o2[k];}
    for(int k=0;k<n/4;k++){double complex wn=w[k],wn4=-I*wn,wn6=+I*wn;a[k]=e[k]+wn*t1[k];a[k+n/2]=e[k]-wn*t1[k];a[k+n/4]=e[k+n/4]+wn4*t2[k];a[k+3*n/4]=e[k+n/4]+wn6*t2[k];}
    free(e);free(o1);free(o2);free(t1);free(t2);free(w);
}

void fft_sincos(double complex *a, int n) {
    if (n <= 1) return;
    if (n == 2) { a[0] += a[1]; a[1] = a[0] - 2*a[1]; return; }
    double complex *e=malloc((n/2)*sizeof(double complex));
    double complex *o1=malloc((n/4)*sizeof(double complex));
    double complex *o2=malloc((n/4)*sizeof(double complex));
    double complex *t1=malloc((n/4)*sizeof(double complex));
    double complex *t2=malloc((n/4)*sizeof(double complex));
    double complex *w=malloc((n/2)*sizeof(double complex));
    for(int i=0,j=0;j<n/2;j++){e[j]=a[i];i+=2;}
    fft_sincos(e,n/2);
    for(int j=0;j<n/4;j++){o1[j]=a[4*j+1];o2[j]=a[4*j+3];}
    fft_sincos(o1,n/4); fft_sincos(o2,n/4);
    for(int k=0;k<n/2;k++){double th=2.0*PI*k/n;w[k]=cos(th)-I*sin(th);}
    for(int k=0;k<n/4;k++){double th=4.0*PI*k/n;w[k+n/4]=cos(th)-I*sin(th);}
    for(int k=0;k<n/4;k++){double complex wk2=w[k+n/4];t1[k]=o1[k]+wk2*o2[k];t2[k]=o1[k]-wk2*o2[k];}
    for(int k=0;k<n/4;k++){double complex wn=w[k],wn4=-I*wn,wn6=+I*wn;a[k]=e[k]+wn*t1[k];a[k+n/2]=e[k]-wn*t1[k];a[k+n/4]=e[k+n/4]+wn4*t2[k];a[k+3*n/4]=e[k+n/4]+wn6*t2[k];}
    free(e);free(o1);free(o2);free(t1);free(t2);free(w);
}

int main(void) {
    srand(42);
    double complex *a = malloc(N_MAX * sizeof(double complex));
    for(int i=0;i<N_MAX;i++) a[i] = ((double)rand()/RAND_MAX-0.5)*128.0;

    clock_t t0, t1;
    double cexp_ms, sincos_ms;

    /* cexp, -O0 */
    memcpy(a, a, N_MAX*sizeof(double complex));
    t0=clock(); fft_cexp(a,N_MAX); t1=clock();
    cexp_ms = 1000.0*(t1-t0)/CLOCKS_PER_SEC;
    printf("%-20s %10.3f ms\n","cexp + -O0",cexp_ms);

    /* sincos, -O0 */
    memcpy(a, a, N_MAX*sizeof(double complex));
    t0=clock(); fft_sincos(a,N_MAX); t1=clock();
    sincos_ms = 1000.0*(t1-t0)/CLOCKS_PER_SEC;
    printf("%-20s %10.3f ms\n","sin/cos + -O0",sincos_ms);
    printf("  -> sin/cos speedup: %5.1fx\n",cexp_ms/sincos_ms);

    free(a);
    return 0;
}
