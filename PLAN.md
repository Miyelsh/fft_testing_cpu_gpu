# REFACTORING PLAN — split_radix repo

## Goal

Rename files for clarity, merge bench into compare_all, add sincos IFFT, remove gen_random().

## Final file layout

| File | Purpose |
|------|---------|
| `split_radix_fft_cpu.cpp` | Canonical algorithm (renamed from `fft_split_radix.cpp`) |
| `fft_compare_all.cu` | N=16384 CPU+GPU round-trip + exp vs sin/cos benchmark (merged from bench + entry) |
| `fft_compare_all_n_sweep.cpp` | Power-of-2 sweep N=1…16384, tests split-radix + sincos (renamed from `split_radix_test.cpp`) |
| `fft_gpu.cu` | GPU hipFFT implementation (unchanged) |
| `fft_gpu_test.cu` | GPU test function (unchanged) |
| `AGENTS.md` | Updated build commands + file references |

## File renames (git mv)

| Old | New |
|-----|-----|
| `fft_split_radix.cpp` | `split_radix_fft_cpu.cpp` |
| `split_radix_entry.cu` | `fft_compare_all.cu` |
| `split_radix_test.cpp` | `fft_compare_all_n_sweep.cpp` |
| `split_radix_bench.cpp` | **DELETED** (merged into `fft_compare_all.cu`) |

## Step-by-step actions

### Step 1: Rename `fft_split_radix.cpp` → `split_radix_fft_cpu.cpp`

```sh
git mv fft_split_radix.cpp split_radix_fft_cpu.cpp
```

Update all `#include "fft_split_radix.cpp"` → `#include "split_radix_fft_cpu.cpp"` in:
- `split_radix_test.cpp` → `fft_compare_all_n_sweep.cpp`
- `split_radix_entry.cu` → `fft_compare_all.cu`
- `split_radix_bench.cpp` → **deleted** (contents merged into `fft_compare_all.cu`)

### Step 2: Add `split_radix_ifft_sincos_inout` to `split_radix_fft_cpu.cpp`

Append after `split_radix_ifft_inout`:

```cpp
void split_radix_ifft_sincos_inout(std::complex<double> *inout, int n) {
    std::complex<double> *b = (std::complex<double>*)malloc(n * sizeof(std::complex<double>));
    for (int i = 0; i < n; i++) b[i] = std::conj(inout[i]);
    split_radix_fft_sincos(b, n);
    for (int i = 0; i < n; i++) inout[i] = std::conj(b[i]) / (double)n;
    free(b);
}
```

### Step 3: Merge bench into `fft_compare_all.cu`

Current `split_radix_entry.cu` (→ `fft_compare_all.cu`) gains the benchmark section from `split_radix_bench.cpp`:

```cpp
    /* --- exp vs sin/cos benchmark --- */
    std::complex<double> *orig = (std::complex<double>*)malloc(MAXN * sizeof(std::complex<double>));
    for (int i = 0; i < MAXN; i++) orig[i] = a_orig[i];

    clock_t t0, t1;
    double exp_ms, sincos_ms;

    /* exp variant */
    memcpy(a, orig, MAXN * sizeof(std::complex<double>));
    t0 = clock(); split_radix_fft(a, NULL, MAXN); t1 = clock();
    exp_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  %-20s %10.3f ms\n", "exp twiddle", exp_ms);

    /* sin/cos variant */
    memcpy(a, orig, MAXN * sizeof(std::complex<double>));
    t0 = clock(); split_radix_fft_sincos(a, MAXN); t1 = clock();
    sincos_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  %-20s %10.3f ms\n", "sin/cos twiddle", sincos_ms);
    printf("  -> sin/cos speedup: %5.1fx\n", exp_ms / sincos_ms);

    free(orig);
```

Full `fft_compare_all.cu` content:

```cpp
#include <cstdio>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <float.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#define MAXN 16384
#define RNG_MAX 64.0

// Include CPU algorithm from split_radix_fft_cpu.cpp
#include "split_radix_fft_cpu.cpp"
// Include GPU test function from fft_gpu_test.cu
#include "fft_gpu_test.cu"

int main(void) {
    printf("=== FFT compare + benchmark (N=%d) ===\n\n", MAXN);

    std::complex<double> *a = (std::complex<double>*)malloc(MAXN * sizeof(std::complex<double>));
    std::complex<double> *a_orig = (std::complex<double>*)malloc(MAXN * sizeof(std::complex<double>));
    srand(42);
    for (int i = 0; i < MAXN; i++) {
        a[i] = std::complex<double>(
            ((double)rand() / RAND_MAX - 0.5) * 2.0 * RNG_MAX,
            0.0
        );
        a_orig[i] = a[i];
    }

    /* --- CPU FFT+IFFT round-trip --- */
    printf("[CPU] FFT + IFFT (round-trip):\n");
    split_radix_fft_inout(a, MAXN);
    split_radix_ifft_inout(a, MAXN);
    double max_err_cpu = 0;
    for (int i = 0; i < MAXN; i++) {
        double e = hypot(a[i].real() - a_orig[i].real(), a[i].imag() - a_orig[i].imag());
        max_err_cpu = e > max_err_cpu ? e : max_err_cpu;
    }
    printf("  Max error: %.2e  ", max_err_cpu);
    printf("%s\n\n", max_err_cpu < 1e-10 ? "PASS" : "FAIL");

    /* --- GPU hipFFT + IFFT round-trip --- */
    printf("[GPU] hipFFT + IFFT (round-trip):\n");
    gpu_fft_test(MAXN);

    /* --- Cross-check CPU at N=256 --- */
    int N_small = 256;
    std::complex<double> *s = (std::complex<double>*)malloc(N_small * sizeof(std::complex<double>));
    std::complex<double> *s_orig = (std::complex<double>*)malloc(N_small * sizeof(std::complex<double>));
    srand(42);
    for (int i = 0; i < N_small; i++) {
        s[i] = std::complex<double>(
            ((double)rand() / RAND_MAX - 0.5) * 2.0 * RNG_MAX,
            0.0
        );
        s_orig[i] = s[i];
    }
    split_radix_fft_inout(s, N_small);
    split_radix_ifft_inout(s, N_small);
    double max_err_256 = 0;
    for (int i = 0; i < N_small; i++) {
        double e = hypot(s[i].real() - s_orig[i].real(), s[i].imag() - s_orig[i].imag());
        max_err_256 = e > max_err_256 ? e : max_err_256;
    }
    printf("[CPU] N=256 round-trip max error: %.2e  %s\n", max_err_256, max_err_256 < 1e-12 ? "PASS" : "FAIL");

    /* --- exp vs sin/cos benchmark --- */
    std::complex<double> *orig = (std::complex<double>*)malloc(MAXN * sizeof(std::complex<double>));
    for (int i = 0; i < MAXN; i++) orig[i] = a_orig[i];

    clock_t t0, t1;
    double exp_ms, sincos_ms;

    /* exp variant */
    memcpy(a, orig, MAXN * sizeof(std::complex<double>));
    t0 = clock(); split_radix_fft(a, NULL, MAXN); t1 = clock();
    exp_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  %-20s %10.3f ms\n", "exp twiddle", exp_ms);

    /* sin/cos variant */
    memcpy(a, orig, MAXN * sizeof(std::complex<double>));
    t0 = clock(); split_radix_fft_sincos(a, MAXN); t1 = clock();
    sincos_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("  %-20s %10.3f ms\n", "sin/cos twiddle", sincos_ms);
    printf("  -> sin/cos speedup: %5.1fx\n", exp_ms / sincos_ms);

    free(a); free(a_orig); free(s); free(s_orig); free(orig);
    return 0;
}
```

### Step 4: Create `fft_compare_all_n_sweep.cpp` (rename of `split_radix_test.cpp`)

```cpp
#include "split_radix_fft_cpu.cpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <float.h>

#define MAXN 16384
#define RNG_MAX 64.0

int main(void) {
    printf("=== N sweep: split-radix + sincos round-trip ===\n\n");

    /* --- split_radix_fft + split_radix_ifft_inout --- */
    printf("[split_radix_fft + split_radix_ifft_inout]\n");
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
        split_radix_fft(copy, NULL, n);
        split_radix_ifft_inout(copy, n);
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

    /* --- split_radix_fft_sincos + split_radix_ifft_sincos_inout --- */
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
        split_radix_fft_sincos(copy, n);
        split_radix_ifft_sincos_inout(copy, n);
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
```

### Step 5: Update `AGENTS.md`

Replace all old names with new names in build commands and algorithm section:

```markdown
# qwen_fft_testing

Split-radix DIT FFT in C + ROCm hipFFT GPU version.

## Build & Run

```sh
# CPU FFT + IFFT test (N=16384)
hipcc -O3 -I/opt/rocm/core-10.1/include -L/opt/rocm/core-10.1/lib -lhipfft -lrocrand -lm fft_compare_all.cu -o fft_compare_all
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ./fft_compare_all

# CPU N sweep (powers of 2, 1→16384)
g++ -std=c++17 -O3 -march=native -ffast-math fft_compare_all_n_sweep.cpp -lm -o fft_compare_all_n_sweep
./fft_compare_all_n_sweep

# GPU (ROCm core-10.1)
hipcc -O3 -I/opt/rocm/core-10.1/include -L/opt/rocm/core-10.1/lib -lhipfft -lrocrand fft_gpu.cu -o fft_gpu
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ./fft_gpu

# GPU test
hipcc -O3 -I/opt/rocm/core-10.1/include -L/opt/rocm/core-10.1/lib -lhipfft -lrocrand fft_gpu_test.cu -o fft_gpu_test
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ./fft_gpu_test
```

## Algorithm

- `split_radix_fft_cpu.cpp` — split-radix DIT butterfly: N/2 (even) + two N/4 (stride-4 odd: 1,5,9... and 3,7,11...); CUDA-compatible `.cpp` (uses `std::complex`)
- IFFT = `conjugate(FFT(conjugate(x)))/N` — no duplicate code
- `fft_compare_all_n_sweep.cpp` — sweeps N=1…16384, tests split-radix + sincos round-trip
- `fft_compare_all.cu` — unified CPU+GPU test at N=16384 + exp vs sin/cos benchmark

## Gotchas

- **hipFFT does NOT divide by N on inverse** — must do `d_in[i] /= N` manually after BACKWARD (see `fft_gpu_test.cu:37-39`)
- GPU uses `float2` internally, not `double`; N=16384 float error ~2.7e-5
- IFFT in `fft_gpu_test.cu` reads from `d_out` (FFT output), not `d_in`
- Benchmarks show sin/cos + `-O3 -march=native -ffast-math` ≈ 4× faster than `exp` at `-O0`
- No Makefile or test framework — everything is a single `g++`/`hipcc` command
```

### Step 6: Delete `split_radix_bench.cpp`

```sh
rm split_radix_bench.cpp
```

## Summary of changes

| Action | Files affected |
|--------|---------------|
| Rename `fft_split_radix.cpp` → `split_radix_fft_cpu.cpp` | `split_radix_fft_cpu.cpp`, `fft_compare_all.cu`, `fft_compare_all_n_sweep.cpp` |
| Rename `split_radix_entry.cu` → `fft_compare_all.cu` | `fft_compare_all.cu` |
| Rename `split_radix_test.cpp` → `fft_compare_all_n_sweep.cpp` | `fft_compare_all_n_sweep.cpp` |
| Merge bench into `fft_compare_all.cu` | `fft_compare_all.cu` (gains benchmark), `split_radix_bench.cpp` (deleted) |
| Add `split_radix_ifft_sincos_inout` | `split_radix_fft_cpu.cpp` |
| Update `AGENTS.md` | `AGENTS.md` |

## Verification checklist

After all changes:
- [ ] `fft_compare_all.cu` compiles with `hipcc` → passes
- [ ] `fft_compare_all_n_sweep.cpp` compiles with `g++` → passes
- [ ] `fft_gpu.cu` compiles with `hipcc` → passes
- [ ] `fft_gpu_test.cu` compiles with `hipcc` → passes
- [ ] No remaining references to old names in `AGENTS.md`
