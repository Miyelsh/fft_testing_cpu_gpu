WORKDIR  := work
ROCMLIB  := /opt/rocm/core-10.1
HIPCC    := hipcc
CXX      := g++

CPU_CXXFLAGS := -std=c++17 -O3 -march=native -ffast-math
CPU_LDLIBS   := -lm
HIP_CXXFLAGS := -O3 -I$(ROCMLIB)/include -L$(ROCMLIB)/lib
HIP_LDLIBS   := -lhipfft -lrocrand -lm

BINARIES := $(WORKDIR)/fft_compare_all \
            $(WORKDIR)/fft_compare_all_n_sweep \
            $(WORKDIR)/fft_gpu

.PHONY: all run clean

all: $(BINARIES)

$(WORKDIR):
	mkdir -p $(WORKDIR)

$(WORKDIR)/fft_compare_all: fft_compare_all.cu split_radix_fft_cpu.cpp fft_gpu_test.cu | $(WORKDIR)
	$(HIPCC) $(HIP_CXXFLAGS) fft_compare_all.cu -o $@ $(HIP_LDLIBS)

$(WORKDIR)/fft_compare_all_n_sweep: fft_compare_all_n_sweep.cpp split_radix_fft_cpu.cpp | $(WORKDIR)
	$(CXX) $(CPU_CXXFLAGS) fft_compare_all_n_sweep.cpp -o $@ $(CPU_LDLIBS)

$(WORKDIR)/fft_gpu: fft_gpu.cu | $(WORKDIR)
	$(HIPCC) $(HIP_CXXFLAGS) fft_gpu.cu -o $@ $(HIP_LDLIBS)

run: all
	LD_LIBRARY_PATH=$(ROCMLIB)/lib $(WORKDIR)/fft_compare_all
	LD_LIBRARY_PATH=$(ROCMLIB)/lib $(WORKDIR)/fft_compare_all_n_sweep
	LD_LIBRARY_PATH=$(ROCMLIB)/lib $(WORKDIR)/fft_gpu

clean:
	rm -rf $(WORKDIR)
