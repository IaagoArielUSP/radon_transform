# Define paths:
ifeq ($(origin ORIGINDIR), undefined)
	ORIGINDIR := $(shell pwd)
endif
ifeq ($(origin SRCDIR), undefined)
	SRCDIR := $(ORIGINDIR)/src
endif
ifeq ($(origin TEMPDIR), undefined)
	TEMPDIR := $(ORIGINDIR)/tmp
endif
ifeq ($(origin INCLUDEDIR), undefined)
	INCLUDEDIR := $(ORIGINDIR)/include
endif
ifeq ($(origin PYTHONDIR), undefined)
	PYTHONDIR := $(shell pwd)/python
endif

.PHONY: all
all: $(TEMPDIR)/cuda_radon.so

CCBIN := /usr/bin/gcc-9

ARCHS := -gencode arch=compute_50,code=compute_50
$(TEMPDIR)/cuda_radon.so : $(TEMPDIR)/cuda_radon.o
	cd $(TEMPDIR) && gcc $(TEMPDIR)/cuda_radon.o -shared -o $(TEMPDIR)/cuda_radon.so
$(TEMPDIR)/cuda_radon.o : $(SRCDIR)/cuda_radon.cu
	mkdir -p $(TEMPDIR)
	nvcc --compiler-bindir=$(CCBIN) -c -O3 --compiler-options '-fPIC' -o $(TEMPDIR)/cuda_radon.o -m64 $(ARCHS) $(SRCDIR)/cuda_radon.cu


