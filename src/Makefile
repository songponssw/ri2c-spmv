SHELL := '/bin/bash'
CC = gcc


# Define vriables
UTILS := utils.c conversions.c mmio.c
INCLUDES := -I./include
LIBS := -lm

SPMV_KERNELS := spmv_kernel/spmv_coo.c \
								spmv_kernel/spmv_csr.c \
								spmv_kernel/spmv_dia.c \
								spmv_kernel/spmv_ell.c

SPMV_OMP_KERNELS := spmv_kernel/spmv_coo_omp.c \
								spmv_kernel/spmv_csr_omp.c \
								spmv_kernel/spmv_dia_omp.c \
								spmv_kernel/spmv_ell_omp.c


# This add fopenmp because there is directives duruing conversions, improving performanc.
CFLAGS := -m64 -march=native -mtune=native -O3 -Wno-unused-result -fopenmp
SRCS := $(UTILS)  
OBJS := 
ASMS := 
OUT=run



PRINT ?= 0
ifeq ($(PRINT), 1)
  CFLAGS += -DPRINT
	OUT:=$(OUT)_print
endif



TARGETS := exectime exectime_omp features ldist k_dist row_dist


# Define SRC depends on target
ifeq ($(filter exectime,$(MAKECMDGOALS)),exectime)
    SRCS += $(SPMV_KERNELS) exectime.c
endif

ifeq ($(filter exectime_omp, $(MAKECMDGOALS)), exectime_omp)
	SRCS += $(SPMV_OMP_KERNELS) exectime_omp.c 
endif

ifeq ($(filter features, $(MAKECMDGOALS)),features)
	SRCS += features.c
endif

# Ldist 
ifeq ($(filter ldist, $(MAKECMDGOALS)), ldist)
	SRCS += $(SPMV_OMP_KERNELS) ldist.c 
endif


ifeq ($(filter k_dist, $(MAKECMDGOALS)),k_dist)
	SRCS += k_dist.c
endif

ifeq ($(filter row_dist, $(MAKECMDGOALS)),row_dist)
	SRCS += row_dist.c
endif




# Define Object depends on SRCS
OBJS := $(SRCS:.c=.o)
ASMS := $(SRCS:.c=.s)



# Execute target
$(TARGETS): $(OBJS) $(ASMS)
	echo $(SPMV_KERNELS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(OUT)_$@ $(OBJS) $(LIBS)
	rm -f *.o spmv_kernel/*.o
	mv $(OUT)_$@ ../bin/
	make clean


# This target tell how to generate .o from .c files
# $< = first prereq
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

%.s: %.c
	$(CC) $(CFLAGS) -S $(INCLUDES) $< 

clean:
	rm -f $(OUT)* *.o *.s
	rm -f *.o spmv_kernel/*.o


.PHONY: exectime exectime_omp papi papi_omp features clean
