#ifndef UTILS_H
#define UTILS_H

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmio.h"

int count_nnz(FILE *f);
void quickSort(int arr[], int arr2[], double arr3[], int low, int high);
void init_arr(int N, double *a);
void zero_arr(int N, double *a);
void print_arr(int N, char *name, double *array);

void get_matrix_market(FILE *f, int *M, int *N, int *nz, int **row, int **col,
                       double **val);
void anti_dia_info(int *row_ptr, int *colind, double *val, int **offset, int N,
                   int *nd, int nnz, int **nnz_k);

#endif
