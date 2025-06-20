#include "conversions.h"
#include <omp.h>

void coo_csr(int nz, int N, int *row, int *col, double *coo_val, int **row_ptr,
             int **colind, double **val) {
  // Allocate ptr
  int *tempRowPtr, *tempColind;
  double *tempVal;

  tempRowPtr = calloc(N + 1, sizeof(int));
  tempColind = (int *)malloc(nz * sizeof(int));
  tempVal = (double *)malloc(nz * sizeof(double));

  int i, j, j0, r, c;
  double data;

  for (i = 0; i < nz; i++) {
    tempRowPtr[row[i]]++;
  }

  j = 0;
  for (i = 0; i < N; i++) {
    j0 = tempRowPtr[i];
    tempRowPtr[i] = j;
    j += j0;
  }

  for (i = 0; i < nz; i++) {
    r = row[i];
    c = col[i];
    data = coo_val[i];
    j = tempRowPtr[r];
    tempColind[j] = c;
    tempVal[j] = data;
    tempRowPtr[r]++;
  }

  for (i = N - 1; i > 0; i--) {
    tempRowPtr[i] = tempRowPtr[i - 1];
  }
  tempRowPtr[0] = 0;
  tempRowPtr[N] = nz;

  // Derefercing back to original pointer
  *row_ptr = tempRowPtr;
  *colind = tempColind;
  *val = tempVal;
}

void csr_ell(int *row_ptr, int *colind, double *val, int **indices,
             double **data, int N, int *num_cols, int nnz) {
  int i, j, k, col, max = 0, temp = 0;
  clock_t start, stop;

  for (i = 0; i < N; i++) {
    temp = row_ptr[i + 1] - row_ptr[i];
    if (max < temp)
      max = temp;
  }
  *num_cols = max;

  *data = (double *)calloc((size_t)N * max, sizeof(double));
  if (*data == NULL) {
    fprintf(stderr, "couldn't allocate ell_data using calloc");
    exit(1);
  }
  *indices = (int *)malloc((size_t)N * max * sizeof(int));
  if (*indices == NULL) {
    fprintf(stderr, "couldn't allocate indices using malloc");
    exit(1);
  }

/*
 *
 * Add OpenMP when converting
 */
#pragma omp parallel for
  for (i = 0; i < max; i++) {
    for (j = 0; j < N; j++) {
      (*indices)[i * N + j] = -1;
    }
  }

#pragma omp parallel for
  for (i = 0; i < N; i++) {
    k = 0;
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      (*data)[k * N + i] = val[j];
      (*indices)[k * N + i] = colind[j];
      k++;
    }
  }
}

void csr_dia(int *row_ptr, int *colind, double *val, int **offset,
             double **data, int N, int *nd, int *stride, int nnz) {
  int i, j, num_diag, min, *ind, index, diag_no, col, k;
  clock_t start, stop;
  int move;
  num_diag = 0;

  ind = (int *)calloc(2 * N - 1, sizeof(int));
  if (ind == NULL) {
    fprintf(stderr, "couldn't allocate ind using calloc\n");
    exit(1);
  }

  // start = clock();
  for (i = 0; i < N; i++) {
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      if (!ind[N + colind[j] - i - 1]++)
        num_diag++;
    }
  }
  *nd = num_diag;

  *offset = (int *)malloc(num_diag * sizeof(int));
  if (*offset == NULL) {
    fprintf(stderr, "couldn't allocate *offset using malloc\n");
    exit(1);
  }
  diag_no = -((2 * N - 1) / 2);
  min = abs(diag_no);
  index = 0;
  for (i = 0; i < 2 * N - 1; i++) {
    if (ind[i]) {
      (*offset)[index++] = diag_no;
      if (min > abs(diag_no))
        min = abs(diag_no);
    }
    diag_no++;
  }
  *stride = N - min;
  size_t size = (size_t)num_diag * *stride;
  *data = (double *)malloc((size_t)(size) * sizeof(double));
  if (*data == NULL) {
    fprintf(stdout, "allocate: %f GB\n", (double)(size * 8 / 1e9));
    fprintf(stderr, "num_diag: %d\n", num_diag);
    fprintf(stderr, "stride: %d\n", *stride);
    fprintf(stderr, "couldn't allocate *data using calloc\n");
    fprintf(stderr, "%s\n", strerror(errno));
    fprintf(stderr, "%d\n", errno);
    exit(1);
  }

  for (i = 0; i < N; i++) {
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      col = colind[j];
      for (k = 0; k < num_diag; k++) {
        move = 0;
        if (col - i == (*offset)[k]) {
          if ((*offset)[k] < 0)
            move = N - *stride;
          (*data)[(size_t)k * *stride + i - move] = val[j];
          break;
        }
      }
    }
  }
}

void csr_dia2dim(int *row_ptr, int *colind, double *val, int **offset,
                 double ***data, int N, int *nd, int *stride, int nnz) {
  int i, j, num_diag, min, *ind, index, diag_no, col, k;
  clock_t start, stop;
  int move;
  num_diag = 0;

  ind = (int *)calloc(2 * N - 1, sizeof(int));
  if (ind == NULL) {
    fprintf(stderr, "couldn't allocate ind using calloc\n");
    exit(1);
  }

  // start = clock();
  // Count diagonal line
  for (i = 0; i < N; i++) {
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      if (!ind[N + colind[j] - i - 1]++)
        num_diag++;
    }
  }
  *nd = num_diag;

  // create offset array
  *offset = (int *)malloc(num_diag * sizeof(int));
  if (*offset == NULL) {
    fprintf(stderr, "couldn't allocate *offset using malloc\n");
    exit(1);
  }
  diag_no = -((2 * N - 1) / 2);
  min = abs(diag_no);
  index = 0;
  for (i = 0; i < 2 * N - 1; i++) {
    if (ind[i]) {
      (*offset)[index++] = diag_no;
      if (min > abs(diag_no))
        min = abs(diag_no);
    }
    diag_no++;
  }

  *stride = N - min;
  *data = (double **)malloc((size_t)(num_diag) * sizeof(double *));

  for (int z = 0; z < num_diag; z++)
    (*data)[z] = (double *)malloc((size_t)(*stride) * sizeof(double));

  for (i = 0; i < N; i++) {
#pragma omp parallel for
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      col = colind[j];
      for (k = 0; k < num_diag; k++) {
        int f = (*offset)[k];
        if (col - i == f) {
          (*data)[k][i + f] = val[j];
          break; // Because echo element must contain to only 1 diagonal
        }
      }
    }
  }
}

void csr_dia_feature(int *row_ptr, int *colind, double *val, int **offset,
                     double **data, int N, int *nd, int *stride, int nnz,
                     int **offset_count) {
  int i, j, num_diag, min, *ind, index, diag_no, col, k;
  clock_t start, stop;
  int move;
  num_diag = 0;

  ind = (int *)calloc(2 * N - 1, sizeof(int));
  if (ind == NULL) {
    fprintf(stderr, "couldn't allocate ind using calloc\n");
    exit(1);
  }

  // start = clock();
  for (i = 0; i < N; i++) {
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      if (!ind[N + colind[j] - i - 1]++)
        num_diag++;
    }
  }
  *nd = num_diag;

  *offset = (int *)malloc(num_diag * sizeof(int));
  *offset_count = (int *)malloc(num_diag * sizeof(int));
  if (*offset == NULL) {
    fprintf(stderr, "couldn't allocate *offset using malloc\n");
    exit(1);
  }
  diag_no = -((2 * N - 1) / 2);
  min = abs(diag_no);
  index = 0;
  for (i = 0; i < 2 * N - 1; i++) {
    if (ind[i]) {
      (*offset)[index++] = diag_no;
      if (min > abs(diag_no))
        min = abs(diag_no);
    }
    diag_no++;
  }
  *stride = N - min;
  for (i = 0; i < N; i++) {
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      col = colind[j];
      for (k = 0; k < num_diag; k++) {
        move = 0;
        if (col - i == (*offset)[k]) {
          (*offset_count)[k] += 1;
          // if ((*offset)[k] < 0)
          // move = N - *stride;
          // (*data)[(size_t)k * *stride + i - move] = val[j];
          break;
        }
      }
    }
  }
}

