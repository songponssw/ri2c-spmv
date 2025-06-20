#include "utils.h"
#include "mmio.h"
#include <sys/mman.h>
#include <x86intrin.h>

void get_matrix_market(FILE *f, int *M, int *N, int *nz, int **row, int **col,
                       double **val) {

  // Matrix Market stores only the lower part of symmetric matrix
  // !! not count the main diagonal line
  *nz = count_nnz(f);
  rewind(f);

  int ret_code, entries;
  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process MMarket.\n");
    exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, M, N, &entries)) != 0) {
    exit(1);
  }

  if (*M > *N)
    N = M;
  else
    M = N;

  int *tempRow = malloc((*nz) * sizeof(int));
  int *tempCol = malloc((*nz) * sizeof(int));
  double *tempVal = malloc((*nz) * sizeof(double));

  int r, c, i;
  double v;
  int k = 0;

  if ((mm_is_symmetric(matcode)) | mm_is_skew(matcode)) {
    // if ((mm_is_symmetric(matcode))) {
    if (!mm_is_pattern(matcode)) {
      for (i = 0; i < entries; i++) {
        fscanf(f, "%d %d %lf\n", &r, &c, &v);
        if (v < 0 || v > 0) {
          tempRow[k] = r - 1;
          tempCol[k] = c - 1;
          tempVal[k] = v;
          if (fpclassify(tempVal[k]) == FP_NAN) {
            fprintf(stderr, "bad value : nan\n");
            exit(1);
          }
          if (fpclassify(tempVal[k]) == FP_INFINITE) {
            fprintf(stderr, "bad value : infinite\n");
            exit(1);
          }
          if (fpclassify(tempVal[k]) == FP_SUBNORMAL) {
            fprintf(stderr, "bad value : subnormal\n");
            tempVal[k] = 0.0;
          }
          if (r == c) {
            k++;
          } else { // Create a symmetry port from values
            tempRow[k + 1] = tempCol[k];
            tempCol[k + 1] = tempRow[k];
            tempVal[k + 1] = v;
            if (fpclassify(tempVal[k + 1]) == FP_SUBNORMAL) {
              fprintf(stderr, "bad value : subnormal\n");
              tempVal[k + 1] = 0.0;
            }
            k = k + 2;
          }
        }
      }
    } else {
      for (i = 0; i < entries; i++) {
        fscanf(f, "%d %d\n", &r, &c);
        tempRow[k] = r - 1;
        tempCol[k] = c - 1;
        tempVal[k] = 1.0;
        if (r == c) {
          k++;
        } else {
          tempRow[k + 1] = tempCol[k];
          tempCol[k + 1] = tempRow[k];
          tempVal[k + 1] = 1.0;
          k = k + 2;
        }
      }
    }
  } else { // If input doesn't a symmetry matrix
    if (!mm_is_pattern(matcode)) {
      for (i = 0; i < entries; i++) {
        fscanf(f, "%d %d %lf\n", &r, &c, &v);
        if (v < 0 || v > 0) {
          tempRow[k] = r - 1;
          tempCol[k] = c - 1;
          tempVal[k] = v;
          if (fpclassify(tempVal[k]) == FP_NAN) {
            fprintf(stderr, "bad value : nan\n");
            exit(1);
          }
          if (fpclassify(tempVal[k]) == FP_INFINITE) {
            fprintf(stderr, "bad value : infinite\n");
            exit(1);
          }
          if (fpclassify(tempVal[k]) == FP_SUBNORMAL) {
            fprintf(stderr, "bad value : subnormal\n");
            tempVal[k] = 0.0;
          }
          k++;
        }
      }
    } else {
      for (i = 0; i < entries; i++) {
        fscanf(f, "%d %d\n", &r, &c);
        tempRow[i] = r - 1;
        tempCol[i] = c - 1;
        tempVal[i] = 1.0;
      }
    }
  }

  // Needed to sort value. Specifically for symetric case
  quickSort(tempRow, tempCol, tempVal, 0, (*nz) - 1);
  *row = tempRow;
  *col = tempCol;
  *val = tempVal;
}

int count_nnz(FILE *f) {
  MM_typecode matcode;
  int i, M, N, entries, anz, r, c, ret_code;
  double v;

  if (mm_read_banner(f, &matcode) != 0) {
    fprintf(stderr, "Could not process Matrix Market banner.\n");
    exit(1);
  }
  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode)) {
    fprintf(stderr, "Sorry, this application does not support ");
    fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &entries)) != 0)
    exit(1);

  /* reseve memory for matrices */

  if (mm_is_symmetric(matcode)) {
    anz = 0;
    if (!mm_is_pattern(matcode)) {
      for (i = 0; i < entries; i++) {
        fscanf(f, "%d %d %lf\n", &r, &c, &v);
        if (v < 0 || v > 0) {
          if (r == c)
            anz++;
          else
            anz = anz + 2;
        }
      }
    } else {
      for (i = 0; i < entries; i++) {
        fscanf(f, "%d %d\n", &r, &c);
        if (r == c)
          anz++;
        else
          anz = anz + 2;
      }
    }
  } else {
    anz = 0;
    for (i = 0; i < entries; i++) {
      fscanf(f, "%d %d %lf\n", &r, &c, &v);
      if (v < 0 || v > 0)
        anz++;
    }
  }
  return anz;
}

void swap(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}
void swap_val(double *a, double *b) {
  double t = *a;
  *a = *b;
  *b = t;
}

void quickSort(int arr[], int arr2[], double arr3[], int left, int right) {
  int i = left, j = right;
  int pivot = arr[(left + right) / 2];
  int pivot_col = arr2[(left + right) / 2];

  /* partition */
  while (i <= j) {
    while (arr[i] < pivot || (arr[i] == pivot && arr2[i] < pivot_col))
      i++;
    while (arr[j] > pivot || (arr[j] == pivot && arr2[j] > pivot_col))
      j--;
    if (i <= j) {
      swap(&arr[i], &arr[j]);
      swap(&arr2[i], &arr2[j]);
      swap_val(&arr3[i], &arr3[j]);
      i++;
      j--;
    }
  }

  /* recursion */
  if (left < j)
    quickSort(arr, arr2, arr3, left, j);
  if (i < right)
    quickSort(arr, arr2, arr3, i, right);
}

void init_arr(int N, double *a) {
  int i;
  for (i = 0; i < N; i++) {
    a[i] = i;
  }
}

// void init_x0(int N, double *a) {
//   int i;
//   a[0] = 1;
//   for (i = 0; i < N; i++) {
//     a[i] = a[0];
//   }
// }

void zero_arr(int N, double *a) {
  int i;
  for (i = 0; i < N; i++) {
    a[i] = 0.0;
  }
}

// print array to std out
void print_arr(int N, char *name, double *array) {
  int i, j;
  printf("\n%s\n", name);
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      printf("%g\t", array[N * i + j]);
    }
    printf("\n");
  }
}

// Anti diagonal
void anti_dia_info(int *row_ptr, int *colind, double *val, int **offset, int N,
                   int *nd, int nnz, int **nnz_k) {
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
      // if (!ind[N + colind[j] - i - 1]++)
      if (!ind[i + colind[j]]++) // This also count nnz in a offset
        num_diag++;
    }
  }
  *nd = num_diag;

  *offset = (int *)malloc(num_diag * sizeof(int));
  *nnz_k = (int *)malloc(num_diag * sizeof(int));
  if (*offset == NULL) {
    fprintf(stderr, "couldn't allocate *offset using malloc\n");
    exit(1);
  }
  diag_no = -((2 * N - 1) / 2);
  min = abs(diag_no);
  index = 0;
  for (i = 0; i < 2 * N - 1; i++) {
    if (ind[i]) {
      (*offset)[index] = diag_no;
      (*nnz_k)[index] = ind[i];
      index++;

      if (min > abs(diag_no))
        min = abs(diag_no);
    }
    diag_no++;
  }
}
