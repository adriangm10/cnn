#pragma once

#include <stddef.h>

#define MAT2D_GET(mat, i, j) mat.elems[i * mat.cols + j]

typedef struct mat {
  size_t *dims;
  size_t dim_count;
  double *elems;
} mat_t;

typedef struct {
  size_t cols;
  size_t rows;
  double *elems;
} Mat2D;

Mat2D mul_Mat2D(const Mat2D *m1, const Mat2D *m2);
void destroy_Mat2D(Mat2D *m);
Mat2D new_Mat2D(const size_t rows, const size_t cols);
