#include "mat.h"
#include "test_utils.h"
#include <bits/time.h>
#include <time.h>

void mul_test() {
  Mat2D m1 = new_Mat2D(2, 2);
  Mat2D m2 = new_Mat2D(2, 2);
  Mat2D res = new_Mat2D(2, 2);

  m1.elems[0] = 3;
  m1.elems[1] = 2;
  m1.elems[2] = 1;
  m1.elems[3] = 4;

  m2.elems[0] = 5;
  m2.elems[1] = 0;
  m2.elems[2] = 6;
  m2.elems[3] = 7;

  mul_Mat2D(&m1, &m2, &res);
  assert(res.cols == 2);
  assert(res.rows == 2);
  assert(res.elems[0] == 27);
  assert(res.elems[1] == 14);
  assert(res.elems[2] == 29);
  assert(res.elems[3] == 28);

  destroy_Mat2D(&m1);
  destroy_Mat2D(&m2);
  destroy_Mat2D(&res);
}

void mul_performace() {
  struct timespec start, end;
  const size_t ROWS = 1000;
  const size_t COLS = 1500;
  double t = 0.0;

  Mat2D m1 = new_Mat2D(ROWS, COLS);
  Mat2D m2 = new_Mat2D(COLS, ROWS);
  Mat2D out = new_Mat2D(ROWS, ROWS);

  for (int i = 0; i < 15; ++i) {
    random_init_Mat2D(&m1, -100, 100);
    random_init_Mat2D(&m2, -100, 100);
    random_init_Mat2D(&out, -100, 100);

    clock_gettime(CLOCK_REALTIME, &start);
    mul_Mat2D(&m1, &m2, &out);
    clock_gettime(CLOCK_REALTIME, &end);
    t += (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/(double)1e9;
  }

  printf("mean exec time: %f\n", t / 15.0);
  destroy_Mat2D(&m1);
  destroy_Mat2D(&m2);
  destroy_Mat2D(&out);
}

void vec_mul_performance() {
  struct timespec start, end;
  const size_t ROWS = 2500;
  const size_t COLS = 3000;
  double t_vec_mat = 0.0, t_mat_mat = 0.0, t_mat_vec = 0.0;

  Mat2D m1 = new_Mat2D(1, COLS);
  Mat2D m2 = new_Mat2D(COLS, ROWS);
  Mat2D out = new_Mat2D(1, ROWS);

  Mat2D mat = new_Mat2D(ROWS, COLS);
  Mat2D col = new_Mat2D(COLS, 1);
  Mat2D out1 = new_Mat2D(ROWS, 1);

  for (int i = 0; i < 15; ++i) {
    random_init_Mat2D(&m1, -100, 100);
    random_init_Mat2D(&m2, -100, 100);
    random_init_Mat2D(&out, -100, 100);
    random_init_Mat2D(&mat, -100, 100);
    random_init_Mat2D(&col, -100, 100);
    random_init_Mat2D(&out1, -100, 100);

    clock_gettime(CLOCK_REALTIME, &start);
    vec_Mat2D_mul(&m1, &m2, &out);
    clock_gettime(CLOCK_REALTIME, &end);
    t_vec_mat += (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/(double)1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    Mat2D_col_mul(&mat, &col, &out1);
    clock_gettime(CLOCK_REALTIME, &end);
    t_mat_vec += (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/(double)1e9;

    clock_gettime(CLOCK_REALTIME, &start);
    mul_Mat2D(&m1, &m2, &out);
    clock_gettime(CLOCK_REALTIME, &end);
    t_mat_mat += (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/(double)1e9;
  }

  printf("mean vec_mat exec time: %f\n", t_vec_mat / 15.0);
  printf("mean mat_vec exec time: %f\n", t_mat_vec / 15.0);
  printf("mean mat_mat exec time: %f\n", t_mat_mat / 15.0);

  destroy_Mat2D(&m1);
  destroy_Mat2D(&m2);
  destroy_Mat2D(&out);
  destroy_Mat2D(&mat);
  destroy_Mat2D(&col);
  destroy_Mat2D(&out1);
}

void conv_0padding_1stride_test() {
  double i1[] = {
    1.2, 1.5, 2.1, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0,
  };

  double f1[] = {
    1.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    1.0, 0.0, 1.0
  };

  Mat2D input = {
    .cols = 5,
    .rows = 5,
    .elems = i1,
  };

  Mat2D filter = {
    .cols = 3,
    .rows = 3,
    .elems = f1,
  };

  double out[9];
  Mat2D output = {
    .cols = 3,
    .rows = 3,
    .elems = out,
  };

  convolution2D(&input, &filter, 1, 0, &output);

  print_Mat2D(&output, "\n");

  assert(out[0] == 5.3); assert(out[1] == 3.5); assert(out[2] == 5.1);
  assert(out[3] == 2.0); assert(out[4] == 4.0); assert(out[5] == 3.0);
  assert(out[6] == 2.0); assert(out[7] == 3.0); assert(out[8] == 4.0);
}

void conv_0padding_2stride_test() {
  double i1[] = {
    1.2, 1.5, 2.1, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0,
  };

  double f1[] = {
    1.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    1.0, 0.0, 1.0
  };

  Mat2D input = {
    .cols = 5,
    .rows = 5,
    .elems = i1,
  };

  Mat2D filter = {
    .cols = 3,
    .rows = 3,
    .elems = f1,
  };

  double out[4];
  Mat2D output = {
    .cols = 2,
    .rows = 2,
    .elems = out,
  };

  convolution2D(&input, &filter, 2, 0, &output);

  print_Mat2D(&output, "\n");

  assert(out[0] == 5.3); assert(out[1] == 5.1);
  assert(out[2] == 2.0); assert(out[3] == 4.0);
}

void conv_2padding_1stride_test() {
  double i1[] = {
    1.2, 1.5, 2.1, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0,
  };

  double f1[] = {
    1.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    1.0, 0.0, 1.0
  };

  Mat2D input = {
    .cols = 5,
    .rows = 5,
    .elems = i1,
  };

  Mat2D filter = {
    .cols = 3,
    .rows = 3,
    .elems = f1,
  };

  double out[49];
  Mat2D output = {
    .cols = 7,
    .rows = 7,
    .elems = out,
  };

  convolution2D(&input, &filter, 1, 2, &output);

  print_Mat2D(&output, "\n");

  assert(out[0] == 1.2);  assert(out[1] == 1.5);  assert(out[2] == 3.3);  assert(out[3] == 1.5);  assert(out[4] == 2.1);  assert(out[5] == 0.0);  assert(out[6] == 0.0);
  assert(out[7] == 0.0);  assert(out[8] == 2.2);  assert(out[9] == 2.5);  assert(out[10] == 4.1); assert(out[11] == 1.0); assert(out[12] == 1.0); assert(out[13] == 0.0);
  assert(out[14] == 1.2); assert(out[15] == 1.5); assert(out[16] == 5.3); assert(out[17] == 3.5); assert(out[18] == 5.1); assert(out[19] == 1.0); assert(out[20] == 1.0);
  assert(out[21] == 0.0); assert(out[22] == 1.0); assert(out[23] == 2.0); assert(out[24] == 4.0); assert(out[25] == 3.0); assert(out[26] == 3.0); assert(out[27] == 0.0);
  assert(out[28] == 0.0); assert(out[29] == 1.0); assert(out[30] == 2.0); assert(out[31] == 3.0); assert(out[32] == 4.0); assert(out[33] == 1.0); assert(out[34] == 1.0);
  assert(out[35] == 0.0); assert(out[36] == 0.0); assert(out[37] == 2.0); assert(out[38] == 2.0); assert(out[39] == 1.0); assert(out[40] == 1.0); assert(out[41] == 0.0);
  assert(out[42] == 0.0); assert(out[43] == 1.0); assert(out[44] == 1.0); assert(out[45] == 1.0); assert(out[46] == 1.0); assert(out[47] == 0.0); assert(out[48] == 0.0);
}

void max_pooling_test() {
  double i1[] = {
    1.2, 1.5, 2.1, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0,
  };

  Mat2D input = {
    .cols = 5,
    .rows = 5,
    .elems = i1,
  };

  double out[4];
  Mat2D output = {
    .cols = 2,
    .rows = 2,
    .elems = out,
  };

  max_pooling2D(&input, &output, 2);

  print_Mat2D(&output, "\n");

  assert(out[0] == 1.5); assert(out[1] == 2.1);
  assert(out[2] == 0.0); assert(out[3] == 1.0);
}

void avg_pooling_test() {
  double i1[] = {
    1.2, 1.5, 2.1, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0,
  };

  Mat2D input = {
    .cols = 5,
    .rows = 5,
    .elems = i1,
  };

  double out[4];
  Mat2D output = {
    .cols = 2,
    .rows = 2,
    .elems = out,
  };

  avg_pooling2D(&input, &output, 2);

  print_Mat2D(&output, "\n");

  assert(out[0] == 0.925); assert(out[1] == 1.025);
  assert(out[2] == 0.0); assert(out[3] == 1.0);
}

int main(void) {
  test_t tests[] = {
    mul_test,
    conv_0padding_1stride_test,
    conv_0padding_2stride_test,
    conv_2padding_1stride_test,
    max_pooling_test,
    avg_pooling_test,
  };

  run_tests(tests, 6);
  return 0;
}
