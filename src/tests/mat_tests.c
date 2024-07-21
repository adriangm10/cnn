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
  const size_t ROWS = 2000;
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
}

int main(void) {
  test_t tests[] = {mul_test, mul_performace};
  run_tests(tests, 2);
  return 0;
}
