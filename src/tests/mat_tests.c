#include "mat.h"
#include "test_utils.h"

void mul_test() {
  Mat2D m1 = new_Mat2D(2, 2);
  Mat2D m2 = new_Mat2D(2, 2);

  m1.elems[0] = 3;
  m1.elems[1] = 2;
  m1.elems[2] = 1;
  m1.elems[3] = 4;

  m2.elems[0] = 5;
  m2.elems[1] = 0;
  m2.elems[2] = 6;
  m2.elems[3] = 7;

  Mat2D res = mul_Mat2D(&m1, &m2);
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

int main(void) {
  test_t tests[] = {mul_test};
  run_tests(tests, 1);
  return 0;
}
