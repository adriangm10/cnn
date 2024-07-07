#include "cnn.h"
#include "mat.h"
#include "test_utils.h"

#define NN_OUTPUT(nn) nn.layers[nn.layer_count - 1].dl.a

void forward_test() {
  nn_t nn = new_nn();

  add_dense_layer(&nn, 2, 2, RELU);
  add_dense_layer(&nn, 2, 1, RELU);

  assert(nn.layers[0].dl.ws.rows == 2);
  assert(nn.layers[0].dl.ws.cols == 2);
  assert(nn.layers[0].dl.a.rows == 1);
  assert(nn.layers[0].dl.a.cols == 2);

  assert(nn.layers[1].dl.ws.rows == 2);
  assert(nn.layers[1].dl.ws.cols == 1);
  assert(nn.layers[1].dl.a.rows == 1);
  assert(nn.layers[1].dl.a.cols == 1);

  nn.layers[0].dl.bias = -3.0;
  nn.layers[1].dl.bias = 0.0;

  MAT2D_GET(nn.layers[0].dl.ws, 0, 0) = 0.0;
  MAT2D_GET(nn.layers[0].dl.ws, 0, 1) = 1.0;
  MAT2D_GET(nn.layers[0].dl.ws, 1, 0) = 1.0;
  MAT2D_GET(nn.layers[0].dl.ws, 1, 1) = 1.0;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = -1.0;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 1.0;

  double i1[] = { 0.0, 1.0 };
  double i2[] = { 1.0, 3.0 };

  Mat2D input = (Mat2D) {
    .rows = 1,
    .cols = 2,
    .elems = i1,
  };

  nn_forward(&nn, &input);

  Mat2D out = NN_OUTPUT(nn);
  assert(out.elems[0] == 0.0);

  input.elems = i2;
  nn_forward(&nn, &input);

  out = NN_OUTPUT(nn);
  assert(out.elems[0] == 1.0);

  nn_destroy(&nn);
}

int main(void) {
  test_t tests[] = {forward_test};
  run_tests(tests, 1);
  return 0;
}
