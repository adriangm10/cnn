#include "cnn.h"
#include "mat.h"
#include "test_utils.h"
#include <math.h>

#define NN_OUTPUT(nn) nn.layers[nn.layer_count - 1].dl.a

void forward_test() {
  nn_t nn = new_nn();

  nn_add_dense_layer(&nn, 2, 2, RELU);
  nn_add_dense_layer(&nn, 2, 1, RELU);

  assert(nn.layers[1].dl.ws.rows == 2);
  assert(nn.layers[1].dl.ws.cols == 2);
  assert(nn.layers[1].dl.a.rows == 1);
  assert(nn.layers[1].dl.a.cols == 2);

  assert(nn.layers[2].dl.ws.rows == 2);
  assert(nn.layers[2].dl.ws.cols == 1);
  assert(nn.layers[2].dl.a.rows == 1);
  assert(nn.layers[2].dl.a.cols == 1);

  nn.layers[1].dl.bias = -3.0;
  nn.layers[2].dl.bias = 0.0;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = 0.0;
  MAT2D_GET(nn.layers[1].dl.ws, 0, 1) = 1.0;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 1.0;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 1) = 1.0;

  MAT2D_GET(nn.layers[2].dl.ws, 0, 0) = -1.0;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 0) = 1.0;

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

void backprop_test() {
  nn_t nn = new_nn();
  nn_add_dense_layer(&nn, 2, 2, SIGMOID);
  nn_add_dense_layer(&nn, 2, 2, SIGMOID);

  assert(nn.layer_count == 3); //input, dense, dense
  assert(nn.layers[1].dl.ws.rows == 2);
  assert(nn.layers[1].dl.ws.cols == 2);
  assert(nn.layers[1].dl.a.rows == 1);
  assert(nn.layers[1].dl.a.cols == 2);

  assert(nn.layers[2].dl.ws.rows == 2);
  assert(nn.layers[2].dl.ws.cols == 2);
  assert(nn.layers[2].dl.a.rows == 1);
  assert(nn.layers[2].dl.a.cols == 2);

  nn.layers[1].dl.bias = .35;
  nn.layers[2].dl.bias = .6;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = 0.15;
  MAT2D_GET(nn.layers[1].dl.ws, 0, 1) = 0.25;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 0.2;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 1) = 0.3;

  MAT2D_GET(nn.layers[2].dl.ws, 0, 0) = 0.4;
  MAT2D_GET(nn.layers[2].dl.ws, 0, 1) = 0.5;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 0) = 0.45;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 1) = 0.55;

  double i1[] = { .05, .1 };

  Mat2D input = (Mat2D) {
    .rows = 1,
    .cols = 2,
    .elems = i1,
  };

  nn_forward(&nn, &input);
  assert(nn.layers[0].il.input != NULL);
  assert(fabs(nn.layers[1].dl.a.elems[0] - 0.593269992) <= 5e-9 && fabs(nn.layers[1].dl.a.elems[1] - 0.596884378) <= 5e-9);
  Mat2D out = NN_OUTPUT(nn);
  assert(fabs(out.elems[0] - 0.75136507) <= 5e-9 && fabs(out.elems[1] - 0.772928465) <= 5e-9);

  double y1[] = { 0.01, 0.99 };
  Mat2D y = (Mat2D) {
    .cols = 2,
    .rows = 1,
    .elems = y1,
  };

  nn_t g = nn_backprop(&nn, &y);

  printf("backprop test: %.12f\n", g.layers[2].dl.ws.elems[0]);
  printf("backprop test: %.12f\n", g.layers[2].dl.ws.elems[1]);
  printf("backprop test: %.12f\n", g.layers[2].dl.ws.elems[2]);
  printf("backprop test: %.12f\n\n", g.layers[2].dl.ws.elems[3]);

  assert(fabs(g.layers[2].dl.ws.elems[0] - (0.082167041)) <= 5e-9);
  assert(fabs(g.layers[2].dl.ws.elems[1] - (-0.022602540)) <= 5e-9);
  assert(fabs(g.layers[2].dl.ws.elems[2] - (0.082667628)) <= 5e-9);
  assert(fabs(g.layers[2].dl.ws.elems[3] - (-0.022740242)) <= 5e-9);

  printf("backprop test: %.12f\n", g.layers[1].dl.ws.elems[0]);
  printf("backprop test: %.12f\n", g.layers[1].dl.ws.elems[1]);
  printf("backprop test: %.12f\n", g.layers[1].dl.ws.elems[2]);
  printf("backprop test: %.12f\n\n", g.layers[1].dl.ws.elems[3]);

  assert(fabs(g.layers[1].dl.ws.elems[0] - 0.000438568) <= 6e-9);
  assert(fabs(g.layers[1].dl.ws.elems[1] - 0.000497712) <= 6e-9);
  assert(fabs(g.layers[1].dl.ws.elems[2] - 0.000877139) <= 6e-9);
  assert(fabs(g.layers[1].dl.ws.elems[3] - 0.000995420) <= 6e-9);

  nn_destroy(&nn);
  nn_destroy(&g);
}

void fit_test() {
  nn_t nn = new_nn();
  nn_add_dense_layer(&nn, 2, 2, SIGMOID);
  nn_add_dense_layer(&nn, 2, 2, SIGMOID);

  nn.layers[1].dl.bias = .35;
  nn.layers[2].dl.bias = .6;

  MAT2D_GET(nn.layers[1].dl.ws, 0, 0) = 0.15;
  MAT2D_GET(nn.layers[1].dl.ws, 0, 1) = 0.25;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 0) = 0.2;
  MAT2D_GET(nn.layers[1].dl.ws, 1, 1) = 0.3;

  MAT2D_GET(nn.layers[2].dl.ws, 0, 0) = 0.4;
  MAT2D_GET(nn.layers[2].dl.ws, 0, 1) = 0.5;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 0) = 0.45;
  MAT2D_GET(nn.layers[2].dl.ws, 1, 1) = 0.55;

  double i1[] = { .05, .1 };
  Mat2D input = (Mat2D) {
    .rows = 1,
    .cols = 2,
    .elems = i1,
  };

  double labels[] = { .01, .99 };
  Mat2D y = (Mat2D) {
    .rows = 1,
    .cols = 2,
    .elems = labels,
  };

  nn_fit(&nn, &input, &y, 1, 0.5);

  assert(fabs(nn.layers[2].dl.ws.elems[0] - .358916480) <= 5e-9);
  assert(fabs(nn.layers[2].dl.ws.elems[1] - .511301270) <= 5e-9);
  assert(fabs(nn.layers[2].dl.ws.elems[2] - .408666186) <= 5e-9);
  assert(fabs(nn.layers[2].dl.ws.elems[3] - .561370121) <= 5e-9);

  assert(fabs(nn.layers[1].dl.ws.elems[0] - .149780716) <= 5e-9);
  assert(fabs(nn.layers[1].dl.ws.elems[1] - .24975114) <= 5e-9);
  assert(fabs(nn.layers[1].dl.ws.elems[2] - .19956143) <= 5e-9);
  assert(fabs(nn.layers[1].dl.ws.elems[3] - .29950229) <= 5e-9);

  nn_destroy(&nn);
}

int main(void) {
  test_t tests[] = {forward_test, backprop_test, fit_test};
  run_tests(tests, 3);
  return 0;
}
