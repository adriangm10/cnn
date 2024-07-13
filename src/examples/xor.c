#include "cnn.h"
#include "mat.h"
#include <stdio.h>

#define MAX_EPOCH 10000

int main(void) {
  nn_t xor_nn = new_nn();
  nn_add_dense_layer(&xor_nn, 2, 2, SIGMOID);
  nn_add_dense_layer(&xor_nn, 2, 1, SIGMOID);

  double table_input[] = {
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
  };

  double table_results[] = {
    0.0,
    1.0,
    1.0,
    0.0
  };

  Mat2D data = (Mat2D) {
    .cols = 2,
    .rows = 4,
    .elems = table_input
  };

  Mat2D labels = (Mat2D) {
    .cols = 1,
    .rows = 4,
    .elems = table_results,
  };


  for (int i = 0; i < MAX_EPOCH; ++i) {
    nn_fit(&xor_nn, &data, &labels, 1, 1.0);
  }

  nn_forward(&xor_nn, &((Mat2D) { 2, 1, table_input }));
  Mat2D o1 = nn_output(&xor_nn);
  puts("first input: {0, 0}, expected: 0, result:");
  print_Mat2D(&o1, "\n\n");

  nn_forward(&xor_nn, &((Mat2D) { 2, 1, &table_input[2] }));
  Mat2D o2 = nn_output(&xor_nn);
  puts("second input: {0, 1}, expected: 1, result:");
  print_Mat2D(&o2, "\n\n");

  nn_forward(&xor_nn, &((Mat2D) { 2, 1, &table_input[4] }));
  Mat2D o3 = nn_output(&xor_nn);
  puts("third input: {1, 0}, expected: 1, result:");
  print_Mat2D(&o3, "\n\n");

  nn_forward(&xor_nn, &((Mat2D) { 2, 1, &table_input[6] }));
  Mat2D o4 = nn_output(&xor_nn);
  puts("fourth input: {1, 1}, expected: 0, result:");
  print_Mat2D(&o4, "\n\n");

  nn_destroy(&xor_nn);
  return 0;
}
