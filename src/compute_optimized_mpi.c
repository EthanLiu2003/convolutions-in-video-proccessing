#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

matrix_t* allocate_output_matrix(int rows, int cols) {
    matrix_t* matrix = (matrix_t*)_mm_malloc(sizeof(matrix_t), 32);
    if (!matrix) return NULL;
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (int*)_mm_malloc(rows * cols * sizeof(int), 32);
    if (!matrix->data) {
        _mm_free(matrix);
        return NULL;
    }
    return matrix;
}

void flip_matrix(matrix_t* b_matrix, matrix_t* flipped_b) {
    #pragma omp parallel for
    for (int i = 0; i < b_matrix->rows; ++i) {
        for (int j = 0; j < b_matrix->cols; ++j) {
            flipped_b->data[(b_matrix->rows - i - 1) * b_matrix->cols + (b_matrix->cols - j - 1)] = b_matrix->data[i * b_matrix->cols + j];
        }
    }
}
void process_block(matrix_t *a_matrix, matrix_t *flipped_b, int *output_data, int start_row, int start_col, int end_row, int end_col, int output_cols) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = start_col; j < end_col; ++j) {
            int sum = 0;
            for (int k = 0; k < flipped_b->rows; ++k) {
                for (int l = 0; l < flipped_b->cols; ++l) {
                    // Ensure bounds are not exceeded
                    if ((i + k) < a_matrix->rows && (j + l) < a_matrix->cols) {
                        sum += a_matrix->data[(i + k) * a_matrix->cols + (j + l)] * flipped_b->data[k * flipped_b->cols + l];
                    }
                }
            }
            output_data[i * output_cols + j] = sum;
        }
    }
}
// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix
  matrix_t* flipped_b = allocate_output_matrix(b_matrix->rows, b_matrix->cols);
  if (!flipped_b) return -1;
  flip_matrix(b_matrix, flipped_b);
  int output_rows = a_matrix->rows - b_matrix->rows + 1;
  int output_cols = a_matrix->cols - b_matrix->cols + 1;
  *output_matrix = allocate_output_matrix(output_rows, output_cols);
  if (!*output_matrix) {
      free(flipped_b->data);
      free(flipped_b);
      return -1;
  }
  int block_size = 8;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < output_rows; i += block_size) {
        for (int j = 0; j < output_cols; j += block_size) {
            int current_end_row = (i + block_size > output_rows) ? output_rows : (i + block_size);
            int current_end_col = (j + block_size > output_cols) ? output_cols : (j + block_size);
            process_block(a_matrix, flipped_b, (*output_matrix)->data, i, j, current_end_row, current_end_col, output_cols);
        }
    }
  free(flipped_b->data);
  free(flipped_b);
  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
