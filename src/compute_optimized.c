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

int dot_avx2(int32_t* vec1, int32_t* vec2, int n) {
    __m256i sum = _mm256_setzero_si256();

    for (int i = 0; i <= n - 8; i += 8) {
        __m256i a_vec = _mm256_loadu_si256((__m256i*)&vec1[i]);
        __m256i b_vec = _mm256_loadu_si256((__m256i*)&vec2[i]);
        __m256i prod = _mm256_mullo_epi32(a_vec, b_vec);
        sum = _mm256_add_epi32(sum, prod);
    }

    int buffer[8];
    _mm256_storeu_si256((__m256i*)buffer, sum);
    int total = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];

    // Handle the remaining elements
    for (int i = n / 8 * 8; i < n; ++i) {
        total += vec1[i] * vec2[i];
    }

    return total;
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
  int a_cols = a_matrix->cols, b_cols = b_matrix->cols;
  int *a_data = a_matrix->data, *flipped_data = flipped_b->data;
  int *output_data = (*output_matrix)->data;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            int sum = 0;
            if (b_cols >= 8) {
                for (int k = 0; k < b_matrix->rows; ++k) {
                    int l;
                    for (l = 0; l <= b_cols - 8; l += 8) {
                        __m256i a_vec = _mm256_loadu_si256((__m256i*)&a_data[(i + k) * a_cols + (j + l)]);
                        __m256i b_vec = _mm256_loadu_si256((__m256i*)&flipped_data[k * b_cols + l]);
                        __m256i product = _mm256_mullo_epi32(a_vec, b_vec);
                        __m256i temp1 = _mm256_hadd_epi32(product, product);
                        __m256i temp2 = _mm256_hadd_epi32(temp1, temp1);
                        int buffer[8];
                        _mm256_storeu_si256((__m256i*)buffer, temp2);
                        sum += buffer[0] + buffer[4];
                    }
                    for (; l < b_cols; ++l) {
                        sum += a_data[(i + k) * a_cols + (j + l)] * flipped_data[k * b_cols + l];
                    }
                }
            } else {
                for (int k = 0; k < b_matrix->rows; ++k) {
                    for (int l = 0; l < b_cols; ++l) {
                        sum += a_data[(i + k) * a_cols + (j + l)] * flipped_data[k * b_cols + l];
                    }
                }
            }
            output_data[i * output_cols + j] = sum;
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

