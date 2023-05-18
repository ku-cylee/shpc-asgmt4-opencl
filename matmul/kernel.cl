#define TILE_SIZE   32

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int local_row = get_local_id(0);
  int local_col = get_local_id(1);
  int global_row = TILE_SIZE * get_group_id(0) + local_row;
  int global_col = TILE_SIZE * get_group_id(1) + local_col;

  __local float A_tile[TILE_SIZE][TILE_SIZE];
  __local float B_tile[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  for (int t = 0; t < K; t += TILE_SIZE) {
    A_tile[local_row][local_col] = A[global_row * K + t + local_col];
    B_tile[local_row][local_col] = B[(t + local_row) * N + global_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += A_tile[local_row][k] * B_tile[k][local_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[global_row * N + global_col] = sum;
}
