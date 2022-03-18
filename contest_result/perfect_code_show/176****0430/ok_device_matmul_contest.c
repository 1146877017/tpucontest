#include "okk.h"
#include "stdlib.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a)-1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE 524288
#define LOCAL_BANK_SIZE LOCAL_MEM_SIZE / okk_local_mem_bank_per_npu()
#define NEXT_BANK_ADDR(x) (DIV_UP((x), LOCAL_BANK_SIZE) * LOCAL_BANK_SIZE)
#define NPU_NUM 64
#define NO_USE 0
#define FLT_SIZE 4
// #define DEBUG 0
#ifndef DEBUG
#undef OKKERNEL_LOG
#define OKKERNEL_LOG(...)                                                      \
  {}
#endif
typedef struct {
  int left_rows, left_cols, right_cols;
  unsigned long long output_addr;
  unsigned long long left_addr;
  unsigned long long right_addr;
} __attribute__((packed)) param_t;

//
typedef struct {
  int left_row_step, left_col_step, right_col_step, output_row_step,
      output_col_step;
} split_info;

void adjust_split_info(split_info *info) { // may make a better adjust
  // OKKERNEL_ASSERT(0 >1);
  if (info->left_row_step ==
      MAX(info->left_col_step,
          MAX(info->left_row_step, info->right_col_step))) {
    info->left_row_step = MAX(1, info->left_row_step / 2);
  } else if (info->right_col_step ==
             MAX(info->left_col_step,
                 MAX(info->left_row_step, info->right_col_step))) {
    info->right_col_step = MAX(1, info->right_col_step / 2);
  } else { // may be worst choice
    info->left_col_step = MAX(1, info->left_col_step / 2);
  }
  return;
}
void matmul_pingpong_left_origin(const param_t *param) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(2048, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};
  int USE_NPU_NUM = 32;
  if (param->left_rows == 4 || param->left_rows == 256 ||
      param->left_rows == 300 || param->left_rows == 1024 ||
      param->left_cols <= 4 || param->right_cols == 1)
    USE_NPU_NUM = 64;
  int left_cols_per_channel = DIV_UP(info.left_col_step, USE_NPU_NUM),
      right_cols_per_channel = DIV_UP(info.right_col_step, USE_NPU_NUM);

  if (left_cols_per_channel > 128)
    left_cols_per_channel = 128;
  if (right_cols_per_channel > 128)
    right_cols_per_channel = 128;
  local_addr_t output_addr = 0, left_addr[2], right_addr[2];
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] = NEXT_BANK_ADDR(output_shape.n * output_stride.n * FLT_SIZE);
  // right_addr[1] =
  //     NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n *
  //     FLT_SIZE);

  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  // cut left row step
  info.left_row_step *= 2;
  do {
    info.left_row_step /= 2;
    // Local left matrix tensor.
    left_addr[0] = NEXT_BANK_ADDR(right_addr[0] +
                                  right_shape.n * right_stride.n * FLT_SIZE);
    left_shape.n = info.left_row_step;
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    left_addr[1] =
        NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  } while (left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE >
           LOCAL_MEM_SIZE);

  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr, right_addr[0], right_addr[0], left_addr[0],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  // end cal best split
  // batch of right cols
  bool is_parallel = false;
  for (int right_col_idx = 0; right_col_idx < param->right_cols;
       right_col_idx += info.right_col_step) {
    // deal with last irregular block
    int this_right_block_col =
        (right_col_idx + info.right_col_step) <= param->right_cols
            ? info.right_col_step
            : param->right_cols % info.right_col_step;
    for (int left_col_idx = 0; left_col_idx < param->left_cols;
         left_col_idx += info.left_col_step) {
      // deal with last irregular block
      int this_left_block_col =
          (left_col_idx + info.left_col_step) <= param->left_cols
              ? info.left_col_step
              : param->left_cols % info.left_col_step;
      //  S2L right sub-matrix-block and pin it
      //  S2L 2 blocks may be better
      okk_gdma_32bit_matrix_S2L(
          right_addr[0],
          param->right_addr +
              (left_col_idx * param->right_cols + right_col_idx) * FLT_SIZE,
          this_left_block_col, this_right_block_col, right_cols_per_channel,
          param->right_cols);
      OKKERNEL_LOG("Copy right");
      for (int left_row_idx = 0; left_row_idx < param->left_rows;
           left_row_idx += info.left_row_step) {
        int this_left_block_row =
            (left_row_idx + info.left_row_step) <= param->left_rows
                ? info.left_row_step
                : param->left_rows % info.left_row_step;
        OKKERNEL_LOG("\nThis shape is %d %d %d\nFrom %d %d %d ",
                     this_left_block_row, this_left_block_col,
                     this_right_block_col, left_row_idx, left_col_idx,
                     right_col_idx);

        // change left sub-matrix-block
        okk_gdma_32bit_matrix_S2L(
            left_addr[left_row_idx % info.left_row_step],
            param->left_addr +
                (left_row_idx * param->left_cols + left_col_idx) * FLT_SIZE,
            this_left_block_row, this_left_block_col, left_cols_per_channel,
            param->left_cols);
        if (is_parallel)
          okk_parallel_end();
        okk_parallel_start();
        is_parallel = true;
        // do matmul
        okk_bdc_matmul(output_addr + left_row_idx * output_stride.n * FLT_SIZE,
                       left_addr[left_row_idx % info.left_row_step],
                       right_addr[0], NO_USE, this_left_block_row,
                       this_left_block_col, this_right_block_col,
                       left_cols_per_channel, right_cols_per_channel, false,
                       left_col_idx == 0 ? false : true);
        OKKERNEL_LOG("\nWrite to %d",
                     left_row_idx * output_stride.n * FLT_SIZE);
      }
      okk_parallel_end();
      is_parallel = false;
    }
    // Move out
    okk_gdma_32bit_matrix_L2S(param->output_addr + right_col_idx * FLT_SIZE,
                              output_addr, param->left_rows,
                              this_right_block_col, right_cols_per_channel,
                              param->right_cols);
  }
}

// for precision problem, use a silly solution
void matmul_precision(const param_t *param) {
  // OKKERNEL_LOG("\nmatmul_precision");
  split_info s_info = {MIN(2048, param->left_rows), MIN(1, param->left_cols),
                       MIN(4096, param->right_cols), 0, 0};
  //
  global_addr_t left_block_addr = 0, right_block_addr = 0,
                output_block_addr = 0;
  int block_left_row_step = 0, block_left_col_step = 0,
      block_right_col_step = 0, pre_block_left_row_step = 0,
      pre_block_left_col_step = 0, pre_block_right_col_step = 0;
  dim4 output_stride, left_stride, right_stride;
  int left_cols_per_channel, right_cols_per_channel;
  local_addr_t left_addr[2], right_addr[2], output_addr;

  left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
  right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);

  dim4 left_shape = {.n = s_info.left_row_step,
                     .c = DIV_UP(s_info.left_col_step, left_cols_per_channel),
                     .h = 1,
                     .w = left_cols_per_channel};
  dim4 right_shape = {.n = s_info.left_col_step,
                      .c =
                          DIV_UP(s_info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  dim4 output_shape = {
      .n = s_info.left_row_step,
      .c = DIV_UP(s_info.right_col_step, right_cols_per_channel),
      .h = 1,
      .w = right_cols_per_channel};
  while (1) {
    // Local left matrix tensor.
    left_addr[0] = 0;
    left_shape.n = s_info.left_row_step;
    left_shape.c = DIV_UP(s_info.left_col_step, left_cols_per_channel);

    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    right_addr[0] = left_addr[0] + left_stride.n * left_shape.n * sizeof(float);
    right_shape.n = s_info.left_col_step;
    right_shape.c = DIV_UP(s_info.right_col_step, right_cols_per_channel);
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    output_addr =
        right_addr[0] + right_stride.n * right_shape.n * sizeof(float);

    output_shape.n = s_info.left_row_step;
    output_shape.c = DIV_UP(s_info.right_col_step, right_cols_per_channel);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // Safe checking.
    if (output_addr + output_stride.n * output_shape.n * sizeof(float) <=
        LOCAL_MEM_SIZE / 2) {
      break;
    } else {
      adjust_split_info(&s_info);
      continue;
    }
  }
  //
  left_addr[1] = left_addr[0] + LOCAL_MEM_SIZE / 2;
  right_addr[1] = right_addr[0] + LOCAL_MEM_SIZE / 2;
  //
  OKKERNEL_LOG("\n%d %d %d\n%d %d %d", param->left_rows, param->left_cols,
               param->right_cols, s_info.left_row_step, s_info.left_col_step,
               s_info.right_col_step);
  OKKERNEL_LOG("\nMax addr is: %d / %d\n",
               output_addr +
                   output_stride.n * s_info.left_row_step * sizeof(float),
               LOCAL_MEM_SIZE);
  for (int left_row_idx = 0; left_row_idx < param->left_rows;
       left_row_idx += s_info.left_row_step) {
    block_left_row_step = left_row_idx + s_info.left_row_step > param->left_rows
                              ? param->left_rows - left_row_idx
                              : s_info.left_row_step;
    for (int right_col_idx = 0; right_col_idx < param->right_cols;
         right_col_idx += s_info.right_col_step) {
      block_right_col_step =
          right_col_idx + s_info.right_col_step > param->right_cols
              ? param->right_cols - right_col_idx
              : s_info.right_col_step;
      output_block_addr = param->output_addr +
                          left_row_idx * param->right_cols * FLT_SIZE +
                          right_col_idx * FLT_SIZE;
      for (int left_col_idx = 0; left_col_idx < param->left_cols;
           left_col_idx += s_info.left_col_step) {
        // okk_parallel_start();
        block_left_col_step =
            left_col_idx + s_info.left_col_step > param->left_cols
                ? param->left_cols - left_col_idx
                : s_info.left_col_step;
        left_block_addr = param->left_addr +
                          left_row_idx * param->left_cols * FLT_SIZE +
                          left_col_idx * FLT_SIZE;
        right_block_addr = param->right_addr +
                           left_col_idx * param->right_cols * FLT_SIZE +
                           right_col_idx * FLT_SIZE;
        // Copy global left matrix tensor to local left matrix tensor.
        okk_gdma_32bit_matrix_S2L(left_addr[0], left_block_addr,
                                  block_left_row_step, block_left_col_step,
                                  left_cols_per_channel, param->left_cols);
        // Copy global right matrix tensor to local right matrix tensor.
        okk_gdma_32bit_matrix_S2L(right_addr[0], right_block_addr,
                                  block_left_col_step, block_right_col_step,
                                  right_cols_per_channel, param->right_cols);
        okk_bdc_matmul(output_addr, left_addr[0], right_addr[0], NO_USE,
                       block_left_row_step, block_left_col_step,
                       block_right_col_step, left_cols_per_channel,
                       right_cols_per_channel, false,
                       left_col_idx == 0 ? false : true);

        // okk_parallel_end();
      }
      okk_gdma_32bit_matrix_L2S(output_block_addr, output_addr,
                                block_left_row_step, block_right_col_step,
                                right_cols_per_channel, param->right_cols);
    }
  }
}

void matmul_pingpong_lr_rc_lc_2l_2r_1o(const param_t *param) {
  OKKERNEL_LOG("\nmatmul_pingpong_lr_rc_lc_2l_2r_1o");
  // pingpong with loop of lr-rc-lc, has 2 leftaddr, 2 rightaddr and 1 output
  // -> change the loop order may bring block reuse but may also bring mem use,
  // try later if want use 1/2 local, set 128/256/512, use full local, set
  // 256/512/1024
  split_info s_info = {MIN(1024, param->left_rows), MIN(1024, param->left_cols),
                       MIN(1024, param->right_cols), 0, 0};
  //
  global_addr_t left_block_addr = 0, right_block_addr = 0,
                output_block_addr = 0;
  int block_left_row_step = 0, block_left_col_step = 0,
      block_right_col_step = 0, pre_block_left_row_step = 0,
      pre_block_left_col_step = 0, pre_block_right_col_step = 0;
  dim4 output_stride, left_stride, right_stride;
  int left_cols_per_channel, right_cols_per_channel;
  local_addr_t left_addr[2], right_addr[2], output_addr;

  left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
  right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);

  dim4 left_shape = {.n = s_info.left_row_step,
                     .c = DIV_UP(s_info.left_col_step, left_cols_per_channel),
                     .h = 1,
                     .w = left_cols_per_channel};
  dim4 right_shape = {.n = s_info.left_col_step,
                      .c =
                          DIV_UP(s_info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  dim4 output_shape = {
      .n = s_info.left_row_step,
      .c = DIV_UP(s_info.right_col_step, right_cols_per_channel),
      .h = 1,
      .w = right_cols_per_channel};
  while (1) {
    // Local left matrix tensor.
    left_addr[0] = 0;
    left_shape.n = s_info.left_row_step;
    left_shape.c = DIV_UP(s_info.left_col_step, left_cols_per_channel);

    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    right_addr[0] = left_addr[0] + left_stride.n * left_shape.n * sizeof(float);
    right_shape.n = s_info.left_col_step;
    right_shape.c = DIV_UP(s_info.right_col_step, right_cols_per_channel);
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    output_addr =
        right_addr[0] + right_stride.n * right_shape.n * sizeof(float);

    output_shape.n = s_info.left_row_step;
    output_shape.c = DIV_UP(s_info.right_col_step, right_cols_per_channel);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // Safe checking.
    if (output_addr + output_stride.n * output_shape.n * sizeof(float) <=
        LOCAL_MEM_SIZE / 2) {
      break;
    } else {
      adjust_split_info(&s_info);
      continue;
    }
  }
  //
  left_addr[1] = left_addr[0] + LOCAL_MEM_SIZE / 2;
  right_addr[1] = right_addr[0] + LOCAL_MEM_SIZE / 2;
  //
  // OKKERNEL_LOG("\n%d %d %d\n%d %d %d", param->left_rows, param->left_cols,
  //              param->right_cols, s_info.left_row_step, s_info.left_col_step,
  //              s_info.right_col_step);
  // OKKERNEL_LOG("\nMax addr is: %d / %d %d\n",
  //              output_addr +
  //                  output_stride.n * s_info.left_row_step * sizeof(float),
  //              LOCAL_MEM_SIZE,output_addr + param->left_rows *
  //              param->right_cols * FLT_SIZE);
  for (int left_row_idx = 0; left_row_idx < param->left_rows;
       left_row_idx += s_info.left_row_step) {
    block_left_row_step = left_row_idx + s_info.left_row_step > param->left_rows
                              ? param->left_rows - left_row_idx
                              : s_info.left_row_step;
    for (int right_col_idx = 0; right_col_idx < param->right_cols;
         right_col_idx += s_info.right_col_step) {
      block_right_col_step =
          right_col_idx + s_info.right_col_step > param->right_cols
              ? param->right_cols - right_col_idx
              : s_info.right_col_step;
      output_block_addr = param->output_addr +
                          left_row_idx * param->right_cols * FLT_SIZE +
                          right_col_idx * FLT_SIZE;
      for (int left_col_idx = 0;
           left_col_idx < param->left_cols + s_info.left_col_step;
           left_col_idx += s_info.left_col_step) {
        okk_parallel_start();
        if (left_col_idx < param->left_cols) { // i<S
          block_left_col_step =
              left_col_idx + s_info.left_col_step > param->left_cols
                  ? param->left_cols - left_col_idx
                  : s_info.left_col_step;
          left_block_addr = param->left_addr +
                            left_row_idx * param->left_cols * FLT_SIZE +
                            left_col_idx * FLT_SIZE;
          right_block_addr = param->right_addr +
                             left_col_idx * param->right_cols * FLT_SIZE +
                             right_col_idx * FLT_SIZE;
          // Copy global left matrix tensor to local left matrix tensor.
          okk_gdma_32bit_matrix_S2L(
              left_addr[left_col_idx / s_info.left_col_step % 2],
              left_block_addr, block_left_row_step, block_left_col_step,
              left_cols_per_channel, param->left_cols);
          // Copy global right matrix tensor to local right matrix tensor.
          okk_gdma_32bit_matrix_S2L(
              right_addr[left_col_idx / s_info.left_col_step % 2],
              right_block_addr, block_left_col_step, block_right_col_step,
              right_cols_per_channel, param->right_cols);
          pre_block_left_col_step = block_left_col_step;
          pre_block_right_col_step = block_right_col_step;
          pre_block_left_row_step = block_left_row_step;
        }
        if (left_col_idx > 0 &&
            left_col_idx < param->left_cols + s_info.left_col_step) {
          // Matrix multiplication.
          okk_bdc_matmul(
              output_addr,
              left_addr[(left_col_idx / s_info.left_col_step + 1) % 2],
              right_addr[(left_col_idx / s_info.left_col_step + 1) % 2], NO_USE,
              pre_block_left_row_step, pre_block_left_col_step,
              pre_block_right_col_step, left_cols_per_channel,
              right_cols_per_channel, false,
              left_col_idx == s_info.left_col_step ? false : true);
        }
        okk_parallel_end();
      }
      // Copy local output matrix tensor to global
      okk_gdma_32bit_matrix_L2S(output_block_addr, output_addr,
                                block_left_row_step, block_right_col_step,
                                right_cols_per_channel, param->right_cols);
    }
  }
}

void matmul_naive(const param_t *param) {
  OKKERNEL_LOG("\nmatmul_naive");
  // if want use 1/2 local, set 128/256/512, use full
  // local, set 256/512/1024
  split_info s_info = {MIN(1024, param->left_rows), MIN(1024, param->left_cols),
                       MIN(1024, param->right_cols), 0, 0};
  //
  global_addr_t left_block_addr = 0, right_block_addr = 0,
                output_block_addr = 0;
  int block_left_row_step = 0, block_left_col_step = 0,
      block_right_col_step = 0;
  dim4 output_stride, left_stride, right_stride;
  int left_cols_per_channel, right_cols_per_channel;
  local_addr_t left_addr, right_addr, output_addr;

  left_cols_per_channel = DIV_UP(s_info.left_col_step, NPU_NUM);
  if (left_cols_per_channel > 128)
    left_cols_per_channel = 128;
  right_cols_per_channel = DIV_UP(s_info.right_col_step, NPU_NUM);
  if (right_cols_per_channel > 128)
    right_cols_per_channel = 128;
  while (1) {
    // Local left matrix tensor.
    left_addr = 0;
    dim4 left_shape = {.n = s_info.left_row_step,
                       .c = DIV_UP(s_info.left_col_step, left_cols_per_channel),
                       .h = 1,
                       .w = left_cols_per_channel};
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    right_addr = left_addr + left_stride.n * left_shape.n * sizeof(float);
    dim4 right_shape = {
        .n = s_info.left_col_step,
        .c = DIV_UP(s_info.right_col_step, right_cols_per_channel),
        .h = 1,
        .w = right_cols_per_channel};
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    output_addr = right_addr + right_stride.n * right_shape.n * sizeof(float);
    dim4 output_shape = {
        .n = s_info.left_row_step,
        .c = DIV_UP(s_info.right_col_step, right_cols_per_channel),
        .h = 1,
        .w = right_cols_per_channel};
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // Safe checking.
    if (output_addr + output_stride.n * output_shape.n * sizeof(float) <=
        LOCAL_MEM_SIZE) {
      break;
    } else {
      adjust_split_info(&s_info);
      continue;
    }
  }
  // OKKERNEL_LOG("\n%d %d %d\n%d %d %d", param->left_rows, param->left_cols,
  //              param->right_cols, s_info.left_row_step, s_info.left_col_step,
  //              s_info.right_col_step);
  // OKKERNEL_LOG("\nMax addr is: %d / %d\n",
  //              output_addr +
  //                  output_stride.n * s_info.left_row_step * sizeof(float),
  //              LOCAL_MEM_SIZE);
  int is_s2l = false;
  for (int left_row_idx = 0; left_row_idx < param->left_rows;
       left_row_idx += s_info.left_row_step) {
    block_left_row_step = left_row_idx + s_info.left_row_step > param->left_rows
                              ? param->left_rows - left_row_idx
                              : s_info.left_row_step;
    for (int right_col_idx = 0; right_col_idx < param->right_cols;
         right_col_idx += s_info.right_col_step) {
      block_right_col_step =
          right_col_idx + s_info.right_col_step > param->right_cols
              ? param->right_cols - right_col_idx
              : s_info.right_col_step;
      output_block_addr = param->output_addr +
                          left_row_idx * param->right_cols * FLT_SIZE +
                          right_col_idx * FLT_SIZE;
      for (int left_col_idx = 0; left_col_idx < param->left_cols;
           left_col_idx += s_info.left_col_step) {
        block_left_col_step =
            left_col_idx + s_info.left_col_step > param->left_cols
                ? param->left_cols - left_col_idx
                : s_info.left_col_step;
        // OKKERNEL_LOG("\nSteps: %d %d %d ", block_left_row_step,
        //              block_left_col_step, block_right_col_step);
        left_block_addr = param->left_addr +
                          left_row_idx * param->left_cols * FLT_SIZE +
                          left_col_idx * FLT_SIZE;
        right_block_addr = param->right_addr +
                           left_col_idx * param->right_cols * FLT_SIZE +
                           right_col_idx * FLT_SIZE;
        // Copy global left matrix tensor to local left matrix tensor.
        // OKKERNEL_LOG("Copy left S2L");
        okk_gdma_32bit_matrix_S2L(left_addr, left_block_addr,
                                  block_left_row_step, block_left_col_step,
                                  left_cols_per_channel, param->left_cols);
        // Copy global right matrix tensor to local right matrix tensor.
        if (s_info.right_col_step == param->right_cols &&
            s_info.left_col_step == param->left_cols) {
          if (!is_s2l) {
            // OKKERNEL_LOG("Copy right S2L0");
            okk_gdma_32bit_matrix_S2L(right_addr, right_block_addr,
                                      block_left_col_step, block_right_col_step,
                                      right_cols_per_channel,
                                      param->right_cols);
            is_s2l = true;
          }
        } else {
          // OKKERNEL_LOG("Copy right S2L1");
          okk_gdma_32bit_matrix_S2L(right_addr, right_block_addr,
                                    block_left_col_step, block_right_col_step,
                                    right_cols_per_channel, param->right_cols);
        }
        // Matrix multiplication.
        okk_bdc_matmul(
            output_addr, left_addr, right_addr, NO_USE, block_left_row_step,
            block_left_col_step, block_right_col_step, left_cols_per_channel,
            right_cols_per_channel, false, left_col_idx == 0 ? false : true);
      }
      // Copy local output matrix tensor to global
      okk_gdma_32bit_matrix_L2S(output_block_addr, output_addr,
                                block_left_row_step, block_right_col_step,
                                right_cols_per_channel, param->right_cols);
    }
  }
}

void matmul_pingpong_left_case_6(const param_t *param, int use_npu_num_left,
                                 int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(2048, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  local_addr_t output_addr[2], left_addr, right_addr[2];
  output_addr[0] = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  output_addr[1] = output_shape.n * output_stride.n * FLT_SIZE;
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] = NEXT_BANK_ADDR(output_addr[1] +
                                 output_shape.n * output_stride.n * FLT_SIZE);
  right_addr[1] =
      NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr =
      NEXT_BANK_ADDR(right_addr[1] + right_shape.n * right_stride.n * FLT_SIZE);
  left_shape.n = info.left_row_step;
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);

  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr[1], right_addr[0], right_addr[1], left_addr,
               left_addr, left_addr + left_shape.n * left_stride.n * FLT_SIZE);

  okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr, info.left_row_step,
                            info.left_col_step, left_cols_per_channel,
                            param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[0], param->right_addr,
                            info.left_col_step, info.right_col_step,
                            right_cols_per_channel, param->right_cols);
  okk_parallel_start();

  okk_bdc_matmul(output_addr[0], left_addr, right_addr[0], NO_USE,
                 info.left_row_step, info.left_col_step, 2048,
                 left_cols_per_channel, right_cols_per_channel, false, false);

  okk_gdma_32bit_matrix_S2L(right_addr[1], param->right_addr + 2048 * FLT_SIZE,
                            info.left_col_step, 2042, right_cols_per_channel,
                            param->right_cols);
  okk_parallel_end();
  okk_parallel_start();
  okk_bdc_matmul(output_addr[1], left_addr, right_addr[1], NO_USE,
                 info.left_row_step, info.left_col_step, 2042,
                 left_cols_per_channel, right_cols_per_channel, false, false);

  okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr[0],
                            param->left_rows, 2048, right_cols_per_channel,
                            param->right_cols);
  okk_parallel_end();
  okk_gdma_32bit_matrix_L2S(param->output_addr + 2048 * FLT_SIZE,
                            output_addr[1], param->left_rows, 2042,
                            right_cols_per_channel, param->right_cols);
}
void matmul_pingpong_left_case_7(const param_t *param, int use_npu_num_left,
                                 int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(1024, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  local_addr_t output_addr, left_addr[2], right_addr[2];
  output_addr = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] =
      NEXT_BANK_ADDR(output_addr + output_shape.n * output_stride.n * FLT_SIZE);
  right_addr[1] =
      NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr[0] =
      NEXT_BANK_ADDR(right_addr[1] + right_shape.n * right_stride.n * FLT_SIZE);
  left_shape.n = info.left_row_step;
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
  left_addr[1] =
      NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr, right_addr[0], right_addr[1], left_addr[0],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  // 0
  okk_gdma_32bit_matrix_S2L(left_addr[0], param->left_addr, 200, 1024,
                            left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[0], param->right_addr, 1024, 324,
                            right_cols_per_channel, param->right_cols);

  okk_parallel_start();
  okk_bdc_matmul(output_addr, left_addr[0], right_addr[0], NO_USE, 200, 1024,
                 324, left_cols_per_channel, right_cols_per_channel, false,
                 false);

  // 1
  okk_gdma_32bit_matrix_S2L(left_addr[1], param->left_addr + 1024 * FLT_SIZE,
                            200, 1024, left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[1],
                            param->right_addr + 1024 * 324 * FLT_SIZE, 1024,
                            324, right_cols_per_channel, param->right_cols);
  okk_parallel_end();

  okk_parallel_start();
  okk_bdc_matmul(output_addr, left_addr[1], right_addr[1], NO_USE, 200, 1024,
                 324, left_cols_per_channel, right_cols_per_channel, false,
                 true);

  // `2
  okk_gdma_32bit_matrix_S2L(left_addr[0],
                            param->left_addr + 2 * 1024 * FLT_SIZE, 200, 1024,
                            left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[0],
                            param->right_addr + 2 * 1024 * 324 * FLT_SIZE, 1024,
                            324, right_cols_per_channel, param->right_cols);
  okk_parallel_end();

  okk_parallel_start();
  okk_bdc_matmul(output_addr, left_addr[0], right_addr[0], NO_USE, 200, 1024,
                 324, left_cols_per_channel, right_cols_per_channel, false,
                 true);

  // `3
  okk_gdma_32bit_matrix_S2L(left_addr[1],
                            param->left_addr + 3 * 1024 * FLT_SIZE, 200, 1024,
                            left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[1],
                            param->right_addr + 3 * 1024 * 324 * FLT_SIZE, 1024,
                            324, right_cols_per_channel, param->right_cols);
  okk_parallel_end();

  okk_bdc_matmul(output_addr, left_addr[1], right_addr[1], NO_USE, 200, 1024,
                 324, left_cols_per_channel, right_cols_per_channel, false,
                 true);

  okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, 200, 324,
                            right_cols_per_channel, 324);
}
void matmul_pingpong_left_case_8(const param_t *param, int use_npu_num_left,
                                 int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(1024, param->left_cols),
                     MIN(1024, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  local_addr_t output_addr[2], left_addr[2], right_addr[2];
  output_addr[0] = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  output_addr[1] = NEXT_BANK_ADDR(output_addr[0] +
                                  output_shape.n * output_stride.n * FLT_SIZE);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] = NEXT_BANK_ADDR(output_addr[1] +
                                 output_shape.n * output_stride.n * FLT_SIZE);
  right_addr[1] =
      NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr[0] =
      NEXT_BANK_ADDR(right_addr[1] + right_shape.n * right_stride.n * FLT_SIZE);
  left_shape.n = info.left_row_step;
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
  left_addr[1] =
      NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr[0], right_addr[0], right_addr[1], left_addr[0],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  okk_gdma_32bit_matrix_S2L(left_addr[0], param->left_addr, 256, 768,
                            left_cols_per_channel, param->left_cols);
  // 0
  okk_gdma_32bit_matrix_S2L(right_addr[0], param->right_addr, 768, 1024,
                            right_cols_per_channel, param->right_cols);
  okk_parallel_start();
  okk_bdc_matmul(output_addr[0], left_addr[0], right_addr[0], NO_USE, 256, 768,
                 1024, left_cols_per_channel, right_cols_per_channel, false,
                 false);

  // 1
  okk_gdma_32bit_matrix_S2L(right_addr[1], param->right_addr + 1024 * FLT_SIZE,
                            768, 1024, right_cols_per_channel,
                            param->right_cols);
  okk_parallel_end();
  okk_parallel_start();

  okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr[0], 256, 1024,
                            right_cols_per_channel, 3072);
  okk_bdc_matmul(output_addr[1], left_addr[0], right_addr[1], NO_USE, 256, 768,
                 1024, left_cols_per_channel, right_cols_per_channel, false,
                 false);

  // 2
  okk_gdma_32bit_matrix_S2L(right_addr[0],
                            param->right_addr + 2 * 1024 * FLT_SIZE, 768, 1024,
                            right_cols_per_channel, param->right_cols);
  okk_parallel_end();
  okk_parallel_start();

  okk_gdma_32bit_matrix_L2S(param->output_addr + 1024 * FLT_SIZE,
                            output_addr[1], 256, 1024, right_cols_per_channel,
                            3072);
  okk_bdc_matmul(output_addr[0], left_addr[0], right_addr[0], NO_USE, 256, 768,
                 1024, left_cols_per_channel, right_cols_per_channel, false,
                 false);
  okk_parallel_end();
  okk_gdma_32bit_matrix_L2S(param->output_addr + 2 * 1024 * FLT_SIZE,
                            output_addr[0], 256, 1024, right_cols_per_channel,
                            3072);
}

void matmul_pingpong_left_case_9(const param_t *param, int use_npu_num_left,
                                 int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(1024, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  local_addr_t output_addr, left_addr[2], right_addr[2];
  output_addr = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] =
      NEXT_BANK_ADDR(output_addr + output_shape.n * output_stride.n * FLT_SIZE);
  right_addr[1] =
      NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr[0] =
      NEXT_BANK_ADDR(right_addr[1] + right_shape.n * right_stride.n * FLT_SIZE);
  left_shape.n = info.left_row_step;
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
  left_addr[1] =
      NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr, right_addr[0], right_addr[1], left_addr[1],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);

  // 0
  okk_gdma_32bit_matrix_S2L(left_addr[0], param->left_addr, param->left_rows,
                            1024, left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[0], param->right_addr, 1024,
                            param->right_cols, right_cols_per_channel,
                            param->right_cols);
  okk_parallel_start();
  okk_bdc_matmul(output_addr, left_addr[0], right_addr[0], NO_USE,
                 param->left_rows, 1024, param->right_cols,
                 left_cols_per_channel, right_cols_per_channel, false, false);
  // 1
  okk_gdma_32bit_matrix_S2L(left_addr[1], param->left_addr + 1024 * FLT_SIZE,
                            param->left_rows, 1024, left_cols_per_channel,
                            param->left_cols);

  okk_gdma_32bit_matrix_S2L(
      right_addr[1], param->right_addr + param->right_cols * 1024 * FLT_SIZE,
      1024, 768, right_cols_per_channel, param->right_cols);
  okk_parallel_end();
  okk_parallel_start();
  okk_bdc_matmul(output_addr, left_addr[1], right_addr[1], NO_USE,
                 param->left_rows, 1024, param->right_cols,
                 left_cols_per_channel, right_cols_per_channel, false, true);
  // 2
  okk_gdma_32bit_matrix_S2L(
      left_addr[0], param->left_addr + 2 * 1024 * FLT_SIZE, param->left_rows,
      1024, left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(
      right_addr[0],
      param->right_addr + 2 * param->right_cols * 1024 * FLT_SIZE, 1024,
      param->right_cols, right_cols_per_channel, param->right_cols);
  okk_parallel_end();
  okk_bdc_matmul(output_addr, left_addr[0], right_addr[0], NO_USE,
                 param->left_rows, 1024, param->right_cols,
                 left_cols_per_channel, right_cols_per_channel, false, true);

  okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows,
                            param->right_cols, right_cols_per_channel,
                            param->right_cols);
}
void matmul_pingpong_left_case_10(const param_t *param, int use_npu_num_left,
                                  int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(1024, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  local_addr_t output_addr, left_addr[2], right_addr[2];
  output_addr = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] =
      NEXT_BANK_ADDR(output_addr + output_shape.n * output_stride.n * FLT_SIZE);
  right_addr[1] =
      NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr[0] =
      NEXT_BANK_ADDR(right_addr[1] + right_shape.n * right_stride.n * FLT_SIZE);
  left_shape.n = info.left_row_step;
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
  left_addr[1] =
      NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr, right_addr[0], right_addr[1], left_addr[1],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);

  // 0
  okk_gdma_32bit_matrix_S2L(left_addr[0], param->left_addr, param->left_rows,
                            1024, left_cols_per_channel, param->left_cols);

  okk_gdma_32bit_matrix_S2L(right_addr[0], param->right_addr, 1024,
                            param->right_cols, right_cols_per_channel,
                            param->right_cols);
  okk_parallel_start();
  okk_bdc_matmul(output_addr, left_addr[0], right_addr[0], NO_USE,
                 param->left_rows, 1024, param->right_cols,
                 left_cols_per_channel, right_cols_per_channel, false, false);
  // 1
  okk_gdma_32bit_matrix_S2L(left_addr[1], param->left_addr + 1024 * FLT_SIZE,
                            param->left_rows, 1024, left_cols_per_channel,
                            param->left_cols);

  okk_gdma_32bit_matrix_S2L(
      right_addr[1], param->right_addr + param->right_cols * 1024 * FLT_SIZE,
      1024, param->right_cols, right_cols_per_channel, param->right_cols);
  okk_parallel_end();
  okk_bdc_matmul(output_addr, left_addr[1], right_addr[1], NO_USE,
                 param->left_rows, 1024, param->right_cols,
                 left_cols_per_channel, right_cols_per_channel, false, true);

  okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows,
                            param->right_cols, right_cols_per_channel,
                            param->right_cols);
}
void matmul_pingpong_left_case_11(const param_t *param, int use_npu_num_left,
                                  int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(512, param->left_rows), MIN(2048, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  local_addr_t output_addr[2], left_addr[2], right_addr;
  output_addr[0] = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  output_addr[1] = output_shape.n * output_stride.n * FLT_SIZE;
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr = NEXT_BANK_ADDR(output_addr[1] +
                              output_shape.n * output_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr[0] =
      NEXT_BANK_ADDR(right_addr + right_shape.n * right_stride.n * FLT_SIZE);
  left_shape.n = info.left_row_step;
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
  left_addr[1] =
      NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr[1], right_addr, right_addr, left_addr[0],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, info.left_col_step,
                            info.right_col_step, right_cols_per_channel,
                            param->right_cols);

  okk_gdma_32bit_matrix_S2L(left_addr[0], param->left_addr, 512,
                            info.left_col_step, left_cols_per_channel,
                            param->left_cols);
  okk_parallel_start();
  okk_bdc_matmul(output_addr[0], left_addr[0], right_addr, NO_USE, 512,
                 info.left_col_step, 1024, left_cols_per_channel,
                 right_cols_per_channel, false, false);

  okk_gdma_32bit_matrix_S2L(
      left_addr[1], param->left_addr + 512 * 1024 * FLT_SIZE, 512,
      info.left_col_step, left_cols_per_channel, param->left_cols);
  okk_parallel_end();
  okk_parallel_start();
  okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr[0], 512, 1024,
                            right_cols_per_channel, param->right_cols);

  okk_bdc_matmul(output_addr[1], left_addr[1], right_addr, NO_USE, 512,
                 info.left_col_step, 1024, left_cols_per_channel,
                 right_cols_per_channel, false, false);
  okk_parallel_end();
  okk_gdma_32bit_matrix_L2S(param->output_addr + 1024 * 512 * FLT_SIZE,
                            output_addr[1], 512, 1024, right_cols_per_channel,
                            param->right_cols);
}
void matmul_pingpong_left_case_13_14(const param_t *param, int use_npu_num_left,
                                     int use_npu_num_right, int left_row_step) {

  int left_cols_per_channel = DIV_UP(param->left_cols, use_npu_num_left),
      right_cols_per_channel = DIV_UP(param->right_cols, use_npu_num_right);

  local_addr_t output_addr[2], left_addr[2], right_addr;
  output_addr[0] = 0;
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = left_row_step,
                       .c = DIV_UP(param->right_cols, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  output_addr[1] = NEXT_BANK_ADDR(output_shape.n * output_stride.n * FLT_SIZE);
  // Local right matrix tensor.
  dim4 right_shape = {.n = param->left_cols,
                      .c = DIV_UP(param->right_cols, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr = NEXT_BANK_ADDR(output_addr[1] +
                              output_shape.n * output_stride.n * FLT_SIZE);
  dim4 left_shape = {.n = left_row_step,
                     .h = 1,
                     .c = DIV_UP(param->left_cols, left_cols_per_channel),
                     .w = left_cols_per_channel};
  left_addr[0] =
      NEXT_BANK_ADDR(right_addr + right_shape.n * right_stride.n * FLT_SIZE);
  okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
  left_addr[1] =
      NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);

  OKKERNEL_LOG("\nOut %u\nRight %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr[1], right_addr, left_addr[0], left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, param->left_cols,
                            param->right_cols, right_cols_per_channel,
                            param->right_cols);
  for (int n_idx = 0; n_idx < param->left_rows + left_row_step * 2;
       n_idx += left_row_step) {
    okk_parallel_start();
    if (n_idx < param->left_rows) {
      okk_gdma_32bit_matrix_S2L(
          left_addr[(n_idx / left_row_step) % 2],
          param->left_addr + n_idx * param->left_cols * FLT_SIZE,
          n_idx + left_row_step <= param->left_rows
              ? left_row_step
              : param->left_rows % left_row_step,
          param->left_cols, left_cols_per_channel, param->left_cols);
    }
    if (n_idx > 0 && n_idx < param->left_rows + left_row_step) {
      okk_bdc_matmul(
          output_addr[(n_idx / left_row_step + 1) % 2],
          left_addr[(n_idx / left_row_step + 1) % 2], right_addr, NO_USE,
          n_idx <= param->left_rows ? left_row_step
                                    : param->left_rows % left_row_step,
          param->left_cols, param->right_cols, left_cols_per_channel,
          right_cols_per_channel, false, false);
    }
    if (n_idx > left_row_step) {
      okk_gdma_32bit_matrix_L2S(
          param->output_addr +
              (n_idx - 2 * left_row_step) * param->right_cols * FLT_SIZE,
          output_addr[(n_idx / left_row_step) % 2],
          n_idx - left_row_step <= param->left_rows
              ? left_row_step
              : param->left_rows % left_row_step,
          param->right_cols, right_cols_per_channel, param->right_cols);
    }
    okk_parallel_end();
  }
}
void matmul_pingpong_left_case_12(const param_t *param, int use_npu_num_left,
                                  int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(2048, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  if (left_cols_per_channel > 128)
    left_cols_per_channel = 128;
  if (right_cols_per_channel > 128)
    right_cols_per_channel = 128;
  local_addr_t output_addr = 0, left_addr[2], right_addr[2];
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] = NEXT_BANK_ADDR(output_shape.n * output_stride.n * FLT_SIZE);

  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  // cut left row step
  info.left_row_step *= 2;
  do {
    info.left_row_step /= 2;
    // Local left matrix tensor.
    left_addr[0] = NEXT_BANK_ADDR(right_addr[0] +
                                  right_shape.n * right_stride.n * FLT_SIZE);
    left_shape.n = info.left_row_step;
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    left_addr[1] =
        NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  } while (left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE >
           LOCAL_MEM_SIZE);

  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr, right_addr[0], right_addr[0], left_addr[0],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  // end cal best split
  // batch of right cols
  bool is_parallel = false;
  for (int right_col_idx = 0; right_col_idx < param->right_cols;
       right_col_idx += info.right_col_step) {
    // deal with last irregular block
    int this_right_block_col =
        (right_col_idx + info.right_col_step) <= param->right_cols
            ? info.right_col_step
            : param->right_cols % info.right_col_step;
    for (int left_col_idx = 0; left_col_idx < param->left_cols;
         left_col_idx += info.left_col_step) {
      // deal with last irregular block
      int this_left_block_col =
          (left_col_idx + info.left_col_step) <= param->left_cols
              ? info.left_col_step
              : param->left_cols % info.left_col_step;
      //  S2L right sub-matrix-block and pin it
      //  S2L 2 blocks may be better
      okk_gdma_32bit_matrix_S2L(
          right_addr[0],
          param->right_addr +
              (left_col_idx * param->right_cols + right_col_idx) * FLT_SIZE,
          this_left_block_col, this_right_block_col, right_cols_per_channel,
          param->right_cols);
      OKKERNEL_LOG("Copy right");
      for (int left_row_idx = 0; left_row_idx < param->left_rows;
           left_row_idx += info.left_row_step) {
        int this_left_block_row =
            (left_row_idx + info.left_row_step) <= param->left_rows
                ? info.left_row_step
                : param->left_rows % info.left_row_step;
        OKKERNEL_LOG("\nThis shape is %d %d %d\nFrom %d %d %d ",
                     this_left_block_row, this_left_block_col,
                     this_right_block_col, left_row_idx, left_col_idx,
                     right_col_idx);

        // change left sub-matrix-block
        okk_gdma_32bit_matrix_S2L(
            left_addr[left_row_idx % info.left_row_step],
            param->left_addr +
                (left_row_idx * param->left_cols + left_col_idx) * FLT_SIZE,
            this_left_block_row, this_left_block_col, left_cols_per_channel,
            param->left_cols);
        if (is_parallel)
          okk_parallel_end();
        okk_parallel_start();
        is_parallel = true;
        // do matmul
        okk_bdc_matmul(output_addr + left_row_idx * output_stride.n * FLT_SIZE,
                       left_addr[left_row_idx % info.left_row_step],
                       right_addr[0], NO_USE, this_left_block_row,
                       this_left_block_col, this_right_block_col,
                       left_cols_per_channel, right_cols_per_channel, false,
                       left_col_idx == 0 ? false : true);
        OKKERNEL_LOG("\nWrite to %d",
                     left_row_idx * output_stride.n * FLT_SIZE);
      }
    }
    // Move out
    okk_gdma_32bit_matrix_L2S(param->output_addr + right_col_idx * FLT_SIZE,
                              output_addr, param->left_rows,
                              this_right_block_col, right_cols_per_channel,
                              param->right_cols);
    okk_parallel_end();
    is_parallel = false;
  }
}

void matmul_pingpong_left(const param_t *param, int use_npu_num_left,
                          int use_npu_num_right) {
  // strat cal best split
  split_info info = {MIN(2048, param->left_rows), MIN(2048, param->left_cols),
                     MIN(2048, param->right_cols), 0, 0};

  int left_cols_per_channel = DIV_UP(info.left_col_step, use_npu_num_left),
      right_cols_per_channel = DIV_UP(info.right_col_step, use_npu_num_right);

  if (left_cols_per_channel > 128)
    left_cols_per_channel = 128;
  if (right_cols_per_channel > 128)
    right_cols_per_channel = 128;
  local_addr_t output_addr = 0, left_addr[2], right_addr[2];
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = param->left_rows,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] = NEXT_BANK_ADDR(output_shape.n * output_stride.n * FLT_SIZE);

  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  // cut left row step
  info.left_row_step *= 2;
  do {
    info.left_row_step /= 2;
    // Local left matrix tensor.
    left_addr[0] = NEXT_BANK_ADDR(right_addr[0] +
                                  right_shape.n * right_stride.n * FLT_SIZE);
    left_shape.n = info.left_row_step;
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    left_addr[1] =
        NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
  } while (left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE >
           LOCAL_MEM_SIZE);

  OKKERNEL_LOG("\nParams: %d %d %d\nSteps: %d %d %d", param->left_rows,
               param->left_cols, param->right_cols, info.left_row_step,
               info.left_col_step, info.right_col_step);
  OKKERNEL_LOG("\nOut %u\nRight1 %u\nRight2 %u\nLeft1: %u\nLeft2: %u\nAll %u",
               output_addr, right_addr[0], right_addr[0], left_addr[0],
               left_addr[1],
               left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);
  // end cal best split
  // batch of right cols
  bool is_parallel = false;
  for (int right_col_idx = 0; right_col_idx < param->right_cols;
       right_col_idx += info.right_col_step) {
    // deal with last irregular block
    int this_right_block_col =
        (right_col_idx + info.right_col_step) <= param->right_cols
            ? info.right_col_step
            : param->right_cols % info.right_col_step;
    for (int left_col_idx = 0; left_col_idx < param->left_cols;
         left_col_idx += info.left_col_step) {
      // deal with last irregular block
      int this_left_block_col =
          (left_col_idx + info.left_col_step) <= param->left_cols
              ? info.left_col_step
              : param->left_cols % info.left_col_step;
      //  S2L right sub-matrix-block and pin it
      //  S2L 2 blocks may be better
      okk_gdma_32bit_matrix_S2L(
          right_addr[0],
          param->right_addr +
              (left_col_idx * param->right_cols + right_col_idx) * FLT_SIZE,
          this_left_block_col, this_right_block_col, right_cols_per_channel,
          param->right_cols);
      OKKERNEL_LOG("Copy right");
      for (int left_row_idx = 0; left_row_idx < param->left_rows;
           left_row_idx += info.left_row_step) {
        int this_left_block_row =
            (left_row_idx + info.left_row_step) <= param->left_rows
                ? info.left_row_step
                : param->left_rows % info.left_row_step;
        OKKERNEL_LOG("\nThis shape is %d %d %d\nFrom %d %d %d ",
                     this_left_block_row, this_left_block_col,
                     this_right_block_col, left_row_idx, left_col_idx,
                     right_col_idx);

        // change left sub-matrix-block
        okk_gdma_32bit_matrix_S2L(
            left_addr[left_row_idx % info.left_row_step],
            param->left_addr +
                (left_row_idx * param->left_cols + left_col_idx) * FLT_SIZE,
            this_left_block_row, this_left_block_col, left_cols_per_channel,
            param->left_cols);
        if (is_parallel)
          okk_parallel_end();
        okk_parallel_start();
        is_parallel = true;
        // do matmul
        okk_bdc_matmul(output_addr + left_row_idx * output_stride.n * FLT_SIZE,
                       left_addr[left_row_idx % info.left_row_step],
                       right_addr[0], NO_USE, this_left_block_row,
                       this_left_block_col, this_right_block_col,
                       left_cols_per_channel, right_cols_per_channel, false,
                       left_col_idx == 0 ? false : true);
        OKKERNEL_LOG("\nWrite to %d",
                     left_row_idx * output_stride.n * FLT_SIZE);
      }
      okk_parallel_end();
      is_parallel = false;
    }
    // Move out
    okk_gdma_32bit_matrix_L2S(param->output_addr + right_col_idx * FLT_SIZE,
                              output_addr, param->left_rows,
                              this_right_block_col, right_cols_per_channel,
                              param->right_cols);
  }
}

// for last 2 case, out can't be saved as full row_size, split them and calc
void matmul_split_left_row(const param_t *param, int use_npu_num_left,
                           int use_npu_num_right) {
  int global_left_row_step = 2048;
  split_info info = {MIN(global_left_row_step, param->left_rows),
                     MIN(2048, param->left_cols), MIN(1024, param->right_cols),
                     0, 0};
  int left_cols_per_channel = DIV_UP(info.left_col_step, NPU_NUM),
      right_cols_per_channel = DIV_UP(info.right_col_step, NPU_NUM);
  if (left_cols_per_channel > 128)
    left_cols_per_channel = 128;
  if (right_cols_per_channel > 128)
    right_cols_per_channel = 128;
  local_addr_t output_addr = 0, left_addr[2], right_addr[2];
  dim4 output_stride, left_stride, right_stride;
  dim4 output_shape = {.n = global_left_row_step,
                       .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                       .h = 1,
                       .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // Local right matrix tensor.
  dim4 right_shape = {.n = info.left_col_step,
                      .c = DIV_UP(info.right_col_step, right_cols_per_channel),
                      .h = 1,
                      .w = right_cols_per_channel};
  okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
  right_addr[0] = NEXT_BANK_ADDR(output_shape.n * output_stride.n * FLT_SIZE);
  right_addr[1] =
      NEXT_BANK_ADDR(right_addr[0] + right_shape.n * right_stride.n * FLT_SIZE);

  dim4 left_shape = {.n = info.left_row_step,
                     .h = 1,
                     .c = DIV_UP(info.left_col_step, left_cols_per_channel),
                     .w = left_cols_per_channel};
  info.left_row_step *= 2;
  do {
    global_left_row_step /= 2;
    output_shape.n = global_left_row_step;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    right_addr[0] = NEXT_BANK_ADDR(output_shape.n * output_stride.n * FLT_SIZE);
    // right_addr[1] = NEXT_BANK_ADDR(right_addr[0] +
    //                                right_shape.n * right_stride.n *
    //                                FLT_SIZE);

    // Local left matrix tensor.
    left_addr[0] = NEXT_BANK_ADDR(right_addr[0] +
                                  right_shape.n * right_stride.n * FLT_SIZE);
    info.left_row_step = global_left_row_step;
    left_shape.n = info.left_row_step;
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    left_addr[1] =
        NEXT_BANK_ADDR(left_addr[0] + left_shape.n * left_stride.n * FLT_SIZE);
    while (info.left_row_step &&
           left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE >
               LOCAL_MEM_SIZE) {
      info.left_row_step /= 2;
      left_shape.n = info.left_row_step;
      okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
      left_addr[1] = NEXT_BANK_ADDR(left_addr[0] +
                                    left_shape.n * left_stride.n * FLT_SIZE);
    }

    OKKERNEL_LOG("%d %d ", global_left_row_step,
                 left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE);

  } while (info.left_row_step == 0 ||
           left_addr[1] + left_shape.n * left_stride.n * FLT_SIZE >
               LOCAL_MEM_SIZE);
  OKKERNEL_LOG("%d", global_left_row_step);
  // now split by row and cal
  param_t output_block = {.left_rows = global_left_row_step,
                          .left_cols = param->left_cols,
                          .right_cols = param->right_cols,
                          .output_addr = param->output_addr,
                          .left_addr = param->left_addr,
                          .right_addr = param->right_addr};
  for (int row_idx = 0; row_idx < param->left_rows;
       row_idx += global_left_row_step) {
    output_block.left_rows =
        (row_idx + global_left_row_step) <= param->left_rows
            ? global_left_row_step
            : param->left_rows % global_left_row_step;
    output_block.output_addr =
        param->output_addr + row_idx * param->right_cols * FLT_SIZE;
    output_block.left_addr =
        param->left_addr + row_idx * param->left_cols * FLT_SIZE;
    matmul_pingpong_left(&output_block, use_npu_num_left, use_npu_num_right);
  }
}

void matmul_contest(const void *args) {
  okk_initialize();
  param_t *param = (param_t *)args;
  switch (param->left_rows) {
  case 2:
    switch (param->left_cols) {
    case 100352: // case 0
      matmul_precision(param);
      // matmul_pingpong_left(param, 64);
      break;
    case 1280: // case 1
      matmul_pingpong_left(param, 16, 16);
      break;
    case 25088: // case 2
      matmul_precision(param);
      break;
    };
    break;
  case 4: // case 3    64
    matmul_pingpong_left(param, 8, 64);
    break;
  case 32: // case 4
    matmul_pingpong_left(param, 8, 8);
    break;
  case 64: // case 5
    matmul_precision(param);
    break;
  case 79: // case 6
    matmul_pingpong_left_case_6(param, 8, 64);
    break;
  case 200: // case 7
            // origin 24
            //  3232 21    3264 16
            //  6432 16    6464 10
    matmul_pingpong_left_origin(param);
    // matmul_pingpong_left_case_7(param, 64, 64);
    break;
  case 256: // 64
    switch (param->left_cols) {
    case 768: // case 8
      matmul_pingpong_left_case_8(param, 16, 64);
      break;
    case 3072: // case 9
      matmul_pingpong_left_case_9(param, 64, 64);
      break;
    };
    break;
  case 300:                                     // case 10   64
    matmul_pingpong_left_case_10(param, 32, 8); // 2048 2048 2048
    break;
  case 1024:                                     // case 11   64
    matmul_pingpong_left_case_11(param, 64, 64); // 2048 2048 2048
    break;
  case 2048:                                     // case 12 64
    matmul_pingpong_left_case_12(param, 32, 64); // 2048 2048 2048
    break;
  case 12544: // case 13 64
    matmul_pingpong_left_case_13_14(param, 64, 64, 512);
    break;
  case 100352: // case 14 64
    matmul_pingpong_left_case_13_14(param, 8, 1, 512);
    break;
  }

  // if (param->left_cols > 8192)
  //   matmul_precision(param);
  // else if (param->left_rows > 2048) {
  //   matmul_split_left_row(param);
  // } else {
  //   matmul_pingpong_left(param);
  // }

  // if (param->left_cols > 8192)
  //   matmul_precision(param);
  // else if (param->left_cols <= 1024)
  //   matmul_naive(param);
  // else
  //   matmul_pingpong_lr_rc_lc_2l_2r_1o(param);
  OKKERNEL_LOG("\n");
  okk_poll();
}

OKKERNEL_FUNC_REGISTER(matmul_contest);
