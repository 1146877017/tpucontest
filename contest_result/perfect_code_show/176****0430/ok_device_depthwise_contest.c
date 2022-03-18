#include "okk.h"
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
  int N, C, H, W;
  int kernel_h, kernel_w;
  int pad_top, pad_bottom, pad_left, pad_right;
  int stride_h, stride_w;
  int dilation_h, dilation_w;
  unsigned long long output_addr;
  unsigned long long input_addr;
  unsigned long long kernel_addr;
} __attribute__((packed)) param_t;

void depthwise_split_N(const param_t *param) {
  const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
  const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
  const int output_h =
      (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) /
          param->stride_h +
      1;
  const int output_w =
      (param->W + param->pad_left + param->pad_right - kernel_w_ext) /
          param->stride_w +
      1;
  dim4 output_shape = {.n = 1, .c = param->C, .h = output_h, .w = output_w};
  dim4 input_shape = {.n = 1, .c = param->C, .h = param->H, .w = param->W};
  dim4 kernel_shape = {
      .n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
  // depthwise
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  dim4 output_stride, input_stride, kernel_stride;
  // output is 64-byte aligned layout
  local_addr_t output_addr[2] = {0, 0};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // input is 64-byte aligned layout
  local_addr_t input_addr[2] = {
      output_addr[0] + output_shape.n * output_stride.n * FLT_SIZE, 0};
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  // kernel is compact layout
  local_addr_t kernel_addr =
      input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE;
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);
  int batch_input_size =
          input_shape.c * input_shape.h * input_shape.w * FLT_SIZE,
      batch_output_size =
          output_shape.c * output_shape.h * output_shape.w * FLT_SIZE;
  // check local memory exceeded
  OKKERNEL_LOG("Max addr: %u %u %u",
               kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE,
               LOCAL_MEM_SIZE,
               kernel_addr * 2 + kernel_shape.n * kernel_stride.n * FLT_SIZE);
  //
  if (kernel_addr * 2 + kernel_shape.n * kernel_stride.n * FLT_SIZE <
      LOCAL_MEM_SIZE) {
    output_addr[1] =
        output_addr[0] + output_shape.n * output_stride.n * FLT_SIZE;
    input_addr[0] = output_addr[1] * 2;
    input_addr[1] = input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE;
    kernel_addr = input_addr[1] + input_shape.n * input_stride.n * FLT_SIZE;
    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                           &kernel_stride, NULL);
    //
    OKKERNEL_LOG("N pingpong");
    for (int i = 0; i < param->N + 2; i++) {
      okk_parallel_start();
      if (i < param->N) {
        // copy input from global memory to local memory
        okk_gdma_32bit_cpy_S2L(input_addr[i % 2],
                               param->input_addr + i * batch_input_size,
                               &input_shape, NULL, NULL);
      }
      if (i > 0 && i < param->N + 1) {
        okk_bdc_depthwise2d(output_addr[(i + 1) % 2], input_addr[(i + 1) % 2],
                            kernel_addr, NO_USE, &input_shape, param->kernel_h,
                            param->kernel_w, false, &padding, &stride,
                            &dilation);
      }
      if (i > 1) {
        // copy output from local memory to global memory
        okk_gdma_32bit_cpy_L2S(param->output_addr + (i - 2) * batch_output_size,
                               output_addr[i % 2], &output_shape, NULL, NULL);
      }
      okk_parallel_end();
    }
  } else if (output_shape.n * output_stride.n +
                 input_shape.n * input_stride.n * 2 +
                 kernel_shape.n * kernel_stride.n * FLT_SIZE >
             LOCAL_MEM_SIZE) {
    OKKERNEL_LOG("NN pingpong");

    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                           &kernel_stride, NULL);
    for (int i = 0; i < param->N; i++) {
      // copy input from global memory to local memory
      okk_gdma_32bit_cpy_S2L(input_addr[0],
                             param->input_addr + i * batch_input_size,
                             &input_shape, NULL, NULL);

      okk_bdc_depthwise2d(output_addr[0], input_addr[0], kernel_addr, NO_USE,
                          &input_shape, param->kernel_h, param->kernel_w, false,
                          &padding, &stride, &dilation);
      // copy output from local memory to global memory
      okk_gdma_32bit_cpy_L2S(param->output_addr + i * batch_output_size,
                             output_addr[0], &output_shape, NULL, NULL);
    }
  } else { // 2 input HARDCODE
    OKKERNEL_LOG("NN pingpong 2 input");
    okk_128_byte_aligned_stride_for_32bit(&kernel_stride, 0, &kernel_shape);
    kernel_addr = 0;
    output_addr[0] = kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE;
    input_addr[0] =
        output_addr[0] + output_shape.n * output_stride.n * FLT_SIZE;
    input_addr[1] = input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE;
    OKKERNEL_LOG("Max addr: %u %u",
                 input_addr[1] + input_shape.n * input_stride.n * FLT_SIZE,
                 LOCAL_MEM_SIZE);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                           &kernel_stride, NULL);
    // -> 0
    okk_gdma_32bit_cpy_S2L(input_addr[0],
                           param->input_addr + 0 * batch_input_size,
                           &input_shape, NULL, NULL);
    // -> 1 Cacl 0
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr[1],
                           param->input_addr + 1 * batch_input_size,
                           &input_shape, NULL, NULL);
    okk_bdc_depthwise2d(output_addr[0], input_addr[0], kernel_addr, NO_USE,
                        &input_shape, param->kernel_h, param->kernel_w, false,
                        &padding, &stride, &dilation);
    okk_parallel_end();
    // <-Out
    okk_gdma_32bit_cpy_L2S(param->output_addr + 0 * batch_output_size,
                           output_addr[0], &output_shape, NULL, NULL);
    okk_parallel_start();
    // -> 0 Cacl 1
    okk_gdma_32bit_cpy_S2L(input_addr[0],
                           param->input_addr + 2 * batch_input_size,
                           &input_shape, NULL, NULL);
    okk_bdc_depthwise2d(output_addr[0], input_addr[1], kernel_addr, NO_USE,
                        &input_shape, param->kernel_h, param->kernel_w, false,
                        &padding, &stride, &dilation);
    okk_parallel_end();
    // <-Out
    okk_gdma_32bit_cpy_L2S(param->output_addr + 1 * batch_output_size,
                           output_addr[0], &output_shape, NULL, NULL);

    // -> 1 Cacl 0
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr[1],
                           param->input_addr + 3 * batch_input_size,
                           &input_shape, NULL, NULL);
    okk_bdc_depthwise2d(output_addr[0], input_addr[0], kernel_addr, NO_USE,
                        &input_shape, param->kernel_h, param->kernel_w, false,
                        &padding, &stride, &dilation);
    okk_parallel_end();
    // <-Out

    okk_gdma_32bit_cpy_L2S(param->output_addr + 2 * batch_output_size,
                           output_addr[0], &output_shape, NULL, NULL);

    //  Cacl 1
    okk_bdc_depthwise2d(output_addr[0], input_addr[1], kernel_addr, NO_USE,
                        &input_shape, param->kernel_h, param->kernel_w, false,
                        &padding, &stride, &dilation);

    // <-Out
    okk_gdma_32bit_cpy_L2S(param->output_addr + 3 * batch_output_size,
                           output_addr[0], &output_shape, NULL, NULL);
  }
}
void depthwise_split_N_H_USE_MORE_NPU_HARDCODE(const param_t *param) {
  int TUNE_H = 40;

  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
  const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
  // OKKERNEL_LOG("\nKernel ext: %d %d", kernel_h_ext, kernel_w_ext);
  const int output_h =
      (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) /
          param->stride_h +
      1;
  const int output_w =
      (param->W + param->pad_left + param->pad_right - kernel_w_ext) /
          param->stride_w +
      1;
  dim4 output_stride, input_stride, kernel_stride, global_input_stride,
      global_output_stride;
  dim4 output_shape = {
      .n = 1, .c = param->C * param->N, .h = output_h, .w = output_w};
  okk_continuous_stride(&global_output_stride, &output_shape);
  output_shape.h = TUNE_H;
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // OKKERNEL_LOG("\nOutput: %d %d %d %d", output_shape.n, output_shape.c,
  //              output_shape.h, output_shape.w);
  dim4 global_input_shape = {
      .n = 1, .c = param->C * param->N, .h = param->H, .w = param->W};
  okk_continuous_stride(&global_input_stride, &global_input_shape);
  // try 1 row per time, can be optimized
  dim4 input_shape = {.n = 1,
                      .c = param->C * param->N,
                      .h = kernel_h_ext + stride.h * (TUNE_H - 1),
                      .w = param->W};
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  // OKKERNEL_LOG("\nInput: %d %d %d %d", input_shape.n, input_shape.c,
  //              input_shape.h, input_shape.w);
  dim4 kernel_shape = {.n = 1,
                       .c = param->C * param->N,
                       .h = param->kernel_h,
                       .w = param->kernel_w};
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);

  // output is 64-byte aligned layout
  local_addr_t output_addr[2] = {0, 0};
  output_addr[1] = output_addr[0] + output_shape.n * output_stride.n * FLT_SIZE;
  // input is 64-byte aligned layout
  local_addr_t input_addr[2] = {
      output_addr[1] + output_shape.n * output_stride.n * FLT_SIZE, 0};
  input_addr[1] = input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE;
  // kernel is compact layout
  local_addr_t kernel_addr =
      input_addr[1] + input_shape.n * input_stride.n * FLT_SIZE;
  // check local memory exceeded
  OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE <
                  LOCAL_MEM_SIZE);
  OKKERNEL_LOG("\nMax addr: %u %u ",
               kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE,
               LOCAL_MEM_SIZE);
  // copy kernel from global memory to local memory
  kernel_shape.c = param->C;
  okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                         &kernel_stride, NULL);
  okk_gdma_32bit_cpy_S2L(kernel_addr + LOCAL_MEM_SIZE * param->C,
                         param->kernel_addr, &kernel_shape, &kernel_stride,
                         NULL);
  okk_gdma_32bit_cpy_S2L(kernel_addr + LOCAL_MEM_SIZE * param->C * 2,
                         param->kernel_addr, &kernel_shape, &kernel_stride,
                         NULL);
  okk_gdma_32bit_cpy_S2L(kernel_addr + LOCAL_MEM_SIZE * param->C * 3,
                         param->kernel_addr, &kernel_shape, &kernel_stride,
                         NULL);
  kernel_shape.c = param->C * param->N;
  // BUGGY silly but useful, not general
  int bdc = 0, gdma = 1;
  OKKERNEL_LOG("NH no pingpong");
  for (int h_idx = 0; h_idx + kernel_h_ext - 1 + padding.bottom <
                      param->H + stride.h * TUNE_H * 2;
       h_idx +=
       h_idx == 0 ? stride.h * TUNE_H - param->pad_top : stride.h * TUNE_H) {
    okk_parallel_start();
    if (h_idx + kernel_h_ext - 1 + padding.bottom < param->H) {
      if (h_idx > 0 &&
          h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 < param->H) {
        input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
        padding.top = 0;
        padding.bottom = 0;
      } else if (h_idx == 0) {
        input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h - padding.top;
        padding.top = param->pad_top;
        padding.bottom = 0;
      } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                 param->H) {
        input_shape.h = param->H - h_idx;
        padding.top = 0;
        padding.bottom = kernel_h_ext - input_shape.h;
      }

      okk_gdma_32bit_cpy_S2L(input_addr[gdma],
                             param->input_addr + h_idx * param->W * FLT_SIZE,
                             &input_shape, NULL, &global_input_stride);
    }
    if (h_idx > 0 && h_idx + kernel_h_ext - 1 + padding.bottom <
                         param->H + stride.h * TUNE_H) {
      int h_idx_temp = MAX(0, h_idx - stride.h * TUNE_H);
      if (h_idx_temp > 0 &&
          h_idx_temp + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 < param->H) {
        input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
        padding.top = 0;
        padding.bottom = 0;
      } else if (h_idx_temp == 0) {
        input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h - padding.top;
        padding.top = param->pad_top;
        padding.bottom = 0;
      } else if (h_idx_temp + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                 param->H) {
        input_shape.h = param->H - h_idx_temp;
        padding.top = 0;
        padding.bottom = kernel_h_ext - input_shape.h;
      }

      okk_bdc_depthwise2d(output_addr[bdc], input_addr[bdc], kernel_addr,
                          NO_USE, &input_shape, param->kernel_h,
                          param->kernel_w, false, &padding, &stride, &dilation);
    }

    if (h_idx > stride.h * TUNE_H - param->pad_top) {
      int h_idx_temp = MAX(0, h_idx - stride.h * TUNE_H * 2);
      if (h_idx_temp > 0 &&
          h_idx_temp + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 < param->H) {
        input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
        padding.top = 0;
        padding.bottom = 0;
      } else if (h_idx_temp == 0) {
        input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h - padding.top;
        padding.top = param->pad_top;
        padding.bottom = 0;
      } else if (h_idx_temp + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                 param->H) {
        input_shape.h = param->H - h_idx_temp;
        padding.top = 0;
        padding.bottom = kernel_h_ext - input_shape.h;
      }

      okk_gdma_32bit_cpy_L2S(
          param->output_addr + (h_idx_temp + param->pad_top) / stride.h *
                                   output_shape.w * FLT_SIZE,
          output_addr[gdma], &output_shape, &global_output_stride, NULL);
    }
    okk_parallel_end();
    bdc = 1 - bdc;
    gdma = 1 - gdma;
  }
}
void depthwise_split_N_USE_MORE_NPU_HARDCODE(const param_t *param) {
  const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
  const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
  const int output_h =
      (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) /
          param->stride_h +
      1;
  const int output_w =
      (param->W + param->pad_left + param->pad_right - kernel_w_ext) /
          param->stride_w +
      1;
  dim4 output_shape = {
      .n = 1, .c = param->C * param->N, .h = output_h, .w = output_w};
  dim4 input_shape = {
      .n = 1, .c = param->C * param->N, .h = param->H, .w = param->W};
  dim4 kernel_shape = {.n = 1,
                       .c = param->C * param->N,
                       .h = param->kernel_h,
                       .w = param->kernel_w};
  // depthwise
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  dim4 output_stride, input_stride, kernel_stride;
  // output is 64-byte aligned layout
  local_addr_t output_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // input is 64-byte aligned layout
  local_addr_t input_addr =
      output_addr + output_shape.n * output_stride.n * FLT_SIZE;
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  // kernel is compact layout
  local_addr_t kernel_addr =
      input_addr + input_shape.n * input_stride.n * FLT_SIZE;
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);
  // check local memory exceeded
  if (kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE >
      LOCAL_MEM_SIZE) {
    depthwise_split_N_H_USE_MORE_NPU_HARDCODE(param);
    return;
  }
  OKKERNEL_LOG("Max addr: %u %u",
               kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE,
               LOCAL_MEM_SIZE);

  OKKERNEL_LOG("USE 12 NPU NO PING PONG");
  kernel_shape.c = param->C;
  okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                         &kernel_stride, NULL);
  okk_gdma_32bit_cpy_L2L(kernel_addr + LOCAL_MEM_SIZE * 3, kernel_addr,
                         &kernel_shape, NULL, NULL);
  kernel_shape.c = param->C * 2;
  okk_gdma_32bit_cpy_L2L(kernel_addr + LOCAL_MEM_SIZE * 6, kernel_addr,
                         &kernel_shape, NULL, NULL);
  // okk_gdma_32bit_cpy_S2L(kernel_addr + LOCAL_MEM_SIZE * 3,
  // param->kernel_addr,
  //                        &kernel_shape, &kernel_stride, NULL);
  // okk_gdma_32bit_cpy_S2L(kernel_addr + LOCAL_MEM_SIZE * 6,
  // param->kernel_addr,
  //                        &kernel_shape, &kernel_stride, NULL);
  // okk_gdma_32bit_cpy_S2L(kernel_addr + LOCAL_MEM_SIZE * 9,
  // param->kernel_addr,
  //                        &kernel_shape, &kernel_stride, NULL);
  kernel_shape.c = param->C * param->N;
  // copy input from global memory to local memory
  okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &input_shape, NULL,
                         NULL);

  okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, NO_USE,
                      &input_shape, param->kernel_h, param->kernel_w, false,
                      &padding, &stride, &dilation);
  // copy output from local memory to global memory
  okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &output_shape, NULL,
                         NULL);
}

void depthwise_contest(const void *args) {
  okk_initialize();
  param_t *param = (param_t *)args;
  if (param->H == 640) {
    depthwise_split_N_H_USE_MORE_NPU_HARDCODE(param);
  } else if (param->C * param->N < NPU_NUM)
    depthwise_split_N_USE_MORE_NPU_HARDCODE(param);
  else
    depthwise_split_N(param);
  // depthwise_split_N_H(param);
  OKKERNEL_LOG("\n");
  okk_poll();
}
OKKERNEL_FUNC_REGISTER(depthwise_contest);
