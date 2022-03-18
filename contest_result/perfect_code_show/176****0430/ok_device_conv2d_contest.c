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
// #define TUNE_H 2 // should be removed after demo
// #define DEBUG 0
#ifndef DEBUG
#undef OKKERNEL_LOG
#define OKKERNEL_LOG(...)                                                      \
  {}
#endif

typedef struct {
  int N, IC, OC, H, W;
  int kernel_h, kernel_w;
  int pad_top, pad_bottom, pad_left, pad_right;
  int stride_h, stride_w;
  int dilation_h, dilation_w;
  unsigned long long output_addr;
  unsigned long long input_addr;
  unsigned long long kernel_addr;
} __attribute__((packed)) param_t;

void case_tune_N(const param_t *param) {
  int n_step = MIN(1, param->N);
  const int IC_new = (param->IC + 1) / 2;
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
      .n = n_step, .c = param->OC, .h = output_h, .w = output_w};
  dim4 input_shape = {
      .n = n_step, .c = param->IC, .h = param->H, .w = param->W};
  dim4 kernel_shape = {.n = IC_new,
                       .c = param->OC,
                       .h = param->kernel_h,
                       .w = param->kernel_w * 2};
  // conv2d
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  // view the data type of kernel as fp32x2
  dim4 kernel_shape_2IC = {
      .n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
  dim4 kernel_stride_2IC;
  dim4 output_stride, input_stride, kernel_stride;
  // output is 64-byte aligned layout
  local_addr_t output_addr[2] = {0, 0};
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  output_addr[1] = output_addr[0] + output_shape.n * output_stride.n * FLT_SIZE;
  // input is 64-byte aligned layout
  local_addr_t input_addr[2] = {
      output_addr[1] + output_shape.n * output_stride.n * FLT_SIZE, 0};
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  input_addr[1] = input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE;
  // kernel is compact layout
  local_addr_t kernel_addr =
      input_addr[1] + input_shape.n * input_stride.n * FLT_SIZE;
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);
  // copy kernel from global memory to local memory
  okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                         &kernel_stride, NULL);
  okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

  OKKERNEL_LOG("Max addr %u %u",
               kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE,
               LOCAL_MEM_SIZE);
  for (int n_idx = 0; n_idx < DIV_UP(param->N, n_step) + 2; n_idx++) {
    okk_parallel_start();
    if (n_idx < DIV_UP(param->N, n_step)) {
      okk_gdma_32bit_cpy_S2L(input_addr[n_idx % 2],
                             param->input_addr +
                                 n_idx * n_step * input_shape.c *
                                     input_shape.h * input_shape.w * FLT_SIZE,
                             &input_shape, NULL, NULL);
    }
    if (n_idx > 1) {
      okk_gdma_32bit_cpy_L2S(param->output_addr +
                                 (n_idx - 2) * n_step * output_shape.c *
                                     output_shape.h * output_shape.w * FLT_SIZE,
                             output_addr[n_idx % 2], &output_shape, NULL, NULL);
    }
    if (n_idx > 0 && n_idx < DIV_UP(param->N, n_step) + 1) {
      okk_bdc_conv2d(output_addr[(n_idx + 1) % 2], input_addr[(n_idx + 1) % 2],
                     kernel_addr, NO_USE, &input_shape, param->OC,
                     param->kernel_h, param->kernel_w, &input_stride,
                     &kernel_stride_2IC, false, false, &padding, &stride,
                     &dilation);
    }
    okk_parallel_end();
  }
}

void case_tune_NH(const param_t *param, int tune_h) {
  // conv2d
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  //
  const int IC_new = (param->IC + 1) / 2;
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
  int TUNE_H = tune_h;
  dim4 kernel_shape = {.n = IC_new,
                       .c = param->OC,
                       .h = param->kernel_h,
                       .w = param->kernel_w * 2};
  // view the data type of kernel as fp32x2
  dim4 kernel_shape_2IC = {
      .n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
  dim4 kernel_stride_2IC;
  dim4 output_stride, input_stride, kernel_stride, global_input_stride,
      global_output_stride;
  // output is 64-byte aligned layout
  local_addr_t output_addr[2] = {0, 0};
  dim4 output_shape = {.n = 1, .c = param->OC, .h = output_h, .w = output_w};
  okk_continuous_stride(&global_output_stride, &output_shape);
  output_shape.h = TUNE_H;
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  output_addr[1] = NEXT_BANK_ADDR(output_addr[0] +
                                  output_shape.n * output_stride.n * FLT_SIZE);
  // input is 64-byte aligned layout
  dim4 global_input_shape = {
      .n = 1, .c = param->IC, .h = param->H, .w = param->W};
  okk_continuous_stride(&global_input_stride, &global_input_shape);
  local_addr_t input_addr[2] = {
      NEXT_BANK_ADDR(output_addr[1] +
                     output_shape.n * output_stride.n * FLT_SIZE),
      0};
  dim4 input_shape = {.n = 1,
                      .c = param->IC,
                      .h = kernel_h_ext + stride.h * (TUNE_H - 1),
                      .w = param->W};
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  input_addr[1] =
      NEXT_BANK_ADDR(input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE);
  // kernel is compact layout
  local_addr_t kernel_addr =
      NEXT_BANK_ADDR(input_addr[1] + input_shape.n * input_stride.n * FLT_SIZE);
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);
  // copy kernel from global memory to local memory
  okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                         &kernel_stride, NULL);

  okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

  OKKERNEL_LOG("\nSize occupied: input %d, kernel %d, output %d TUNE_H %d",
               input_shape.n * input_stride.n * FLT_SIZE,
               kernel_shape.n * kernel_stride.n * FLT_SIZE,
               output_shape.n * output_stride.n * FLT_SIZE, TUNE_H);

  int batch_in_size = param->H * param->W * param->IC * FLT_SIZE,
      batch_out_size = param->OC * output_h * output_w * FLT_SIZE;

  for (int n_idx = 0; n_idx < param->N; n_idx++) {
    for (int h_idx = -param->pad_top;
         h_idx + kernel_h_ext - 1 < param->H + stride.h * TUNE_H * 2;
         h_idx += stride.h * TUNE_H) {
      okk_parallel_start();
      if (h_idx + param->pad_top > stride.h * TUNE_H) {
        int temp_h_idx = h_idx - stride.h * TUNE_H * 2;
        if (temp_h_idx < 0) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + temp_h_idx;
          padding.top = -temp_h_idx;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
        } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                   param->H) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
          padding.top = 0;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
        } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                   param->H) {
          input_shape.h = param->H - temp_h_idx;
          padding.top = 0;
          padding.bottom = param->pad_bottom;
          output_shape.h =
              (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                  param->stride_h +
              1;
        }
        okk_gdma_32bit_cpy_L2S(
            param->output_addr +
                (temp_h_idx + param->pad_top) / stride.h * output_shape.w *
                    FLT_SIZE +
                n_idx * batch_out_size,
            output_addr[((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) %
                        2],
            &output_shape, &global_output_stride, NULL);
        OKKERNEL_LOG("<- %d/%d", (temp_h_idx + param->pad_top) / stride.h,((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) %
                        2);
      }
      if (h_idx + kernel_h_ext - 1 < param->H) {
        if (h_idx < 0) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + h_idx;
          padding.top = -h_idx;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
        } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                   param->H) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
          padding.top = 0;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
        } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                   param->H) {
          input_shape.h = param->H - h_idx;
          padding.top = 0;
          padding.bottom = param->pad_bottom;
          output_shape.h =
              (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                  param->stride_h +
              1;
        }
        okk_gdma_32bit_cpy_S2L(
            input_addr[((h_idx + param->pad_top) / (stride.h * TUNE_H)) % 2],
            param->input_addr + MAX(h_idx, 0) * param->W * FLT_SIZE +
                n_idx * batch_in_size,
            &input_shape, NULL, &global_input_stride);
        OKKERNEL_LOG("-> %d/%d", MAX(h_idx, 0),((h_idx + param->pad_top) / (stride.h * TUNE_H)) % 2);

      }
      if (h_idx + param->pad_top > 0 &&
          h_idx + kernel_h_ext - 1 < param->H + stride.h * TUNE_H) {

        int temp_h_idx = h_idx - stride.h * TUNE_H;
        if (temp_h_idx < 0) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + temp_h_idx;
          padding.top = -temp_h_idx;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
        } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                   param->H) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
          padding.top = 0;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
          // OKKERNEL_LOG("Normal");
        } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                   param->H) {
          input_shape.h = param->H - temp_h_idx;
          padding.top = 0;
          padding.bottom = param->pad_bottom;
          output_shape.h =
              (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                  param->stride_h +
              1;
        }
        okk_bdc_conv2d(
            output_addr[((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) %
                        2],
            input_addr[((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) %
                       2],
            kernel_addr, NO_USE, &input_shape, param->OC, param->kernel_h,
            param->kernel_w, &input_stride, &kernel_stride_2IC, false, false,
            &padding, &stride, &dilation);
        OKKERNEL_LOG("^ %d",((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) %
                        2);

      }
      okk_parallel_end();
    }
  }
}

void conv2_split_N_H(const param_t *param) {
  // conv2d
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  //
  const int IC_new = (param->IC + 1) / 2;
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
  int TUNE_H = 16;
  dim4 kernel_shape = {.n = IC_new,
                       .c = param->OC,
                       .h = param->kernel_h,
                       .w = param->kernel_w * 2};
  // view the data type of kernel as fp32x2
  dim4 kernel_shape_2IC = {
      .n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
  dim4 kernel_stride_2IC;
  dim4 output_stride, input_stride, kernel_stride, global_input_stride,
      global_output_stride;
  // output is 64-byte aligned layout
  local_addr_t output_addr[2] = {0, 0};
  dim4 output_shape = {.n = 1, .c = param->OC, .h = output_h, .w = output_w};
  okk_continuous_stride(&global_output_stride, &output_shape);
  output_shape.h = TUNE_H;
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // input is 64-byte aligned layout
  dim4 global_input_shape = {
      .n = 1, .c = param->IC, .h = param->H, .w = param->W};
  okk_continuous_stride(&global_input_stride, &global_input_shape);
  local_addr_t input_addr[2] = {
      output_addr[0] + output_shape.n * output_stride.n * FLT_SIZE, 0};
  dim4 input_shape = {.n = 1,
                      .c = param->IC,
                      .h = kernel_h_ext + stride.h * (TUNE_H - 1),
                      .w = param->W};
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  // kernel is compact layout
  local_addr_t kernel_addr =
      input_addr[0] + input_shape.n * input_stride.n * FLT_SIZE;
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);
  // copy kernel from global memory to local memory
  okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                         &kernel_stride, NULL);

  okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

  // OKKERNEL_LOG("\nSize occupied: input %d, kernel %d, output %d TUNE_H %d",
  //              input_shape.n * input_stride.n * FLT_SIZE,
  //              kernel_shape.n * kernel_stride.n * FLT_SIZE,
  //              output_shape.n * output_stride.n * FLT_SIZE, TUNE_H);
  // OKKERNEL_LOG("\n %u %u",
  //              kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE,
  //              LOCAL_MEM_SIZE);
  // return;
  // copy input from global memory to local memory
  // okk_128_byte_aligned_stride_for_32bit(&kernel_stride, 0, &kernel_shape);
  int batch_in_size = param->H * param->W * param->IC * FLT_SIZE,
      batch_out_size = param->OC * output_h * output_w * FLT_SIZE;

  if (kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE <
      LOCAL_MEM_SIZE / 2) {
    OKKERNEL_LOG("N H ping pong");
    input_addr[1] = input_addr[0] + LOCAL_MEM_SIZE / 2;
    output_addr[1] = output_addr[0] + LOCAL_MEM_SIZE / 2;
    for (int n_idx = 0; n_idx < param->N; n_idx++) {
      for (int h_idx = -param->pad_top;
           h_idx + kernel_h_ext - 1 < param->H + stride.h * TUNE_H * 2;
           h_idx += stride.h * TUNE_H) {
        //
        okk_parallel_start();
        // OKKERNEL_LOG("new iter");
        if (h_idx + param->pad_top > stride.h * TUNE_H) {
          int temp_h_idx = h_idx - stride.h * TUNE_H * 2;
          //   OKKERNEL_LOG("Move out %d hidx(t): %d/%d",
          //                ((h_idx + param->pad_top) / (stride.h * TUNE_H)) %
          //                2, h_idx, temp_h_idx);

          if (temp_h_idx < 0) {
            input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + temp_h_idx;
            padding.top = -temp_h_idx;
            padding.bottom = 0;
            output_shape.h = TUNE_H;
            // OKKERNEL_LOG(
            //     "\n temp_h_idx<0, with params: \ninshape.h %d \npadding "
            //     "%d",
            //     input_shape.h, padding.top);
          } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                     param->H) {
            input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
            padding.top = 0;
            padding.bottom = 0;
            output_shape.h = TUNE_H;
            // OKKERNEL_LOG("Normal");
          } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                     param->H) {
            input_shape.h = param->H - temp_h_idx;
            // OKKERNEL_LOG("temp_h_idx %d, H %d, inshape.H %d, %d", temp_h_idx,
            //              param->H, input_shape.h,
            //              temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext
            //              -
            //                  1);
            padding.top = 0;
            padding.bottom = param->pad_bottom;
            output_shape.h =
                (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                    param->stride_h +
                1;
            // OKKERNEL_LOG("Last input shape is %d, output shape is %d",
            //              input_shape.h, output_shape.h);
          }
          okk_gdma_32bit_cpy_L2S(param->output_addr +
                                     (temp_h_idx + param->pad_top) / stride.h *
                                         output_shape.w * FLT_SIZE +
                                     n_idx * batch_out_size,
                                 output_addr[((temp_h_idx + param->pad_top) /
                                              (stride.h * TUNE_H)) %
                                             2],
                                 &output_shape, &global_output_stride, NULL);
          //   break;
          //   OKKERNEL_LOG(
          //       "From %d-%d to %d-%d", temp_h_idx, (temp_h_idx +
          //       input_shape.h), (temp_h_idx + param->pad_top) / stride.h,
          //       (temp_h_idx + param->pad_top) / stride.h + output_shape.h);
        }
        if (h_idx + kernel_h_ext - 1 < param->H) {
          //   OKKERNEL_LOG("Move in %d",
          //                ((h_idx + param->pad_top) / (stride.h * TUNE_H)) %
          //                2);
          if (h_idx < 0) {
            input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + h_idx;
            padding.top = -h_idx;
            padding.bottom = 0;
            output_shape.h = TUNE_H;
            // OKKERNEL_LOG("\n h_idx<0, with params: \ninshape.h %d \npadding "
            //              "%d",
            //              input_shape.h, padding.top);
          } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                     param->H) {
            input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
            padding.top = 0;
            padding.bottom = 0;
            output_shape.h = TUNE_H;
            // OKKERNEL_LOG("Normal");
          } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                     param->H) {
            input_shape.h = param->H - h_idx;
            padding.top = 0;
            padding.bottom = param->pad_bottom;
            output_shape.h =
                (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                    param->stride_h +
                1;
            // OKKERNEL_LOG("Last input shape is %d, output shape is %d",
            //              input_shape.h, output_shape.h);
          }
          okk_gdma_32bit_cpy_S2L(
              input_addr[((h_idx + param->pad_top) / (stride.h * TUNE_H)) % 2],
              param->input_addr + MAX(h_idx, 0) * param->W * FLT_SIZE +
                  n_idx * batch_in_size,
              &input_shape, NULL, &global_input_stride);
        }
        if (h_idx + param->pad_top > 0 &&
            h_idx + kernel_h_ext - 1 < param->H + stride.h * TUNE_H) {

          int temp_h_idx = h_idx - stride.h * TUNE_H;
          //   OKKERNEL_LOG("Calc %d",
          //                ((temp_h_idx + param->pad_top) / (stride.h *
          //                TUNE_H)) %
          //                    2);

          if (temp_h_idx < 0) {
            input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + temp_h_idx;
            padding.top = -temp_h_idx;
            padding.bottom = 0;
            output_shape.h = TUNE_H;
            // OKKERNEL_LOG(
            //     "\n temp_h_idx<0, with params: \ninshape.h %d \npadding "
            //     "%d",
            //     input_shape.h, padding.top);
          } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                     param->H) {
            input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
            padding.top = 0;
            padding.bottom = 0;
            output_shape.h = TUNE_H;
            // OKKERNEL_LOG("Normal");
          } else if (temp_h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                     param->H) {
            input_shape.h = param->H - temp_h_idx;
            padding.top = 0;
            padding.bottom = param->pad_bottom;
            output_shape.h =
                (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                    param->stride_h +
                1;
            // OKKERNEL_LOG("Last input shape is %d, output shape is %d",
            //              input_shape.h, output_shape.h);
          }
          okk_bdc_conv2d(
              output_addr
                  [((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) % 2],
              input_addr[((temp_h_idx + param->pad_top) / (stride.h * TUNE_H)) %
                         2],
              kernel_addr, NO_USE, &input_shape, param->OC, param->kernel_h,
              param->kernel_w, &input_stride, &kernel_stride_2IC, false, false,
              &padding, &stride, &dilation);
        }
        okk_parallel_end();
      }
    }
  } else {

    //  for (int h_idx = -param->pad_top;
    //        h_idx + kernel_h_ext - 1 < param->H + stride.h * TUNE_H * 2;
    //        h_idx += stride.h * TUNE_H) {

    OKKERNEL_LOG("NH N PP");
    for (int n_idx = 0; n_idx < param->N; n_idx++) {
      for (int h_idx = -param->pad_top;
           h_idx + kernel_h_ext - 1 - param->pad_bottom < param->H;
           h_idx += stride.h * TUNE_H) {
        //
        OKKERNEL_LOG("h idx : %d", h_idx);
        // OKKERNEL_ASSERT(h_idx > 0);
        if (h_idx < 0) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h + h_idx;
          padding.top = -h_idx;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
          OKKERNEL_LOG("\n h_idx<0, with params: \ninshape.h %d \npadding "
                       "%d",
                       input_shape.h, padding.top);
        } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 <
                   param->H) {
          input_shape.h = kernel_h_ext + (TUNE_H - 1) * stride.h;
          padding.top = 0;
          padding.bottom = 0;
          output_shape.h = TUNE_H;
          OKKERNEL_LOG("Normal");
        } else if (h_idx + stride.h * (TUNE_H - 1) + kernel_h_ext - 1 >=
                   param->H) {
          input_shape.h = param->H - h_idx;
          padding.top = 0;
          padding.bottom = param->pad_bottom;
          output_shape.h =
              (input_shape.h + padding.top + padding.bottom - kernel_h_ext) /
                  param->stride_h +
              1;
          //   OKKERNEL_LOG("Last input shape is %d, output shape is %d",
          //                input_shape.h, output_shape.h);
        }
        okk_gdma_32bit_cpy_S2L(input_addr[0],
                               param->input_addr +
                                   MAX(h_idx, 0) * param->W * FLT_SIZE +
                                   n_idx * batch_in_size,
                               &input_shape, NULL, &global_input_stride);

        okk_bdc_conv2d(output_addr[0], input_addr[0], kernel_addr, NO_USE,
                       &input_shape, param->OC, param->kernel_h,
                       param->kernel_w, &input_stride, &kernel_stride_2IC,
                       false, false, &padding, &stride, &dilation);

        okk_gdma_32bit_cpy_L2S(param->output_addr +
                                   (h_idx + param->pad_top) / stride.h *
                                       output_shape.w * FLT_SIZE +
                                   n_idx * batch_out_size,
                               output_addr[0], &output_shape,
                               &global_output_stride, NULL);
        OKKERNEL_LOG("outshape.h %d", output_shape.h);
        // OKKERNEL_LOG("From %d-%d to %d-%d", h_idx, (h_idx + input_shape.h),
        //              (h_idx + param->pad_top) / stride.h,
        //              (h_idx + param->pad_top) / stride.h + output_shape.h);
        // break;
      }
    }
  }
}

void conv2_split_N(const param_t *param) {
  int n_step = MIN(1, param->N);
  const int IC_new = (param->IC + 1) / 2;
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
      .n = n_step, .c = param->OC, .h = output_h, .w = output_w};
  dim4 input_shape = {
      .n = n_step, .c = param->IC, .h = param->H, .w = param->W};
  dim4 kernel_shape = {.n = IC_new,
                       .c = param->OC,
                       .h = param->kernel_h,
                       .w = param->kernel_w * 2};
  // conv2d
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  // view the data type of kernel as fp32x2
  dim4 kernel_shape_2IC = {
      .n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
  dim4 kernel_stride_2IC;
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
  // copy kernel from global memory to local memory
  okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape,
                         &kernel_stride, NULL);
  okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
  if (kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE >
      LOCAL_MEM_SIZE / 2) {
    conv2_split_N_H(param);
    return;
  }
  // copy input from global memory to local memory
  // okk_128_byte_aligned_stride_for_32bit(&kernel_stride, 0, &kernel_shape);
  OKKERNEL_LOG("Max addr %u %u",
               kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE,
               LOCAL_MEM_SIZE);
  if ((kernel_addr + kernel_shape.n * kernel_stride.n * FLT_SIZE) * 2 <=
      LOCAL_MEM_SIZE) {
    OKKERNEL_LOG(" N PINGPONG");
    //
    output_addr[1] = output_addr[0] + LOCAL_MEM_SIZE / 2;
    input_addr[1] = input_addr[0] + LOCAL_MEM_SIZE / 2;
    for (int n_idx = 0; n_idx < DIV_UP(param->N, n_step) + 2; n_idx++) {
      okk_parallel_start();
      if (n_idx < DIV_UP(param->N, n_step)) {
        // OKKERNEL_LOG("move %d -> input %u", n_idx, n_idx % 2);
        okk_gdma_32bit_cpy_S2L(input_addr[n_idx % 2],
                               param->input_addr +
                                   n_idx * n_step * input_shape.c *
                                       input_shape.h * input_shape.w * FLT_SIZE,
                               &input_shape, NULL, NULL);
      }
      if (n_idx > 1) {
        // OKKERNEL_LOG("move %d <- output %u", n_idx - 2, n_idx % 2);
        // copy output from local memory to global memory
        okk_gdma_32bit_cpy_L2S(
            param->output_addr + (n_idx - 2) * n_step * output_shape.c *
                                     output_shape.h * output_shape.w * FLT_SIZE,
            output_addr[n_idx % 2], &output_shape, NULL, NULL);
      }
      if (n_idx > 0 && n_idx < DIV_UP(param->N, n_step) + 1) {
        // OKKERNEL_LOG("Cal step %u", n_idx - 1);
        okk_bdc_conv2d(output_addr[(n_idx + 1) % 2],
                       input_addr[(n_idx + 1) % 2], kernel_addr, NO_USE,
                       &input_shape, param->OC, param->kernel_h,
                       param->kernel_w, &input_stride, &kernel_stride_2IC,
                       false, false, &padding, &stride, &dilation);
      }
      okk_parallel_end();
    }
  } else {
    OKKERNEL_LOG(" N no PINGPONG");
    for (int n_idx = 0; n_idx < DIV_UP(param->N, n_step); n_idx++) {
      okk_gdma_32bit_cpy_S2L(input_addr[0],
                             param->input_addr +
                                 n_idx * n_step * input_shape.c *
                                     input_shape.h * input_shape.w * FLT_SIZE,
                             &input_shape, NULL, NULL);

      okk_bdc_conv2d(output_addr[0], input_addr[0], kernel_addr, NO_USE,
                     &input_shape, param->OC, param->kernel_h, param->kernel_w,
                     &input_stride, &kernel_stride_2IC, false, false, &padding,
                     &stride, &dilation);
      okk_gdma_32bit_cpy_L2S(param->output_addr +
                                 n_idx * n_step * output_shape.c *
                                     output_shape.h * output_shape.w * FLT_SIZE,
                             output_addr[0], &output_shape, NULL, NULL);
    }
  }
}

void conv2d_split_IC(const param_t *param, int ic_step, bool reuse_output,
                     int index) {
  int IC_step = ic_step;
  int N_step = 1;
  const int IC_new = (param->IC + 1) / 2;
  // conv2d
  Padding padding = {.top = param->pad_top,
                     .bottom = param->pad_bottom,
                     .left = param->pad_left,
                     .right = param->pad_right};
  dim2 stride = {.h = param->stride_h, .w = param->stride_w};
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
      .n = N_step, .c = param->OC, .h = output_h, .w = output_w};
  dim4 input_shape = {
      .n = N_step, .c = IC_step * 2, .h = param->H, .w = param->W};
  dim4 kernel_shape = {.n = IC_step,
                       .c = param->OC,
                       .h = param->kernel_h,
                       .w = param->kernel_w * 2};
  dim4 output_stride, input_stride, kernel_stride;
  // output is 64-byte aligned layout
  local_addr_t output_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
  // input is 64-byte aligned layout
  local_addr_t input_addr[2] = {
      output_addr + output_shape.n * output_stride.n * sizeof(float), 0};
  okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
  input_addr[1] =
      input_addr[0] + input_shape.n * input_stride.n * sizeof(float);
  // kernel is compact layout
  local_addr_t kernel_addr[2] = {
      input_addr[1] + input_shape.n * input_stride.n * sizeof(float), 0};
  okk_128_byte_aligned_stride_for_32bit(&kernel_stride, 0, &kernel_shape);
  kernel_addr[1] = kernel_addr[0] + kernel_shape.n * kernel_stride.n * FLT_SIZE;
  okk_compact_stride(&kernel_stride, 0, &kernel_shape);
  dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
  // view the data type of kernel as fp32x2
  dim4 kernel_shape_2IC = {
      .n = IC_step, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
  dim4 kernel_stride_2IC;
  okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
  OKKERNEL_LOG("\nMax addr %d\nOut %d\nIn %d\nKer1 %d\nKer2 %d",
               kernel_addr[1] + kernel_shape.n * kernel_stride.n * FLT_SIZE,
               output_addr, input_addr[0], kernel_addr[0], kernel_addr[1]);
  bool is_para = false;
  for (int n_idx = 0; n_idx < param->N; n_idx += N_step) {
    global_addr_t this_batch_input_addr =
        n_idx * param->IC * param->H * param->W * FLT_SIZE;
    global_addr_t this_batch_out_addr =
        n_idx * output_shape.c * output_shape.h * output_shape.w * FLT_SIZE;
    OKKERNEL_LOG("%u %u", this_batch_input_addr, this_batch_out_addr);
    if (reuse_output) {
      okk_gdma_32bit_cpy_S2L(output_addr,
                             param->output_addr + this_batch_out_addr,
                             &output_shape, NULL, NULL);
      okk_bdc_mul_C(output_addr, output_addr, 1.0 * (index - 1), &output_shape,
                    NULL, NULL);
    }
    for (int IC_idx = 0; IC_idx < IC_new; IC_idx += IC_step) {
      int this_input_addr =
          IC_idx * 2 * input_shape.h * input_shape.w * FLT_SIZE;
      okk_gdma_32bit_cpy_S2L(input_addr[(IC_idx / IC_step) % 2],
                             param->input_addr + this_batch_input_addr +
                                 this_input_addr,
                             &input_shape, NULL, NULL);
      global_addr_t this_kernel_addr =
          IC_idx * kernel_shape.c * kernel_shape.h * kernel_shape.w * FLT_SIZE;
      okk_gdma_32bit_cpy_S2L(kernel_addr[(IC_idx / IC_step) % 2],
                             param->kernel_addr + this_kernel_addr,
                             &kernel_shape, &kernel_stride, NULL);
      if (is_para) {
        okk_parallel_end();
        is_para = false;
      }
      okk_parallel_start();
      is_para = true;
      okk_bdc_conv2d(output_addr, input_addr[(IC_idx / IC_step) % 2],
                     kernel_addr[(IC_idx / IC_step) % 2], NO_USE, &input_shape,
                     param->OC, param->kernel_h, param->kernel_w, &input_stride,
                     &kernel_stride_2IC, false,
                     reuse_output ? true : (IC_idx == 0 ? false : true),
                     &padding, &stride, &dilation);
    }
    if (is_para) {
      is_para = false;
      okk_parallel_end();
    }
    if (reuse_output) {
      okk_bdc_div_C(output_addr, output_addr, 1.0 * index, &output_shape, NULL,
                    NULL);
    }
    okk_gdma_32bit_cpy_L2S(param->output_addr + this_batch_out_addr,
                           output_addr, &output_shape, NULL, NULL);
  }
}
void conv2d_contest(const void *args) {
  okk_initialize();
  param_t *param = (param_t *)args;
  switch (param->H) {
  case 640:
    case_tune_NH(param, 20);
    break;
  case 384:
  case 512:
    case_tune_NH(param, 16);
    break;
  case 1080:
    case_tune_NH(param, 8);
    break;
    break;
  case 128:
    case_tune_NH(param, 32);
    break;
    // <- NHPP
  case 227:
  case 127:
  case 192:
  case 30: /// 1039
   case 50:
    case_tune_N(param);
    break;
  case 160: // 2360
    conv2_split_N(param);
    break;
  // <- N
  case 28:
    conv2d_split_IC(param, 64, false, 1);
    break;
  case 10:
    conv2d_split_IC(param, 64, false, 1);
    break;
  case 11:
    conv2d_split_IC(param, 32, false, 1);
    break;
  default:
    conv2d_split_IC(param, 16, false, 1);
  }

  // conv2_split_N_H(param);
  // conv2_split_N_IC(param);
  // conv2_split_N(param);
  // if (param->IC >= 512) {
  //   conv2d_split_IC(param);
  // } else {
  //   conv2_split_N(param);
  // }
  OKKERNEL_LOG("\n");
  okk_poll();
}
OKKERNEL_FUNC_REGISTER(conv2d_contest);
