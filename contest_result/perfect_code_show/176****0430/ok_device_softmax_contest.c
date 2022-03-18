#include "bm_atomic.h"
#include "bmlib_runtime.h"
#include "float.h"
#include "okk.h"
#include "math.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a)-1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE 524288
#define LOCAL_BANK_SIZE LOCAL_MEM_SIZE / okk_local_mem_bank_per_npu()
#define NPU_NUM 64
#define NO_USE 0
#define FLT_SIZE 4
#define EXP_TUNE 20
#define LOCAL_ADDR_PTR(x, y) ((float *)okk_local_mem_addr(x, y))
// #define DEBUG 0
#ifndef DEBUG
#undef OKKERNEL_LOG
#define OKKERNEL_LOG(...)                                                      \
  {}
#endif

#ifdef NOOUTPUT
#endif

typedef struct {
  int N, C, H, W;
  unsigned long long output_addr;
  unsigned long long input_addr;
} __attribute__((packed)) param_t;

// case 1
void softmax_64npu_only_c(const param_t *param) {
  dim4 shape = {1, param->N, param->C, param->H * param->W};
  dim4 stride;
  //
  local_addr_t output_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  local_addr_t input_addr = stride.n * shape.n * FLT_SIZE;

  OKKERNEL_LOG("Mem used: %u/%u", input_addr * 2, LOCAL_MEM_SIZE);

  okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);

  okk_bdc_taylor_exp(input_addr, input_addr, &shape, EXP_TUNE);
  okk_bdc_avg_pool2d(output_addr, input_addr, &shape, param->C, 1, NULL, NULL);
  okk_bdc_mul_C(output_addr, output_addr, param->C, &shape, NULL, NULL);

  dim4 stride_avg = {1, shape.w, 0, 1};
  okk_bdc_div(input_addr, input_addr, output_addr, &shape, &stride, &stride,
              &stride_avg);

  okk_gdma_32bit_cpy_L2S(param->output_addr, input_addr, &shape, NULL, NULL);
}

// case 0 // not best for case 2/3
void softmax_64npu_rest(const param_t *param) {
  const int IC_new = (param->C + 1) / 2;
  dim4 shape = {1, param->C, param->H, param->W};
  dim4 stride;
  dim4 kernel_shape = {IC_new, param->C, 1, 1 * 2};
  dim4 kernel_stride = {0, 0, 0, 1};
  dim4 kernel_shape_2IC = {IC_new, param->C, 1, 1};
  dim4 kernel_stride_2IC = {0, 0, 0, 0};
  //
  local_addr_t output_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  local_addr_t input_addr =
      DIV_UP(stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) * LOCAL_BANK_SIZE;
  local_addr_t kernel_addr =
      DIV_UP(input_addr + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;

  OKKERNEL_LOG("Mem used: %u/%u", kernel_addr + 4, LOCAL_MEM_SIZE);

  for (int i = 0; i < param->N; i++) {
    okk_parallel_start();
    okk_bdc_32bit_set_C(kernel_addr, (x32)(1.0f), &kernel_shape,
                        &kernel_stride);
    okk_gdma_32bit_cpy_S2L(input_addr,
                           param->input_addr +
                               i * param->C * param->H * param->W * FLT_SIZE,
                           &shape, NULL, NULL);
    okk_parallel_end();
    okk_bdc_taylor_exp(input_addr, input_addr, &shape, EXP_TUNE);

    okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, &shape,
                   param->C, kernel_shape_2IC.h, kernel_shape_2IC.w, NULL,
                   &kernel_stride_2IC, false, false, NULL, NULL, NULL);

    okk_bdc_div(output_addr, input_addr, output_addr, &shape, &stride, &stride,
                &stride);

    okk_gdma_32bit_cpy_L2S(param->output_addr +
                               i * param->C * param->H * param->W * FLT_SIZE,
                           output_addr, &shape, NULL, NULL);
  }
}

// case 4
void softmax_64npu_NC11_large_N(const param_t *param) {
  int all = param->N;
  int channels = NPU_NUM;
  int step = DIV_UP(all, channels);
  OKKERNEL_LOG("\nStep: %d", step);
  //
  dim4 shape = {1, channels, step, param->C};
  dim4 stride;
  //
  local_addr_t input_addr[2] = {0, LOCAL_MEM_SIZE / 2};
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  ;
  local_addr_t output_addr[2] = {
      DIV_UP(stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) * LOCAL_BANK_SIZE,
      DIV_UP(LOCAL_MEM_SIZE / 2 + stride.n * shape.n * FLT_SIZE,
             LOCAL_BANK_SIZE) *
          LOCAL_BANK_SIZE};
  local_addr_t exp_output_addr[2] = {
      DIV_UP(output_addr[0] + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
          LOCAL_BANK_SIZE,
      DIV_UP(output_addr[1] + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
          LOCAL_BANK_SIZE};
  local_addr_t avg_addr[2] = {
      DIV_UP(exp_output_addr[0] + stride.n * shape.n * FLT_SIZE,
             LOCAL_BANK_SIZE) *
          LOCAL_BANK_SIZE,
      DIV_UP(exp_output_addr[1] + stride.n * shape.n * FLT_SIZE,
             LOCAL_BANK_SIZE) *
          LOCAL_BANK_SIZE};
  dim4 avg_shape = {1, channels, step, 1};
  dim4 avg_stride;
  okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  local_addr_t MEM_USE = avg_addr[0] + avg_stride.n * avg_shape.n * FLT_SIZE;
  OKKERNEL_LOG("\nMem used: %u/%u", MEM_USE, LOCAL_MEM_SIZE);
  // comment for no-use codes for cases
  // if (MEM_USE > LOCAL_MEM_SIZE / 2) {
  //   OKKERNEL_LOG("\nCut");
  //   channels *= 2;
  //   step = DIV_UP(all, channels);
  //   OKKERNEL_ASSERT(step > 0);
  //   shape.h = step;
  //   avg_shape.h = step;
  //   okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  //   okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  //   avg_addr[0] = stride.n * shape.n * FLT_SIZE;
  //   avg_addr[1] = LOCAL_MEM_SIZE / 2 + stride.n * shape.n * FLT_SIZE;
  //   MEM_USE = avg_addr[0] + avg_stride.n * avg_shape.n * FLT_SIZE;
  // }
  avg_stride.w = 0;
  for (int idx = 0; idx < all + 2 * step * channels; idx += step * channels) {
    okk_parallel_start();
    if (idx > step * channels) {
      okk_gdma_32bit_cpy_L2S(
          param->output_addr +
              (idx - step * channels * 2) * param->C * channels * FLT_SIZE,
          output_addr[(idx / step * channels) % 2], &shape, NULL, NULL);
    }
    if (idx < all) {
      okk_gdma_32bit_cpy_S2L(input_addr[(idx / (step * channels)) % 2],
                             param->input_addr +
                                 idx * channels * param->C * FLT_SIZE,
                             &shape, NULL, NULL);
    }

    if (idx > 0 && idx < all + step * channels) {
      int index = (idx / (step * channels) + 1) % 2;
      okk_bdc_taylor_exp(exp_output_addr[index], input_addr[index], &shape,
                         EXP_TUNE);
      okk_bdc_avg_pool2d(avg_addr[index], exp_output_addr[index], &shape, 1,
                         param->C, NULL, NULL);

      okk_bdc_mul_C(avg_addr[index], avg_addr[index], param->C, &shape, NULL,
                    NULL);

      okk_bdc_div(output_addr[index], exp_output_addr[index], avg_addr[index], &shape,
                  NULL, NULL, &avg_stride);
    }
    okk_parallel_end();
  }
}

// case 1
void softmax_64npu_NC11_large_C(const param_t *param) {
  OKKERNEL_LOG("Bank: %u/%u, Local: %u/%u, %u", okk_local_mem_bank_per_npu(),
               okk_local_mem_bank_per_npu() / 4, LOCAL_MEM_SIZE,
               LOCAL_MEM_SIZE / 4,
               LOCAL_MEM_SIZE / okk_local_mem_bank_per_npu());
  int h_size = 2, w_size = 0;
  while (param->C % h_size) {
    h_size++;
  }
  w_size = param->C / h_size;
  dim4 shape = {1, param->N, 1, param->C};
  dim4 stride;
  OKKERNEL_LOG("param: %d %d %d %d", shape.n, shape.c, shape.h, shape.w);

  //
  local_addr_t input_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  local_addr_t output_addr =
      DIV_UP(stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) * LOCAL_BANK_SIZE;
  local_addr_t avg_addr =
      DIV_UP(output_addr + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  ;
  dim4 avg_shape = {1, param->N, 1, 1};
  dim4 avg_stride;
  okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  local_addr_t MEM_USE = avg_addr + avg_stride.n * avg_shape.n * FLT_SIZE;
  OKKERNEL_LOG("\nMem used: %u/%u", MEM_USE, LOCAL_MEM_SIZE);

  // OKKERNEL_ASSERT(MEM_USE > LOCAL_MEM_SIZE);
  okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);

  okk_bdc_taylor_exp(input_addr, input_addr, &shape, EXP_TUNE);

  shape.h = h_size;
  shape.w = w_size;
  okk_bdc_avg_pool2d(avg_addr, input_addr, &shape, h_size, w_size, NULL, NULL);
  shape.h = 1;
  shape.w = param->C;
  okk_bdc_mul_C(avg_addr, avg_addr, param->C, &shape, NULL, NULL);

  avg_stride.h = 0;
  avg_stride.w = 0;
  okk_bdc_div(input_addr, input_addr, avg_addr, &shape, &stride, &stride,
              &avg_stride);

  okk_gdma_32bit_cpy_L2S(param->output_addr, input_addr, &shape, NULL, NULL);
}

void softmax_64npu_NC11_large_C_PingPong(const param_t *param) {
  int h_size = 409, w_size = 0;
  while (param->C % h_size) {
    h_size++;
  }
  w_size = param->C / h_size;
  dim4 shape = {1, NPU_NUM, 1, param->C};
  dim4 shape2 = {1, param->N - NPU_NUM, 1, param->C};
  dim4 stride;
  OKKERNEL_LOG("param: %d %d %d %d", shape.n, shape.c, shape.h, shape.w);

  //
  local_addr_t input_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  local_addr_t output_addr =
      DIV_UP(stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) * LOCAL_BANK_SIZE;
  local_addr_t exp_output_addr =
      DIV_UP(output_addr + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  local_addr_t input_addr2 =
      DIV_UP(exp_output_addr + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  local_addr_t output_addr2 =
      DIV_UP(input_addr2 + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  local_addr_t exp_output_addr2 =
      DIV_UP(output_addr2 + stride.n * shape.n * FLT_SIZE, LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  local_addr_t avg_addr =
      DIV_UP(exp_output_addr2 + stride.n * shape.n * FLT_SIZE,
             LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  dim4 avg_shape = {1, NPU_NUM, 1, 1};
  dim4 avg_shape2 = {1, param->N - NPU_NUM, 1, 1};
  dim4 avg_stride;
  okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  local_addr_t avg_addr2 =
      DIV_UP(avg_addr + avg_stride.n * avg_shape.n * FLT_SIZE,
             LOCAL_BANK_SIZE) *
      LOCAL_BANK_SIZE;
  local_addr_t MEM_USE = avg_addr + avg_stride.n * avg_shape.n * FLT_SIZE;
  OKKERNEL_LOG("\nMem used: %u/%u", MEM_USE, LOCAL_MEM_SIZE);

  // OKKERNEL_ASSERT(MEM_USE > LOCAL_MEM_SIZE);
  okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
  okk_parallel_start();
  okk_gdma_32bit_cpy_S2L(input_addr2,
                         param->input_addr + NPU_NUM * param->C * FLT_SIZE,
                         &shape2, NULL, NULL);

  okk_bdc_taylor_exp(exp_output_addr, input_addr, &shape, EXP_TUNE);

  shape.h = h_size;
  shape.w = w_size;
  okk_bdc_avg_pool2d(avg_addr, exp_output_addr, &shape, h_size, w_size, NULL,
                     NULL);
  shape.h = 1;
  shape.w = param->C;
  okk_bdc_mul_C(avg_addr, avg_addr, param->C, &shape, NULL, NULL);

  avg_stride.h = 0;
  avg_stride.w = 0;
  okk_bdc_div(output_addr, exp_output_addr, avg_addr, &shape, &stride, &stride,
              &avg_stride);
  okk_parallel_end();
  okk_parallel_start();
  okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
  okk_bdc_taylor_exp(exp_output_addr2, input_addr2, &shape2, EXP_TUNE);

  shape2.h = h_size;
  shape2.w = w_size;
  okk_bdc_avg_pool2d(avg_addr2, exp_output_addr2, &shape2, h_size, w_size, NULL,
                     NULL);
  shape2.h = 1;
  shape2.w = param->C;
  okk_bdc_mul_C(avg_addr2, avg_addr2, param->C, &shape2, NULL, NULL);

  avg_stride.h = 0;
  avg_stride.w = 0;
  okk_bdc_div(output_addr2, exp_output_addr2, avg_addr2, &shape2, &stride,
              &stride, &avg_stride);
  okk_parallel_end();
  okk_gdma_32bit_cpy_L2S(param->output_addr + NPU_NUM * param->C * FLT_SIZE,
                         output_addr2, &shape2, NULL, NULL);
}

void softmax_64npu_NCHW(const param_t *param) {
  OKKERNEL_LOG("use nchw");
  int HW = param->H * param->W, HW_FULL = param->H * param->W;
  dim4 shape = {1, param->N, param->C, HW}; // for 1 batch
  dim4 stride;
  OKKERNEL_LOG("param: %d %d %d %d", shape.n, shape.c, shape.h, shape.w);
  //
  local_addr_t input_addr[2] = {0, 0};
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  local_addr_t output_addr[2] = {stride.n * shape.n * FLT_SIZE, 0};
  local_addr_t avg_addr[2] = {2 * stride.n * shape.n * FLT_SIZE, 0};
  dim4 avg_shape = {1, param->N, 1, HW};
  dim4 avg_stride;
  okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  dim4 g_stride = {param->N * param->C * param->H * param->W,
                   param->C * param->H * param->W, param->H * param->W, 1};
  while (avg_addr[0] + avg_stride.n * avg_shape.n * FLT_SIZE >
         LOCAL_MEM_SIZE / 2) {
    HW /= param->H;
    shape.w = HW;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    output_addr[0] = stride.n * shape.n * FLT_SIZE;
    avg_addr[0] = 2 * stride.n * shape.n * FLT_SIZE;
    avg_shape.w = HW;
    okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  }
  input_addr[1] = input_addr[0] + LOCAL_MEM_SIZE / 2;
  output_addr[1] = output_addr[0] + LOCAL_MEM_SIZE / 2;
  avg_addr[1] = avg_addr[0] + LOCAL_MEM_SIZE / 2;

  OKKERNEL_LOG("\nPer Mem Used: %u/%u",
               avg_addr[0] + avg_stride.n * avg_shape.n * FLT_SIZE,
               LOCAL_MEM_SIZE);

  for (int idx = 0; idx < HW_FULL + 2 * HW; idx += HW) {
    okk_parallel_start();
    OKKERNEL_LOG("idx: %d ", idx);
    if (idx < HW_FULL) {
      OKKERNEL_LOG("move into %d", (idx / HW) % 2);
      shape.w = MIN(HW, HW_FULL - idx);
      okk_gdma_32bit_cpy_S2L(input_addr[(idx / HW) % 2],
                             param->input_addr + idx * FLT_SIZE, &shape,
                             &stride, &g_stride);
    }
    if (idx > 0 && idx < HW_FULL + HW) {
      int index = ((idx - HW) / HW) % 2;
      OKKERNEL_LOG("\nmat at %d  %f %f %f %f %f %f", index,
                   *LOCAL_ADDR_PTR(0, input_addr[index]),
                   *LOCAL_ADDR_PTR(0, input_addr[index] + 4),
                   *LOCAL_ADDR_PTR(0, input_addr[index] + 8),
                   *LOCAL_ADDR_PTR(0, input_addr[index] + 12),
                   *LOCAL_ADDR_PTR(0, input_addr[index] + 16),
                   *LOCAL_ADDR_PTR(0, input_addr[index] + 20));
      okk_bdc_taylor_exp(output_addr[index], input_addr[index], &shape,
                         EXP_TUNE);
      // OKKERNEL_LOG("%d %d %d %d", avg_stride.n, avg_stride.c, avg_stride.h,
      //              avg_stride.w);
      okk_bdc_avg_pool2d(avg_addr[index], output_addr[index], &shape, param->C,
                         1, NULL, NULL);
      OKKERNEL_LOG("\navg %f %f %f %f %f %f",
                   *LOCAL_ADDR_PTR(0, avg_addr[index]),
                   *LOCAL_ADDR_PTR(0, avg_addr[index] + 4),
                   *LOCAL_ADDR_PTR(0, avg_addr[index] + 8),
                   *LOCAL_ADDR_PTR(1, avg_addr[index]),
                   *LOCAL_ADDR_PTR(1, avg_addr[index] + 4),
                   *LOCAL_ADDR_PTR(1, avg_addr[index] + 8));

      okk_bdc_mul_C(avg_addr[index], avg_addr[index], param->C, &shape, NULL,
                    NULL);
      avg_stride.h = 0;
      avg_stride.w = 1;
      okk_bdc_div(output_addr[index], output_addr[index], avg_addr[index],
                  &shape, &stride, &stride, &avg_stride);
    }
    if (idx > HW) {
      OKKERNEL_LOG("move from %d", (idx / HW) % 2);

      shape.w = MIN(HW, HW_FULL + 2 * HW - idx);
      okk_gdma_32bit_cpy_L2S(param->output_addr + (idx - 2 * HW) * FLT_SIZE,
                             output_addr[idx / HW % 2], &shape, &g_stride,
                             &stride);
    }
    okk_parallel_end();
  }
}

void softmax_64npu_hardcode(const param_t *param) {
  OKKERNEL_LOG("use hardcode");
  int HW = DIV_UP(param->H * param->W, NPU_NUM / 2),
      HW_FULL = param->H * param->W;
  dim4 shape = {1, 1, param->C, HW}; // for 1 batch
  dim4 stride;
  OKKERNEL_LOG("param: %d %d %d %d", shape.n, shape.c, shape.h, shape.w);
  //
  local_addr_t input_addr = 0;
  okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
  local_addr_t avg_addr = stride.n * shape.n * FLT_SIZE;
  dim4 avg_shape = {1, 1, 1, HW};
  dim4 avg_stride;
  okk_128_byte_aligned_stride_for_32bit(&avg_stride, 0, &avg_shape);
  dim4 g_stride = {param->N * param->C * param->H * param->W,
                   param->C * param->H * param->W, param->H * param->W, 1};
  int input_batch_size = param->C * param->H * param->W;
  OKKERNEL_LOG("\nPer Mem Used: %u/%u %d %d",
               avg_addr + avg_stride.n * avg_shape.n * FLT_SIZE, LOCAL_MEM_SIZE,
               HW, NPU_NUM / param->N);
  for (int n = 0; n < 2; n++) {
    // OKKERNEL_LOG("In w is %d %d", shape.w,(n * DIV_UP(HW_FULL, HW) + idx /
    // HW));
    int temp = g_stride.c;
    shape.c = DIV_UP(HW_FULL, HW);
    // OKKERNEL_LOG(
    //     "\nshape: %d %d %d %d\nstride %d %d %d %d\ngstride %d %d %d %d",
    //     shape.n, shape.c, shape.h, shape.w, stride.n, stride.c, stride.h,
    //     stride.w, g_stride.n, g_stride.c, g_stride.h, g_stride.w);
    g_stride.c = HW;
    okk_gdma_32bit_cpy_S2L(input_addr +
                               LOCAL_MEM_SIZE * n * DIV_UP(HW_FULL, HW),
                           param->input_addr + FLT_SIZE * n * input_batch_size,
                           &shape, &stride, &g_stride);
    shape.w = HW;
    g_stride.c = temp;
  }
  // for (int n = 0; n < param->N; n++) {
  //   for (int idx = 0; idx < HW_FULL; idx += HW) {
  //     shape.w = MIN(HW, HW_FULL - idx);
  //     // OKKERNEL_LOG("In w is %d %d", shape.w,(n * DIV_UP(HW_FULL, HW) + idx
  //     /
  //     // HW));
  //     okk_gdma_32bit_cpy_S2L(
  //         input_addr + LOCAL_MEM_SIZE * (n * DIV_UP(HW_FULL, HW) + idx / HW),
  //         param->input_addr + FLT_SIZE * (n * g_stride.c + idx), &shape,
  //         &stride, &g_stride);
  //     shape.w = HW;
  //   }
  // }
  okk_parallel_start();
  for (int n = 2; n < 4; n++) {
    // OKKERNEL_LOG("In w is %d %d", shape.w,(n * DIV_UP(HW_FULL, HW) + idx /
    // HW));
    int temp = g_stride.c;
    shape.c = DIV_UP(HW_FULL, HW);
    // OKKERNEL_LOG(
    //     "\nshape: %d %d %d %d\nstride %d %d %d %d\ngstride %d %d %d %d",
    //     shape.n, shape.c, shape.h, shape.w, stride.n, stride.c, stride.h,
    //     stride.w, g_stride.n, g_stride.c, g_stride.h, g_stride.w);
    g_stride.c = HW;
    okk_gdma_32bit_cpy_S2L(input_addr + LOCAL_MEM_SIZE / 2 +
                               LOCAL_MEM_SIZE * (n - 2) * DIV_UP(HW_FULL, HW),
                           param->input_addr + FLT_SIZE * n * input_batch_size,
                           &shape, &stride, &g_stride);
    shape.w = HW;
    g_stride.c = temp;
  }

  shape.c = NPU_NUM;
  okk_bdc_taylor_exp(input_addr, input_addr, &shape, EXP_TUNE);
  okk_bdc_avg_pool2d(avg_addr, input_addr, &shape, param->C, 1, NULL, NULL);
  okk_bdc_mul_C(avg_addr, avg_addr, param->C, &shape, NULL, NULL);
  avg_stride.h = 0;
  avg_stride.w = 1;
  okk_bdc_div(input_addr, input_addr, avg_addr, &shape, &stride, &stride,
              &avg_stride);
  shape.c = 1;
  okk_parallel_end();

  okk_parallel_start();
  shape.c = NPU_NUM;
  okk_bdc_taylor_exp(input_addr + LOCAL_MEM_SIZE / 2,
                     input_addr + LOCAL_MEM_SIZE / 2, &shape, EXP_TUNE);
  okk_bdc_avg_pool2d(avg_addr + LOCAL_MEM_SIZE / 2,
                     input_addr + LOCAL_MEM_SIZE / 2, &shape, param->C, 1, NULL,
                     NULL);
  okk_bdc_mul_C(avg_addr + LOCAL_MEM_SIZE / 2, avg_addr + LOCAL_MEM_SIZE / 2,
                param->C, &shape, NULL, NULL);
  avg_stride.h = 0;
  avg_stride.w = 1;
  okk_bdc_div(input_addr + LOCAL_MEM_SIZE / 2, input_addr + LOCAL_MEM_SIZE / 2,
              avg_addr + LOCAL_MEM_SIZE / 2, &shape, &stride, &stride,
              &avg_stride);
  shape.c = 1;

  for (int n = 0; n < 2; n++) {
    // OKKERNEL_LOG("In w is %d %d", shape.w,(n * DIV_UP(HW_FULL, HW) + idx /
    // HW));
    int temp = g_stride.c;
    g_stride.c = HW;
    shape.w = HW;
    shape.c = DIV_UP(HW_FULL, HW) - 1;
    OKKERNEL_LOG(
        "\nshape: %d %d %d %d\nstride %d %d %d %d\ngstride %d %d %d %d",
        shape.n, shape.c, shape.h, shape.w, stride.n, stride.c, stride.h,
        stride.w, g_stride.n, g_stride.c, g_stride.h, g_stride.w);
    okk_gdma_32bit_cpy_L2S(param->output_addr + FLT_SIZE * n * input_batch_size,
                           input_addr +
                               LOCAL_MEM_SIZE * n * DIV_UP(HW_FULL, HW),
                           &shape, &g_stride, &stride);
    shape.c = 1;
    shape.w = HW_FULL % HW;
    okk_gdma_32bit_cpy_L2S(
        param->output_addr +
            FLT_SIZE * (n * input_batch_size + HW_FULL - HW_FULL % HW),
        input_addr + LOCAL_MEM_SIZE * ((n + 1) * DIV_UP(HW_FULL, HW) - 1),
        &shape, &g_stride, &stride);
    g_stride.c = temp;
  }
  okk_parallel_end();
  for (int n = 2; n < 4; n++) {
    // OKKERNEL_LOG("In w is %d %d", shape.w,(n * DIV_UP(HW_FULL, HW) + idx /
    // HW));
    int temp = g_stride.c;
    g_stride.c = HW;
    shape.w = HW;
    shape.c = DIV_UP(HW_FULL, HW) - 1;
    OKKERNEL_LOG(
        "\nshape: %d %d %d %d\nstride %d %d %d %d\ngstride %d %d %d %d",
        shape.n, shape.c, shape.h, shape.w, stride.n, stride.c, stride.h,
        stride.w, g_stride.n, g_stride.c, g_stride.h, g_stride.w);
    okk_gdma_32bit_cpy_L2S(param->output_addr + FLT_SIZE * n * input_batch_size,
                           input_addr + LOCAL_MEM_SIZE / 2 +
                               LOCAL_MEM_SIZE * (n - 2) * DIV_UP(HW_FULL, HW),
                           &shape, &g_stride, &stride);
    shape.c = 1;
    shape.w = HW_FULL % HW;
    okk_gdma_32bit_cpy_L2S(
        param->output_addr +
            FLT_SIZE * (n * input_batch_size + HW_FULL - HW_FULL % HW),
        input_addr + LOCAL_MEM_SIZE / 2 +
            LOCAL_MEM_SIZE * ((n - 1) * DIV_UP(HW_FULL, HW) - 1),
        &shape, &g_stride, &stride);
    g_stride.c = temp;
  }
  // for (int n = 0; n < param->N; n++) {
  //   for (int idx = 0; idx < HW_FULL; idx += HW) {
  //     shape.w = MIN(HW, HW_FULL - idx);
  //     // OKKERNEL_LOG("Out w is %d %d", shape.w,(n * DIV_UP(HW_FULL, HW) +
  //     idx /
  //     // HW));
  //     okk_gdma_32bit_cpy_L2S(
  //         param->output_addr + FLT_SIZE * (n * g_stride.c + idx),
  //         input_addr + LOCAL_MEM_SIZE * (n * DIV_UP(HW_FULL, HW) + idx / HW),
  //         &shape, &g_stride, &stride);
  //     shape.w = HW;
  //   }
  // }
}

void softmax_contest(const void *args) {
  okk_initialize();
  param_t *param = (param_t *)args;
  (void)(param);

  if (1 == param->H && 1 == param->W)
    if (param->C >= param->N)
      if (param->N == 1)
        softmax_64npu_NC11_large_C(param); // case 1
      else
        softmax_64npu_NC11_large_C_PingPong(param);
    else
      softmax_64npu_NC11_large_N(param); //
  else if (1 == param->N) {
    softmax_64npu_rest(param);
  } else {
    // softmax_64npu_NCHW(param);
    softmax_64npu_hardcode(param);
  }
  // softmax_64npu_only_c(param);
  // softmax_64npu_rest(param);
  // softmax_64npu_hardcode(param);
  OKKERNEL_LOG("\n");
  okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
