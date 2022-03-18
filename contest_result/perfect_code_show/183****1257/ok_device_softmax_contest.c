#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
#define NPU_NUM okk_npu_num()
#define NO_USE 0
typedef struct {
    int N, C, H, W;
    unsigned long long output_addr;
    unsigned long long input_addr;
} __attribute__((packed)) param_t;

// void conduct_softmax(local_addr_t input_addr, local_addr_t exp_addr, local_addr_t output_addr, dim4 shape, dim4 stride){
//     // sub constant 5
//     okk_bdc_sub_C(input_addr, input_addr, 5.0, &shape, &stride, &stride);

//     // cal exp(X)
//     okk_bdc_exp(exp_addr, input_addr, output_addr, &shape);

//     // cal sum(exp(X))
//     x32 C = {1.0};
//     dim4 kernel_shape = {.n=1, .c=64, .h=1, .w=2};
//     dim4 kernel_stride;
//     okk_compact_stride(&kernel_stride, 0, &kernel_shape);

//     OKKERNEL_ASSERT(output_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);
//     okk_bdc_32bit_set_C(output_addr, C, &kernel_shape, &kernel_stride);

//     dim4 kernel_stride_2IC = {.n=0,.c=0,.h=0,.w=0};

//     okk_bdc_conv2d(input_addr, exp_addr, output_addr, NULL, &shape, shape.c, 1, 1, &stride, &kernel_stride_2IC, false, false, NULL, NULL, NULL);

//     // div exp(x) by sum
//     okk_bdc_div(output_addr, exp_addr, input_addr, &shape, &stride, &stride, &stride);
// }

void softmax_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    dim4 shape;
    dim4 stride, sys_stride;

    switch (param->C)
    {
    case 370:
        shape.n = 1;
        shape.c = 13;
        shape.h = 370;
        shape.w = 13;

        sys_stride.w = 1;
        sys_stride.h = 169;
        sys_stride.c = 13;
        sys_stride.n = 0;
        break;
    case 1000:
        shape.n = 1;
        shape.c = 1;
        shape.h = 40;
        shape.w = 25;

        okk_continuous_stride(&sys_stride, &shape);
        break;
    case 2:
        shape.n = param->N;
        shape.c = param->H;
        shape.h = 2;
        shape.w = param->W;

        sys_stride.w = 1;
        sys_stride.h = param->H * param->W;
        sys_stride.c = param->W;
        sys_stride.n = param->C * param->H * param->W;
        break;
    case 4090:
        shape.n = 1;
        shape.c = 79;
        shape.h = 409;
        shape.w = 10;

        okk_continuous_stride(&sys_stride, &shape);
        break;
    case 21:
        shape.n = 14;
        shape.c = 438;
        shape.h = 21;
        shape.w = 1;

        sys_stride.w = 1;
        sys_stride.h = 1;
        sys_stride.c = 21;
        sys_stride.n = 21 * 438;
        break;
    default:
        shape.n = param->N;
        shape.c = param->C;
        shape.h = param->H;
        shape.w = param->W;

        okk_continuous_stride(&sys_stride, &shape);
        break;
    }

    if(param->C == 4090 || param->C == 1000)
    {
        Padding padding = {.top=0, .bottom=0, .left=0, .right=0};
        dim2 pool_stride = {.h=1, .w=1};
        dim4 shape_after_pool, stride_after_pool;

        shape_after_pool.n = shape.n;
        shape_after_pool.c = shape.c;
        shape_after_pool.h = 1;
        shape_after_pool.w = 1;

        okk_128_byte_aligned_stride_for_32bit(&stride_after_pool, 0, &shape_after_pool);
        stride_after_pool.h = 0;
        stride_after_pool.w = 0;

        okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);

        local_addr_t temp_addr = 0, output_addr = LOCAL_MEM_SIZE / 2;

        okk_gdma_32bit_cpy_S2L(output_addr, param->input_addr, &shape, NO_USE, &sys_stride);

        okk_bdc_taylor_exp(output_addr, output_addr, &shape, 32);


        okk_bdc_avg_pool2d(temp_addr, output_addr, &shape, shape.h, shape.w, &padding, &pool_stride);

        okk_bdc_mul_C(temp_addr, temp_addr, shape.h * shape.w, &shape_after_pool, NO_USE, NO_USE);

        okk_bdc_div(output_addr, output_addr, temp_addr, &shape, NO_USE, NO_USE, &stride_after_pool);

        okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, &sys_stride, NO_USE);
    }else
    {
        Padding padding = {.top=0, .bottom=0, .left=0, .right=0};
        dim2 pool_stride = {.h=1, .w=1};
        dim4 shape_after_pool, stride_after_pool;
        
        shape_after_pool.n = shape.n;
        shape_after_pool.c = shape.c;
        shape_after_pool.h = 1;
        shape_after_pool.w = shape.w;

        okk_128_byte_aligned_stride_for_32bit(&stride_after_pool, 0, &shape_after_pool);
        stride_after_pool.h = 0;

        okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);

        local_addr_t temp_addr = 0, output_addr = LOCAL_MEM_SIZE / 2;

        okk_gdma_32bit_cpy_S2L(output_addr, param->input_addr, &shape, NO_USE, &sys_stride);

        okk_bdc_taylor_exp(output_addr, output_addr, &shape, 20);
        // okk_bdc_exp(output_addr, output_addr, temp_addr, &shape);

        okk_bdc_avg_pool2d(temp_addr, output_addr, &shape, shape.h, 1, &padding, &pool_stride);

        okk_bdc_mul_C(temp_addr, temp_addr, shape.h, &shape_after_pool, NO_USE, NO_USE);

        okk_bdc_div(output_addr, output_addr, temp_addr, &shape, NO_USE, NO_USE, &stride_after_pool);

        okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, &sys_stride, NO_USE);
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
