#include "okk.h"
#include "bm_atomic.h"
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

#if 0
void softmax_doublec(const void* args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = 2, .c = param->H, .h = param->W, .w = 1 };
    int area = 2 * param->H * param->W * sizeof(float);
    dim4 stride;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    dim4 stride_reduce = { .n = 0, .c = stride.c, .h = stride.h, .w = 1 };
    int size = shape.n * stride.n * sizeof(float);
    local_addr_t input_addrs[2] = {0, size};
    local_addr_t output_addrs[2] = {2*size, 3*size};
    local_addr_t work_addr = 4*size;
    okk_gdma_32bit_cpy_S2L(input_addrs[0], param->input_addr, &shape, NULL, NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addrs[1], param->input_addr + area, &shape, NULL, NULL);
    okk_bdc_exp_tunable(output_addrs[0], input_addrs[0], work_addr, &shape, 5);
    okk_bdc_add(work_addr, output_addrs[0], output_addrs[0] + stride.n * sizeof(float), &shape, NULL, NULL, NULL);
    okk_bdc_div(output_addrs[0], output_addrs[0], work_addr, &shape, NULL, NULL, &stride_reduce);
    okk_parallel_end();
    int i = 1;
    for (; i < param->N-1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(input_addrs[(i+1)%2], param->input_addr + (i+1) * area, &shape, NULL, NULL);
        okk_bdc_exp_tunable(output_addrs[i%2], input_addrs[i%2], work_addr, &shape, 5);
        okk_bdc_add(work_addr, output_addrs[i%2], output_addrs[i%2] + stride.n * sizeof(float), &shape, NULL, NULL, NULL);
        okk_bdc_div(output_addrs[i%2], output_addrs[i%2], work_addr, &shape, NULL, NULL, &stride_reduce);
        okk_gdma_32bit_cpy_L2S(param->output_addr + (i-1) * area, output_addrs[(i-1)%2], &shape, NULL, NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_exp_tunable(output_addrs[i%2], input_addrs[i%2], work_addr, &shape, 5);
    okk_bdc_add(work_addr, output_addrs[i%2], output_addrs[i%2] + stride.n * sizeof(float), &shape, NULL, NULL, NULL);
    okk_bdc_div(output_addrs[i%2], output_addrs[i%2], work_addr, &shape, NULL, NULL, &stride_reduce);
    okk_gdma_32bit_cpy_L2S(param->output_addr + (i-1) * area, output_addrs[(i-1)%2], &shape, NULL, NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(param->output_addr + i * area, output_addrs[i%2], &shape, NULL, NULL);
    okk_poll();
}
void softmax_c(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = param->N, .c = 1, .h = 1, .w = param->C };
    dim4 stride;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    dim4 shape_reduce = { .n = 1, .c = 1, .h = 1, .w = 1 };
    int size = shape.n * stride.n * sizeof(float);
    local_addr_t input_addr = 0;
    local_addr_t output_addr = size;
    local_addr_t work_addr = size * 2;
    dim4 stride_reduce = { .n = 0, .0, .h = 0, .w = 0 };

    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, 7);
    //okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape, 2);
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    okk_poll();
}

void softmax_nc(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = 1, .c = param->N, .h = 1, .w = param->C };
    int seri = 14;
    int Tc = 2; 
    int Sc = DIV_UP(shape.c, Tc);
    if (shape.w > 2048) {
    	seri = 3;
        shape.h = 10;
        shape.w = shape.w / shape.h;
    }
    int Tcc = Tc - 1;
    int Rc = shape.c - Tcc * Sc;
    shape.c = Sc;
    dim4 stride;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    int size = shape.n * stride.n * sizeof(float);
    local_addr_t input_addr = 0, input_addr2 = size;
    local_addr_t output_addr = size * 2, output_addr2 = size * 3;
    local_addr_t work_addr = size * 4;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    dim4 stride_reduce = { .n = stride.n, .c = stride.c, .h = 0, .w = 0 };
    dim4 shape_reduce = { .n = shape.n, .c = shape.c, .h = 1, .w = 1 };
    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr2, param->input_addr + Sc * param->C * sizeof(float), &shape, NULL, NULL);
#ifdef TAYLOR
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, seri);
#else
    okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape, 4);
#endif
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();
    int i = 1;
    for (; i < Tcc; i++) {
            okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(input_addrs[(i+1)%2], param->input_addr + (i+1) * Sc * param->C * sizeof(float), &shape, NULL, NULL);
            okk_bdc_exp_tunable(output_addrs[i%2], input_addrs[i%2], work_addr, &shape, 4);
            okk_bdc_avg_pool2d(work_addr, output_addrs[i%2], &shape, shape.h, shape.w, NULL, NULL);
            okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
            okk_bdc_div(output_addrs[i%2], output_addrs[i%2], work_addr, &shape, &stride, &stride, &stride_reduce);
            okk_gdma_32bit_cpy_L2S(param->output_addr + (i-1) * Sc * param->C * sizeof(float), output_addrs[(i-1)%2], &shape, NULL, NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(param->output_addr + (i-1) * Sc * param->C * sizeof(float), output_addrs[(i-1)%2], &shape, NULL, NULL);
    shape.c = Rc;
    shape_reduce.c = Rc;
    okk_bdc_exp_tunable(output_addrs[i%2], input_addrs[i%2], work_addr, &shape, 4);
    okk_bdc_avg_pool2d(work_addr, output_addrs[i%2], &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addrs[i%2], output_addrs[i%2], work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(param->output_addr + i * Sc * param->C * sizeof(float), output_addrs[i%2], &shape, NULL, NULL);
    okk_poll();
}

void softmax_universal(const void *args) {
    param_t *param = (param_t *)args;
    int Tc = 2;
    int Sc = param->C / Tc;
    dim4 shape = { .n = param->N, .c = param->H, .h = Sc, .w = param->W };
    dim4 input_stride = { .n = param->H*param->W*param->C, .c = param->W, .h = param->H * param->W, .w = 1 };
    dim4 stride;
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    int size = shape.n * stride.n * sizeof(float);
    local_addr_t input_addr = 0;
    local_addr_t output_addr = size;
    local_addr_t work_addr = size * 2;
    local_addr_t input_addr2 = size * 3;
    local_addr_t output_addr2 = size * 4;
    local_addr_t work_addr2 = size * 5;

    dim4 stride_reduce = { .n = stride.n, .c = stride.c, .h = 0, .w = 1 };
    dim4 shape_reduce = { .n = shape.n, .c = shape.c, .h = 1, .w = shape.w };

    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, &input_stride);

    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr2, param->input_addr + Sc * param->H * param->W * sizeof(float), &shape, NULL, &input_stride);
    //okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape, 3);
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, 9);
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, 1, NULL, NULL);
    //okk_bdc_mul_C(work_addr, work_addr, shape.h, &shape_reduce, NULL, NULL); 
    okk_parallel_end();


    //okk_bdc_exp_tunable(output_addr2, input_addr2, work_addr2, &shape, 3);
    okk_bdc_taylor_exp(output_addr2, input_addr2, &shape, 9);
    okk_bdc_avg_pool2d(work_addr2, output_addr2, &shape, shape.h, 1, NULL, NULL);

    okk_bdc_add(work_addr, work_addr, work_addr2, &shape_reduce, NULL, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr2, output_addr2, work_addr, &shape, &stride, &stride, &stride_reduce);

    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(param->output_addr + Sc * param->H * param->W * sizeof(float), output_addr2, &shape, &input_stride, NULL);
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();

    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, &input_stride, NULL);

    okk_poll();
}
#endif

void softmax_0(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = 1, .c = 13, .h = 185, .w = 13 };
    dim4 shape_reduce = { .n = 1, .c = 13, .h = 1, .w = 13 };
    dim4 stride = { .n = 2432, .c = 2432, .h = 13, .w = 1 };
    dim4 input_stride = { .n = 62530, .c = 13, .h = 169, .w = 1 };
    dim4 stride_reduce = { .n = 2432, .c = 2432, .h = 0, .w = 1 };
    local_addr_t input_addr = 0;
    local_addr_t output_addr = 9728;
    local_addr_t work_addr = 19456;
    local_addr_t input_addr2 = 29184;
    local_addr_t output_addr2 = 38912;
    local_addr_t work_addr2 = 48640;

    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, &input_stride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr2, param->input_addr + 125060, &shape, NULL, &input_stride);
    //okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape, 3);
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, 9);
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, 1, NULL, NULL);
    //okk_bdc_mul_C(work_addr, work_addr, shape.h, &shape_reduce, NULL, NULL); 
    okk_parallel_end();

    //okk_bdc_exp_tunable(output_addr2, input_addr2, work_addr2, &shape, 3);
    okk_bdc_taylor_exp(output_addr2, input_addr2, &shape, 9);
    okk_bdc_avg_pool2d(work_addr2, output_addr2, &shape, shape.h, 1, NULL, NULL);

    okk_bdc_add(work_addr, work_addr, work_addr2, &shape_reduce, NULL, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr2, output_addr2, work_addr, &shape, &stride, &stride, &stride_reduce);

    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(param->output_addr + 125060, output_addr2, &shape, &input_stride, NULL);
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();

    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, &input_stride, NULL);

    okk_poll();
}
void softmax_1(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = 1, .c = 1, .h = 1, .w = 1000 };
    dim4 shape_reduce = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 stride = { .n = 1024, .c = 1024, .h = 1000, .w = 1 };
    dim4 stride_reduce = { .n = 0, .0, .h = 0, .w = 0 };
    local_addr_t input_addr = 0;
    local_addr_t output_addr = 4096;
    local_addr_t work_addr = 8192;

    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
#if 1
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, 7);
#else
    okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape, 2);
#endif
#if 0
    SumPoolParam sumpool;
    sumpool.input_addr = output_addr;
    sumpool.output_addr = work_addr;
    sumpool.input_shape[0] = shape.n;
    sumpool.input_shape[1] = shape.c;
    sumpool.input_shape[2] = shape.h;
    sumpool.input_shape[3] = shape.w;
    sumpool.kernel_h = shape.h;
    sumpool.kernel_w = shape.w;
    sumpool.stride_h = 1;
    sumpool.stride_w = 1;
    sumpool.ins_h = 0;
    sumpool.ins_w = 0;
    sumpool.coeff = 1.;
    sumpool.pad[0] = 0;
    sumpool.pad[1] = 0;
    sumpool.pad[2] = 0;
    sumpool.pad[3] = 0;
    bm_atomic_sum_pool(&sumpool);
#else
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
#endif
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    okk_poll();
}
void softmax_2(const void* args) {
    param_t *param = (param_t *)args;
    int area = 355448;
    dim4 shape = { .n = 2, .c = 157, .h = 283, .w = 1 };
    dim4 stride_reduce = { .n = 0, .c = 288, .h = 1, .w = 1 };
    local_addr_t input_addrs[2] = {0, 6912};
    local_addr_t output_addrs[2] = {13824, 20736};
    local_addr_t work_addr = 27648;

    okk_gdma_32bit_cpy_S2L(input_addrs[0], param->input_addr, &shape, NULL, NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addrs[1], param->input_addr + area, &shape, NULL, NULL);
    okk_bdc_exp_tunable(output_addrs[0], input_addrs[0], work_addr, &shape, 5);
    okk_bdc_add(work_addr, output_addrs[0], output_addrs[0] + 3456, &shape, NULL, NULL, NULL);
    okk_bdc_div(output_addrs[0], output_addrs[0], work_addr, &shape, NULL, NULL, &stride_reduce);
    okk_parallel_end();
    int i = 1;
    for (; i < param->N-1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(input_addrs[(i+1)%2], param->input_addr + (i+1) * area, &shape, NULL, NULL);
        okk_bdc_exp_tunable(output_addrs[i%2], input_addrs[i%2], work_addr, &shape, 5);
        okk_bdc_add(work_addr, output_addrs[i%2], output_addrs[i%2] + 3456, &shape, NULL, NULL, NULL);
        okk_bdc_div(output_addrs[i%2], output_addrs[i%2], work_addr, &shape, NULL, NULL, &stride_reduce);
        okk_gdma_32bit_cpy_L2S(param->output_addr + (i-1) * area, output_addrs[(i-1)%2], &shape, NULL, NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_exp_tunable(output_addrs[i%2], input_addrs[i%2], work_addr, &shape, 5);
    okk_bdc_add(work_addr, output_addrs[i%2], output_addrs[i%2] + 3456, &shape, NULL, NULL, NULL);
    okk_bdc_div(output_addrs[i%2], output_addrs[i%2], work_addr, &shape, NULL, NULL, &stride_reduce);
    okk_gdma_32bit_cpy_L2S(param->output_addr + (i-1) * area, output_addrs[(i-1)%2], &shape, NULL, NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(param->output_addr + i * area, output_addrs[i%2], &shape, NULL, NULL);
    okk_poll();
}

void softmax_3(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = 1, .c = 40, .h = 10, .w = 409 };
    dim4 shape_reduce = { .n = 1, .c = 40, .h = 1, .w = 1 };
    dim4 stride = { .n = 4096, .c = 4096, .h = 409, .w = 1 };
    dim4 stride_reduce = { .n = 4096, .c = 4096, .h = 0, .w = 0 };
    local_addr_t input_addr = 0;
    local_addr_t output_addr = 32768;
    local_addr_t work_addr = 65536;
    local_addr_t input_addr2 = 16384;
    local_addr_t output_addr2 = 49152;


    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr2, param->input_addr + 654400, &shape, NULL, NULL);
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, 3);
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    shape.c = 39;
    shape_reduce.c = 39;
    okk_bdc_taylor_exp(output_addr2, input_addr2, &shape, 3);
    okk_bdc_avg_pool2d(work_addr, output_addr2, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr2, output_addr2, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(param->output_addr + 654400, output_addr2, &shape, NULL, NULL);
    okk_poll();
}

void softmax_4(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = { .n = 1, .c = 3066, .h = 1, .w = 21 };
    dim4 shape_reduce = { .n = 1, .c = 3066, .h = 1, .w = 1 };
    dim4 stride = { .n = 1536, .c = 32, .h = 21, .w = 1 };
    dim4 stride_reduce = { .n = 1536, .c = 32, .h = 0, .w = 0 };
    local_addr_t input_addr = 0;
    local_addr_t output_addr = 12288;
    local_addr_t work_addr = 24576;
    local_addr_t input_addr2 = 6144;
    local_addr_t output_addr2 = 18432;

    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(input_addr2, param->input_addr + 257544, &shape, NULL, NULL);
    okk_bdc_taylor_exp(output_addr, input_addr, &shape, 15);
    //okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape, 4);
    okk_bdc_avg_pool2d(work_addr, output_addr, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr, output_addr, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();


    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    shape.c = 3066;
    shape_reduce.c = 3066;
    okk_bdc_taylor_exp(output_addr2, input_addr2, &shape, 15);
    // okk_bdc_exp_tunable(output_addr2, input_addr2, work_addr, &shape, 4);
    okk_bdc_avg_pool2d(work_addr, output_addr2, &shape, shape.h, shape.w, NULL, NULL);
    okk_bdc_mul_C(work_addr, work_addr, shape.h * shape.w, &shape_reduce, NULL, NULL); 
    okk_bdc_div(output_addr2, output_addr2, work_addr, &shape, &stride, &stride, &stride_reduce);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(param->output_addr + 257544, output_addr2, &shape, NULL, NULL);
    okk_poll();
}
void softmax_contest(const void *args) {
    param_t *param = (param_t *)args;
    switch (param->C) {
	case 370:
            softmax_0(args);
	    return;
	case 1000:
            softmax_1(args);
	    return;
	case 2:
            softmax_2(args);
	    return;
	case 4090:
	    softmax_3(args);
	    return;
	case 21:
	    softmax_4(args);
	    return;
    }
#if 0
    if (param->C == 2) {
    	softmax_doublec(args);
    } else if (param->H == 1 && param->W == 1) {
        if (param->N == 1) {
            softmax_c(args);
        } else {
            softmax_nc(args);
        }
    } else {
        softmax_universal(args);
    }
#endif
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
