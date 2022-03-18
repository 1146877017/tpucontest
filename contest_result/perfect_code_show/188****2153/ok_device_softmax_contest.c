//
// wp 100
//
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

/*
 * param->h == 1 && param->w == 1
 */
void use_pooling_v1(param_t *param){

    const int max_n = (param->N > NPU_NUM) ? NPU_NUM : param->N;

    const int n_num = DIV_UP(param->N, max_n);
    const int n_overage = param->N - (n_num-1)*max_n;

    const dim4 input_shape = {.n = 1, .c = max_n, .h=1, .w = param->C};
    const dim4 input_overage_shape = {.n = 1, .c = n_overage, .h=1,.w = param->C};


    const dim4 pooling_shape = {.n = 1, .c = max_n, .h = 1, .w = 1};
    const dim4 pooling_overage_shape = {.n = 1, .c = n_overage, .h = 1, .w = 1};

    dim4 input_stride, pooling_stride;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&pooling_stride, 0, &pooling_shape);

    const int input_bytes = input_shape.n*input_stride.n* sizeof(float);
    const int pooling_bytes = pooling_shape.n*pooling_stride.n* sizeof(float);

    const int input_addr = 0;
    const int exp_res_addr = input_addr + input_bytes;
    const int exp_work_addr = exp_res_addr + input_bytes;
    const int pooling_addr = exp_work_addr;
    const int sum_addr = pooling_addr + pooling_bytes;
//    const int kernel_addr = sum_addr + pooling_bytes;
    const int kernel_addr = pooling_addr;

    OKKERNEL_ASSERT(sum_addr + pooling_bytes <= LOCAL_MEM_SIZE);

    for (int i = 0; i < n_num; ++i) {
        okk_gdma_32bit_cpy_S2L(
            input_addr,
            param->input_addr + i*max_n*param->C* sizeof(float),
            (i == n_num - 1) ? &input_overage_shape : &input_shape,
            NULL,
            NULL);

        okk_bdc_exp(exp_res_addr, input_addr, exp_work_addr,
                (i == n_num - 1) ? &input_overage_shape : &input_shape
                );

        okk_bdc_avg_pool2d(
                pooling_addr,
                exp_res_addr,
                (i == n_num - 1) ? &input_overage_shape : &input_shape,
                1,
                param->C,
                NULL,
                NULL);

        okk_bdc_mul_C(sum_addr, pooling_addr, param->C,
                (i == n_num - 1) ? &pooling_overage_shape : &pooling_shape,
                NULL, NULL);


        okk_bdc_reciprocal(kernel_addr, sum_addr,
                (i == n_num - 1) ? &pooling_overage_shape : &pooling_shape,
                NULL, NULL);

        okk_bdc_depthwise2d(
                input_addr,
                exp_res_addr,
                kernel_addr,
                NO_USE,
                (i == n_num - 1) ? &input_overage_shape : &input_shape,
                1,
                1,
                false,
                NULL,
                NULL,
                NULL);

        okk_gdma_32bit_cpy_L2S(
                param->output_addr + i*max_n*param->C* sizeof(float),
                input_addr,
                (i == n_num - 1) ? &input_overage_shape : &input_shape,
                NULL,
                NULL);
    }

}



void soso(param_t *param){
    const int hw = param->H * param->W;
    const dim4 input_shape = {.n = param->N, .c = param->C, .h = 1, .w = hw};
    const int input_addr = 0;
    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &input_shape, NULL, NULL);

    dim4 input_stride;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);

    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int exp_res_addr = input_addr + input_bytes;
    const int exp_work_addr = exp_res_addr + input_bytes;

    okk_bdc_exp(exp_res_addr, input_addr, exp_work_addr, &input_shape);

    const int IC_new = (param->C + 1) / 2;
    const dim4 conv_output_shape = {.n = param->N, .c = 1, .h = 1, .w = hw};
    const dim4 kernel_shape = {.n = IC_new, .c = 1, .h = 1, .w = 1 * 2};

    dim4 conv_output_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&conv_output_stride, 0, &conv_output_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    const int conv_output_bytes = conv_output_stride.n * conv_output_shape.n * sizeof(float);
    const int kernel_bytes = kernel_stride.n * kernel_shape.n * sizeof(float);

    const int conv_output_addr = exp_work_addr;
    const int kernel_addr = conv_output_addr + conv_output_bytes;

    if(kernel_addr + kernel_bytes > LOCAL_MEM_SIZE){
        return;
    }

    x32 one = {.fp32 = 1};
    okk_bdc_32bit_set_C(kernel_addr, one, &kernel_shape, &kernel_stride);

    dim4 kernel_shape_2IC = {.n = IC_new, .c =1, .h = 1, .w = 1};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

    okk_bdc_conv2d(
            conv_output_addr,
            exp_res_addr,
            kernel_addr,
            NO_USE,
            &input_shape,
            1,
            1,
            1,
            &input_stride,
            &kernel_stride_2IC,
            false,
            false,
            NULL,
            NULL,
            NULL);

    const int res_addr = input_addr;
    const int sub_addr = kernel_addr;

    const dim4 shape_1n = {.n = 1, .c = 1, .h =  1, .w = conv_output_shape.w};
    dim4 stride_1n;
    okk_128_byte_aligned_stride_for_32bit(&stride_1n, 0, &shape_1n);
    for (int i = 0; i < param->N; ++i) {
        unsigned int row = 0;
        unsigned int col = 0;
        for (int c = 0; c < input_shape.c; ++c) {
            okk_gdma_32bit_cpy_L2L(
                    sub_addr + i*input_stride.n* sizeof(float) + col*LOCAL_MEM_SIZE + row * stride_1n.n * sizeof(float),
                    conv_output_addr + i*conv_output_stride.n* sizeof(float),
                    &shape_1n,
                    NULL,
                    NULL);

            col++;
            if(col >= NPU_NUM){
                col = 0;
                row++;
            }
        }
    }

    okk_bdc_div(
            res_addr,
            exp_res_addr,
            sub_addr,
            &input_shape,
            NULL,
            NULL,
            NULL
            );

    okk_gdma_32bit_cpy_L2S(
            param->output_addr,
            res_addr,
            &input_shape,
            NULL,
            NULL);

}


void fig(param_t *param){

    const int hw = param->H * param->W;
    const int hw_max = 1024;
    const int hw_num = DIV_UP(hw, hw_max);
    const int hw_overage = hw - (hw_num-1) * hw_max;

    const dim4 input_shape = {.n = 1, .c = param->N, .h=param->C, .w = hw_max};
    const dim4 input_overage_shape = {.n = 1, .c = param->N, .h=param->C,.w = hw_overage};

    const dim4 pooling_shape = {.n = 1, .c = param->N, .h = 1, .w = hw_max};
    const dim4 pooling_overage_shape = {.n = 1, .c = param->N, .h = 1, .w = hw_overage};

    dim4 input_stride, pooling_stride;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&pooling_stride, 0, &pooling_shape);

    const int input_bytes = input_shape.n*input_stride.n* sizeof(float);
    const int pooling_bytes = pooling_shape.n*pooling_stride.n* sizeof(float);

    const int input_addr = 0;
    const int exp_res_addr = input_addr + input_bytes;
    const int exp_work_addr = exp_res_addr + input_bytes;
    const int pooling_addr = exp_work_addr;
    const int sum_addr = pooling_addr + pooling_bytes;

    if(sum_addr + pooling_bytes > LOCAL_MEM_SIZE){
        return;
    }

    const dim4 ddr_stride= {.n=param->N*param->C*hw, .c=param->C*hw, .h = hw, .w = 1};

    for (int i = 0; i < hw_num; ++i) {
        okk_gdma_32bit_cpy_S2L(
                input_addr,
                param->input_addr + i*hw_max* sizeof(float),
                (i == hw_num - 1) ? &input_overage_shape : &input_shape,
                NULL,
                &ddr_stride);

        okk_bdc_exp(exp_res_addr, input_addr, exp_work_addr,
                    (i == hw_num - 1) ? &input_overage_shape : &input_shape
        );

        okk_bdc_avg_pool2d(
                pooling_addr,
                exp_res_addr,
                (i == hw_num - 1) ? &input_overage_shape : &input_shape,
                param->C,
                1,
                NULL,
                NULL);

        okk_bdc_mul_C(sum_addr, pooling_addr, param->C,
                      (i == hw_num - 1) ? &pooling_overage_shape : &pooling_shape,
                      NULL, NULL);

        for (int j = 0; j < param->C; ++j) {
            okk_bdc_div(
                    input_addr + j*( i == hw_num-1 ? hw_overage : hw_max) * sizeof(float),
                    exp_res_addr + j*( i == hw_num-1 ? hw_overage : hw_max) * sizeof(float),
                    sum_addr,
                    &pooling_shape,
                    NULL,
                    NULL,
                    NULL
                    );
        }

        okk_gdma_32bit_cpy_L2S(
                param->output_addr + i*hw_max* sizeof(float),
                input_addr,
                (i == hw_num - 1) ? &input_overage_shape : &input_shape,
                &ddr_stride,
                NULL);
    }
}

void softmax_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    if(param->C == 2){
        fig(param);
    }
    if(param->H == 1 && param->C <2048){
        use_pooling_v1(param);
    } else if(param->H <= 13){
        soso(param);
    } else{
        //do nothing
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
