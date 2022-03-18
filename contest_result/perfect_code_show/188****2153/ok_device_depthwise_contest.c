//
// wp2
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
    int kernel_h, kernel_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    unsigned long long output_addr;
    unsigned long long input_addr;
    unsigned long long kernel_addr;
} __attribute__((packed)) param_t;

/*
 * param->H <= 256 && param->W <=256 &&
 * param->C > 3 //ggg
 * n = 1, split channel
 */
void depthwise_channel_v1(param_t *param){
    const int channel_kernel_size_max = 1024; //1024

    const int channel_kernel_num = DIV_UP(param->C, channel_kernel_size_max);
    const int channel_kernel_size_overage = param->C - (channel_kernel_num - 1)*channel_kernel_size_max;

    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    dim4 output_shape = {.n = 1, .c = channel_kernel_size_max, .h = output_h, .w = output_w};
    dim4 output_shape_overage = {.n = 1, .c = channel_kernel_size_overage, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = 1, .c = channel_kernel_size_max, .h = param->H, .w = param->W};
    dim4 input_shape_overage = {.n = 1, .c = channel_kernel_size_overage, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = channel_kernel_size_max, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_shape_overage = {.n = 1, .c = channel_kernel_size_overage, .h = param->kernel_h, .w = param->kernel_w};
    dim4 output_stride, input_stride, kernel_stride, kernel_overage_stride;

    local_addr_t output_addr[2];
    local_addr_t input_addr[2];
    local_addr_t kernel_addr;

    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    okk_compact_stride(&kernel_overage_stride, 0, &kernel_shape_overage);

    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);
    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);

    output_addr[0] = 0;
    output_addr[1] = output_bytes;

    input_addr[0] = output_bytes + output_bytes;
    input_addr[1] = input_addr[0] + input_bytes;

    kernel_addr = input_addr[1] + input_bytes;

    OKKERNEL_ASSERT(kernel_addr + kernel_bytes*channel_kernel_num <= LOCAL_MEM_SIZE);

    int n = 0;
    int n_1 = 0;
    int n_2 = 0;
    int c = 0;
    int c_1 = 0;
    int c_2 = 0;

    unsigned int flag = 0; // 0, 1
    const int S = param->N * channel_kernel_num;

    const int bytes_ddr_bank = param->C * param->H * param->W * sizeof(float);
    const int bytes_ddr_page = channel_kernel_size_max * param->H * param->W * sizeof(float);

    const int bytes_ddr_bank_output = param->C * output_h * output_w * sizeof(float);
    const int bytes_ddr_page_output = channel_kernel_size_max * output_h * output_w * sizeof(float);

    const int bytes_ddr_kernel = channel_kernel_size_max * param->kernel_h * param->kernel_w * sizeof(float);

    Padding padding = {.top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right};

    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    for (int i = 0; i < S+2; ++i) {
        okk_parallel_start();
        if(i < S) {
            okk_gdma_32bit_cpy_S2L(
                    input_addr[flag],
                    param->input_addr + n * bytes_ddr_bank + c * bytes_ddr_page,
                    c == (channel_kernel_num-1) ? (&input_shape_overage) : (&input_shape),
                    NULL,
                    NULL);

            if(n == 0) {
                okk_gdma_32bit_cpy_S2L(
                        kernel_addr + c * kernel_bytes,
                        param->kernel_addr + c*bytes_ddr_kernel,
                        c == (channel_kernel_num-1) ? (&kernel_shape_overage) : (&kernel_shape),
                        c == (channel_kernel_num-1) ? (&kernel_overage_stride) : (&kernel_stride),
                        NULL);
            }
        }

        if(i > 0 && i < S+1){
            okk_bdc_depthwise2d(
                    output_addr[!flag],
                    input_addr[!flag],
                    kernel_addr + c_1*kernel_bytes,
                    NO_USE,
                    c_1 == (channel_kernel_num-1) ? (&input_shape_overage): (&input_shape),
                    param->kernel_h,
                    param->kernel_w,
                    false,
                    &padding,
                    &stride,
                    &dilation);
        }

        if(i > 1){
            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + n_2 * bytes_ddr_bank_output + c_2 * bytes_ddr_page_output,
                    output_addr[flag],
                    c_2 == (channel_kernel_num-1) ? (&output_shape_overage) : (&output_shape),
                    NULL,
                    NULL);
        }

        flag = !flag;
        c_2 = c_1;
        n_2 = n_1;

        c_1 = c;
        n_1 = n;

        c++;
        if(c == channel_kernel_num){
            c = 0;
            n++;
        }
        okk_parallel_end();
    }


}



/*
 * param->N * param->C <= 64 &&
 * param->dilation_h == 1 && param->dilation_w == 1  // ggg
 */
void depthwise_h_v1(param_t *param){

    const int kernel_output_h_max = 16; //32
    const int nc = 12; // param->N * param->C;

#if 0
    const int nc = param->N * param->C;
    OKKERNEL_ASSERT(nc <= NPU_NUM);
    OKKERNEL_ASSERT(param->dilation_h == 1 && param->dilation_w == 1);
#endif

    dim4 output_stride, input_stride, kernel_stride;
    const local_addr_t kernel_addr = 0;
    const dim4 kernel_shape = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            NULL);

    for (int i = 1; i < param->N; ++i) {
        okk_gdma_32bit_cpy_L2L(
                kernel_addr + i*kernel_shape.c*LOCAL_MEM_SIZE,
                kernel_addr,
                &kernel_shape,
                &kernel_stride,
                &kernel_stride);
    }


    const int output_h = (param->H + param->pad_top + param->pad_bottom - param->kernel_h) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - param->kernel_w) / param->stride_w + 1;

    const int S = DIV_UP(output_h, kernel_output_h_max);
    const int kernel_output_h_overage = output_h - (S-1)*kernel_output_h_max;

    dim4 output_shape = {.n = 1, .c = nc, .h = kernel_output_h_max, .w = output_w};
    dim4 output_shape_overage = {.n = 1, .c = nc, .h = kernel_output_h_overage, .w = output_w};

    const int kernel_input_h = (kernel_output_h_max - 1) * param->stride_h + param->kernel_h;
    dim4 input_shape = {.n = 1, .c = nc, .h = kernel_input_h, .w = param->W};

    local_addr_t output_addr[2];
    local_addr_t input_addr[2];

    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);

    const int output_bytes = output_stride.n * sizeof(float); // n = 1
    const int input_bytes = input_stride.n * sizeof(float); // n = 1
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float); // n = 1

    output_addr[0] = kernel_addr +  DIV_UP(kernel_bytes, 128) * 128;;
    output_addr[1] = output_addr[0] + output_bytes;

    input_addr[0] = output_addr[1] + output_bytes;
    input_addr[1] = input_addr[0] + input_bytes;

    if(input_addr[1] + input_bytes > LOCAL_MEM_SIZE){
        return;
    }

    unsigned int flag = 0; // 0, 1

    const int bytes_ddr_kernel_output = kernel_output_h_max * output_w * sizeof(float);

    Padding padding = {.top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right};

    Padding padding_1 = padding;

    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    dim4 input_shape_1 = {.n = 1, .c = nc, .h = 0, .w = param->W};

    const dim4 ddr_input_stride = {.n = param->C * param->H * param->W, .c = param->H * param->W, .h = param->W, .w = 1};
    const dim4 ddr_output_stride = {.n = param->C * output_h * output_w, .c = output_h * output_w, .h = output_w, .w = 1};

    for (int i = 0; i < S+2; ++i) {
        okk_parallel_start();
        if(i < S) {
            int h_in_0_begin = i * kernel_output_h_max *param->stride_h - param->pad_top;

            int h_rod = kernel_input_h;

            padding.top = 0;
            padding.bottom = 0;
            if(h_in_0_begin < 0){
                padding.top = 0-h_in_0_begin;
                if(padding.top > param->pad_top) padding.top = param->pad_top;
                h_rod = h_rod + h_in_0_begin;
                h_in_0_begin = 0;
            }
            if(h_in_0_begin + h_rod > param->H){
                padding.bottom = h_in_0_begin + h_rod - param->H;
                if(padding.bottom > param->pad_bottom) padding.bottom = param->pad_bottom;
                h_rod = (param->H - h_in_0_begin);
            }

            input_shape.h = h_rod;
            okk_gdma_32bit_cpy_S2L(
                    input_addr[flag],
                    param->input_addr + h_in_0_begin * param->W * sizeof(float),
                    &input_shape,
                    NULL,
                    &ddr_input_stride);
        }

        if(i > 0 && i < S+1){
            okk_bdc_depthwise2d(
                    output_addr[!flag],
                    input_addr[!flag],
                    kernel_addr,
                    NO_USE,
                    &input_shape_1,
                    param->kernel_h,
                    param->kernel_w,
                    false,
                    &padding_1,
                    &stride,
                    &dilation);
        }

        if(i > 1){
            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + (i-2) * bytes_ddr_kernel_output,
                    output_addr[flag],
                    i == (S + 1) ? (&output_shape_overage) : (&output_shape),
                    &ddr_output_stride,
                    NULL);
        }

        flag = !flag;
        input_shape_1.h = input_shape.h;
        padding_1 = padding;

        okk_parallel_end();
    }

}



/*
 * param->H <= 256 && param->W <=256 &&
 * param->C > 3 //ggg
 * split n
 */
void depthwise_n_v1(param_t *param){
    const int kernel_size_max = param->C >= 192 ? 2 : 1;

    const int S = param->N/kernel_size_max;

    dim4 output_stride, input_stride, kernel_stride;

    const dim4 kernel_shape = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    const local_addr_t kernel_addr =0;

    okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            NULL);

    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    const dim4 output_shape = {.n = kernel_size_max, .c = param->C, .h = output_h, .w = output_w};
    const dim4 input_shape = {.n = kernel_size_max, .c = param->C, .h = param->H, .w = param->W};

    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);

    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);
    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);

    local_addr_t output_addr[2];
    local_addr_t input_addr[2];

    output_addr[0] = kernel_addr +  DIV_UP(kernel_bytes, 128) * 128;
    output_addr[1] = output_addr[0] + output_bytes;

    input_addr[0] = output_addr[1] + output_bytes;
    input_addr[1] = input_addr[0] + input_bytes;

    if(input_addr[1] + input_bytes > LOCAL_MEM_SIZE){
        return;
    }

    unsigned int flag = 0; // 0, 1

    const int bytes_ddr_input = kernel_size_max * param->C * param->H * param->W * sizeof(float);
    const int bytes_ddr_output = kernel_size_max * param->C * output_h * output_w * sizeof(float);

    const Padding padding = {.top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right};

    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    for (int i = 0; i < S+2; ++i) {
        okk_parallel_start();
        if(i < S) {
            okk_gdma_32bit_cpy_S2L(
                    input_addr[flag],
                    param->input_addr + i * bytes_ddr_input,
                    &input_shape,
                    NULL,
                    NULL);
        }

        if(i > 0 && i < S+1){
            okk_bdc_depthwise2d(
                    output_addr[!flag],
                    input_addr[!flag],
                    kernel_addr,
                    NO_USE,
                    &input_shape,
                    param->kernel_h,
                    param->kernel_w,
                    false,
                    &padding,
                    &stride,
                    &dilation);
        }

        if(i > 1){
            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + (i-2)*bytes_ddr_output,
                    output_addr[flag],
                    &output_shape,
                    NULL,
                    NULL);
        }

        flag = !flag;
        okk_parallel_end();
    }

}


void depthwise_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    if(param->C <= 3){
        depthwise_h_v1(param);
    } else if(param->C >= 1024){
        depthwise_channel_v1(param);
    } else{
        depthwise_n_v1(param);
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(depthwise_contest);
