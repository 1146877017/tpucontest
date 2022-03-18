// wp 100
#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
#define NO_USE 0
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

/*
 * max_n must be 1 or 2
 */
void stream_n(param_t *param, int max_n){

    const int S = param->N/max_n;

    dim4 output_stride, input_stride, kernel_stride;
    const int IC_new = (param->IC + 1) / 2;
    const dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const local_addr_t kernel_addr = 0;

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
    const dim4 output_shape = {.n = max_n, .c = param->OC, .h = output_h, .w = output_w};
    const dim4 input_shape = {.n = max_n, .c = param->IC, .h = param->H, .w = param->W};

    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t input_addr[2];
    local_addr_t output_addr[2];



    input_addr[0] = kernel_addr + DIV_UP(kernel_bytes, 128) * 128;

    input_addr[1] = input_addr[0] + input_bytes;

    output_addr[0] = input_addr[1] + input_bytes;
    output_addr[1] = output_addr[0] + output_bytes;

    if(output_addr[1] + output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    const Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right
    };
    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    const dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

    const int ddr_input_max_n_bytes = max_n * param->IC * param->H * param->W * sizeof(float);
    const int ddr_output_max_n_bytes = max_n * param->OC * output_h * output_w * sizeof(float);

    int flag = 0;
    for (int i = 0; i < S+2; ++i) {
        okk_parallel_start();
        if(i < S){
            okk_gdma_32bit_cpy_S2L(
                    input_addr[flag],
                    param->input_addr + i * ddr_input_max_n_bytes,
                    &input_shape,
                    NULL,
                    NULL);
        }

        if(i > 0 && i < S+1){
            okk_bdc_conv2d(
                    output_addr[!flag],
                    input_addr[!flag],
                    kernel_addr,
                    NO_USE,
                    &input_shape,
                    param->OC,
                    param->kernel_h,
                    param->kernel_w,
                    &input_stride,
                    &kernel_stride_2IC,
                    false,
                    false,
                    &padding,
                    &stride,
                    &dilation);
        }

        if(i > 1){
            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + (i-2) * ddr_output_max_n_bytes,
                    output_addr[flag],
                    &output_shape,
                    NULL,
                    NULL);
        }

        flag = !flag;
        okk_parallel_end();
    }

}

#if 0
void my_demo(param_t *param){

    const int max_n = 1;
    const int S = param->N/max_n;

    dim4 output_stride, input_stride, kernel_stride;
    const int IC_new = (param->IC + 1) / 2;
    const dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const local_addr_t kernel_addr = 0;

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
    const dim4 output_shape = {.n = max_n, .c = param->OC, .h = output_h, .w = output_w};
    const dim4 input_shape = {.n = max_n, .c = param->IC, .h = param->H, .w = param->W};

    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t input_addr;
    local_addr_t output_addr;

    input_addr = kernel_addr + DIV_UP(kernel_bytes, 128) * 128;

    output_addr = input_addr + input_bytes;

    if(output_addr + output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    const Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right
    };
    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    const dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

    const int ddr_input_max_n_bytes = max_n * param->IC * param->H * param->W * sizeof(float);
    const int ddr_output_max_n_bytes = max_n * param->OC * output_h * output_w * sizeof(float);

    for (int i = 0; i < S; ++i) {
        okk_gdma_32bit_cpy_S2L(
                input_addr,
                param->input_addr + i * ddr_input_max_n_bytes,
                &input_shape,
                NULL,
                NULL);

        okk_bdc_conv2d(
                output_addr,
                input_addr,
                kernel_addr,
                NO_USE,
                &input_shape,
                param->OC,
                param->kernel_h,
                param->kernel_w,
                &input_stride,
                &kernel_stride_2IC,
                false,
                false,
                &padding,
                &stride,
                &dilation);

        okk_gdma_32bit_cpy_L2S(
                param->output_addr + i * ddr_output_max_n_bytes,
                output_addr,
                &output_shape,
                NULL,
                NULL);

    }

}
#endif

#if 0
void split_oc(param_t *param, int oc_max){

    const int S = DIV_UP(param->OC, oc_max);
    const int n_max = 1;

    const int IC_new = (param->IC + 1) / 2;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    const dim4 input_shape = {.n = n_max, .c = param->IC, .h = param->H, .w = param->W};
    const dim4 output_shape = {.n = n_max, .c = oc_max, .h = output_h, .w = output_w};
    const dim4 kernel_shape = {.n = IC_new, .c = oc_max, .h = param->kernel_h, .w = param->kernel_w * 2};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);


    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);


    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_bytes;
    local_addr_t kernel_addr = output_addr + output_bytes;

    if(kernel_addr + kernel_bytes > LOCAL_MEM_SIZE){
        OKKERNEL_LOG("out mem\n");
        return;
    }

    const Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right
    };
    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    const dim4 kernel_shape_2IC = {.n = IC_new, .c = oc_max, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);


    const dim4 kernel_ddr_shape = {.n = param->OC * kernel_shape.h * kernel_shape.w, .c = kernel_shape.h * kernel_shape.w, .h = kernel_shape.w, .w = 1};

    const int ddr_tile_kernel_bytes = oc_max * kernel_ddr_shape.c*sizeof(float);
    const int ddr_n_input_bytes = param->IC * param->H * param->W * sizeof(float);
    const int ddr_n_output_bytes = param->OC * output_h * output_w * sizeof(float);
    const int ddr_s_output_bytes = oc_max * output_h * output_w * sizeof(float);

    const int n_num = param->N/n_max;

    for (int n = 0; n < n_num; ++n) {
        okk_gdma_32bit_cpy_S2L(
                input_addr,
                param->input_addr + n * ddr_n_input_bytes,
                &input_shape,
                NULL,
                NULL);

        for (int i = 0; i < S; ++i) {

            okk_gdma_32bit_cpy_S2L(
                    kernel_addr,
                    param->kernel_addr + i * ddr_tile_kernel_bytes,
                    &kernel_shape,
                    &kernel_stride,
                    &kernel_ddr_shape);

            okk_bdc_conv2d(
                    output_addr,
                    input_addr,
                    kernel_addr,
                    NO_USE,
                    &input_shape,
                    oc_max,
                    param->kernel_h,
                    param->kernel_w,
                    &input_stride,
                    &kernel_stride_2IC,
                    false,
                    false,
                    &padding,
                    &stride,
                    &dilation);

            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + n*ddr_n_output_bytes + i * ddr_s_output_bytes,
                    output_addr,
                    &output_shape,
                    NULL,
                    NULL);
        }

    }

}
#endif

#if 0
/*
 * without dilation
 */
void split_h(param_t *param, int output_h_max){
    const int max_n = 1;

    dim4 output_stride, input_stride, kernel_stride;
    const int IC_new = (param->IC + 1) / 2;
    const dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const local_addr_t kernel_addr = 0;

    okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            NULL);

    const int output_h = (param->H + param->pad_top + param->pad_bottom - param->kernel_h) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - param->kernel_w) / param->stride_w + 1;
    const int output_h_num = output_h/output_h_max;

    const int input_h_max = (output_h_max - 1) * param->stride_h + param->kernel_h; // withput dilation

    const dim4 output_shape = {.n = max_n, .c = param->OC, .h = output_h_max, .w = output_w};
    dim4 input_shape = {.n = max_n, .c = param->IC, .h = input_h_max, .w = param->W};

    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t input_addr;
    local_addr_t output_addr;

    input_addr = kernel_addr + DIV_UP(kernel_bytes, 128) * 128;

    output_addr = input_addr + input_bytes;

    if(output_addr + output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right
    };
    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

    const dim4 ddr_input_stride = {.n = param->IC * param->H * param->W, .c = param->H * param->W, .h = param->W, .w = 1};
    const dim4 ddr_output_stride = {.n = param->OC * output_h * output_w, .c = output_h * output_w, .h = output_w, .w = 1};

    const int ddr_input_1n_bytes =  ddr_input_stride.n * sizeof(float);
    const int ddr_output_1n_bytes =  ddr_output_stride.n * sizeof(float);

    for (int n = 0; n < param->N/max_n; ++n) {
        for (int i = 0; i < output_h_num; ++i) {

            int h_in_0_begin = i * output_h_max * param->stride_h - param->pad_top;
            int h_rod = input_h_max;

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
                    input_addr,
                    param->input_addr + n*ddr_input_1n_bytes + h_in_0_begin * param->W * sizeof(float),
                    &input_shape,
                    NULL,
                    &ddr_input_stride);


            okk_bdc_conv2d(
                    output_addr,
                    input_addr,
                    kernel_addr,
                    NO_USE,
                    &input_shape,
                    param->OC,
                    param->kernel_h,
                    param->kernel_w,
                    &input_stride,
                    &kernel_stride_2IC,
                    false,
                    false,
                    &padding,
                    &stride,
                    NULL);

            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + n * ddr_output_1n_bytes + i * output_h_max * output_w * sizeof(float),
                    output_addr,
                    &output_shape,
                    &ddr_output_stride,
                    NULL);
        }
    }

}
#endif

/*
 * without dilation
 */
void split_h_stream(param_t *param, int output_h_max){
    const int max_n = 1;

    dim4 output_stride, input_stride, kernel_stride;
    const int IC_new = (param->IC + 1) / 2;
    const dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const local_addr_t kernel_addr = 0;

    okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            NULL);

    const int output_h = (param->H + param->pad_top + param->pad_bottom - param->kernel_h) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - param->kernel_w) / param->stride_w + 1;
    const int output_h_num = output_h/output_h_max;

    const int input_h_max = (output_h_max - 1) * param->stride_h + param->kernel_h; // withput dilation

    const dim4 output_shape = {.n = max_n, .c = param->OC, .h = output_h_max, .w = output_w};
    dim4 input_shape = {.n = max_n, .c = param->IC, .h = input_h_max, .w = param->W};

    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t input_addr[2];
    local_addr_t output_addr[2];

    input_addr[0] = kernel_addr + DIV_UP(kernel_bytes, 128) * 128;
    input_addr[1] = input_addr[0] + input_bytes;
    output_addr[0] = input_addr[1] + input_bytes;
    output_addr[1] = output_addr[0] + output_bytes;

    if(output_addr[1] + output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right
    };
    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

    const dim4 ddr_input_stride = {.n = param->IC * param->H * param->W, .c = param->H * param->W, .h = param->W, .w = 1};
    const dim4 ddr_output_stride = {.n = param->OC * output_h * output_w, .c = output_h * output_w, .h = output_w, .w = 1};

    const int ddr_input_1n_bytes = ddr_input_stride.n * sizeof(float);
    const int ddr_output_1n_bytes = ddr_output_stride.n * sizeof(float);
    const int ddr_output_1s_bytes = output_h_max * output_w * sizeof(float);

    const int S = param->N * output_h_num;
    unsigned int flag = 0;

    Padding padding_1 = padding;
    dim4 input_shape_1 = {.n = 1, .c = input_shape.c, .h = 0, .w = param->W};

    int n_id = 0;
    int n_id_1 = 0;
    int n_id_2 = 0;
    int output_h_id = 0;
    int output_h_id_1 = 0;
    int output_h_id_2 = 0;

    for (int i = 0; i < S+2; ++i) {
        okk_parallel_start();
        if(i < S){
            int h_in_0_begin = output_h_id * output_h_max * param->stride_h - param->pad_top;
            int h_rod = input_h_max;

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
                    param->input_addr + n_id*ddr_input_1n_bytes + h_in_0_begin * param->W * sizeof(float),
                    &input_shape,
                    NULL,
                    &ddr_input_stride);
        }

        if(i > 0 && i < S+1){
            okk_bdc_conv2d(
                    output_addr[!flag],
                    input_addr[!flag],
                    kernel_addr,
                    NO_USE,
                    &input_shape_1,
                    param->OC,
                    param->kernel_h,
                    param->kernel_w,
                    &input_stride,
                    &kernel_stride_2IC,
                    false,
                    false,
                    &padding_1,
                    &stride,
                    NULL);
        }

        if(i > 1){
            okk_gdma_32bit_cpy_L2S(
                    param->output_addr + n_id_2 * ddr_output_1n_bytes + output_h_id_2 * ddr_output_1s_bytes,
                    output_addr[flag],
                    &output_shape,
                    &ddr_output_stride,
                    NULL);
        }


        n_id_2 = n_id_1;
        n_id_1 = n_id;

        output_h_id_2 = output_h_id_1;
        output_h_id_1 = output_h_id;

        output_h_id++;

        if(output_h_id == output_h_num){
            output_h_id = 0;
            n_id++;
        }

        flag = !flag;
        input_shape_1.h = input_shape.h;
        padding_1 = padding;
        okk_parallel_end();
    }

}

#if 1
void split_ic(param_t *param, int ic_max){

    const int max_n = 1;

    dim4 output_stride, input_stride, kernel_stride;
    const int IC_new = (param->IC + 1) / 2;
    const dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const local_addr_t kernel_addr = 0;

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
    const dim4 output_shape = {.n = max_n, .c = param->OC, .h = output_h, .w = output_w};
    const dim4 input_shape = {.n = max_n, .c = ic_max, .h = param->H, .w = param->W};

    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t input_addr;
    local_addr_t output_addr;

    input_addr = kernel_addr + DIV_UP(kernel_bytes, 128) * 128 + 128;

    output_addr = input_addr + input_bytes;

    if(output_addr + output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    const Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom,
            .left = param->pad_left, .right = param->pad_right
    };
    const dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    const dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    const dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

    const int ddr_input_max_n_bytes = max_n * param->IC * param->H * param->W * sizeof(float);
    const int ddr_output_max_n_bytes = max_n * param->OC * output_h * output_w * sizeof(float);

    for (int n = 0; n < param->N/max_n; ++n) {
        for (int i = 0; i < param->IC/ic_max; ++i) {
            okk_gdma_32bit_cpy_S2L(
                    input_addr,
                    param->input_addr + n * ddr_input_max_n_bytes + i*ic_max*param->H * param->W * sizeof(float),
                    &input_shape,
                    NULL,
                    NULL);

            okk_bdc_conv2d(
                    output_addr,
                    input_addr,
                    kernel_addr + (i*ic_max/2)* kernel_stride.n * sizeof(float) + ((i*ic_max%2 == 0)? 0 : sizeof(float)),
                    NO_USE,
                    &input_shape,
                    param->OC,
                    param->kernel_h,
                    param->kernel_w,
                    &input_stride,
                    &kernel_stride_2IC,
                    false,
                    i == 0 ? false : true,
                    &padding,
                    &stride,
                    &dilation);
        }

        okk_gdma_32bit_cpy_L2S(
                param->output_addr + n * ddr_output_max_n_bytes,
                output_addr,
                &output_shape,
                NULL,
                NULL);
    }
}
#endif

#if 0
void wc(param_t *param){
    const int max_n = 1;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    dim4 output_stride, input_stride, kernel_stride;
    const int IC_new = (param->IC + 1) / 2;
    const dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    const int kernel_bytes = kernel_shape.n * kernel_stride.n * sizeof(float);
    const local_addr_t kernel_addr = 0;

    okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            NULL);

    const dim4 input_shape = {.n = max_n, .c = param->IC, .h = param->H, .w = param->W};
    const dim4 output_shape = {.n = 1, .c = 1,.h = 1, .w = 1};

    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);


    const int input_bytes = input_shape.n * input_stride.n * sizeof(float);
    const int output_bytes = output_shape.n * output_stride.n * sizeof(float);

    local_addr_t input_addr;
    local_addr_t output_addr;

    input_addr = kernel_addr + DIV_UP(kernel_bytes, 128) * 128;
    output_addr = input_addr + input_bytes;

    if(output_addr + output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    const int ddr_input_max_n_bytes = max_n * param->IC * param->H * param->W * sizeof(float);

    for (int n = 0; n < param->N; ++n) {

        okk_gdma_32bit_cpy_S2L(
                input_addr,
                param->input_addr + n * ddr_input_max_n_bytes,
                &input_shape,
                NULL,
                NULL);
        for (int oc = 0; oc < param->OC; ++oc) {

            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    x32 zero;
                    zero.fp32 = 0.0;
                    okk_bdc_32bit_set_C(output_addr, zero, &output_shape, NULL);

                    for (int kh = 0; kh < param->kernel_h; ++kh) {
                        for (int kw = 0; kw < param->kernel_w; ++kw) {
                            int ih = oh * param->stride_h + kh * param->dilation_h - param->pad_top;
                            int iw = ow * param->stride_w + kw * param->dilation_w - param->pad_left;

                            if (ih >= 0 && ih < param->H && iw >= 0 && iw < param->W) {

                                for (int ic = 0; ic < param->IC; ++ic) {

                                    okk_bdc_mac(
                                                output_addr,
                                                kernel_addr + ((ic/2)*kernel_stride.n + oc*kernel_stride.c
                                                    + kh*kernel_stride.h + kw*2 + (ic%2))*sizeof(float),
                                                input_addr + (n*input_stride.n + ic*input_stride.c
                                                    + ih*input_stride.h + iw)*sizeof(float),
                                                &output_shape, NULL, NULL, NULL);

                                }
                            }
                        }
                    }
                    okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (n * param->OC * output_h * output_w + oc * output_h * output_w + oh * output_w + ow)*
                                                 sizeof(float),
                            output_addr,
                            &output_shape,
                            NULL,
                            NULL);
                }
            }
        }
    }

}

#endif

void conv2d_contest(const void *args) {
    okk_initialize();
    (void) (args);
    param_t *param = (param_t *) args;

    if(param->IC == 4032){
        //case 14
        split_ic(param, 1);
    }else if(param->IC == 512 || (param->IC == 1024) || (param->IC == 2048)){
        //case 11 ,12 ,13
       //do nothing
    } else if(param->IC == 128){
        //case 8
        stream_n(param, 2);
    } else if((param->H == 640) || (param->H == 384) || (param->IC == 256)) {
        //case 0, 3, 10
        split_h_stream(param, 32);
    } else if(param->H == 512) {
        //case 1
        split_h_stream(param, 16);
    } else if(param->H == 1080){
        //case 2
        split_h_stream(param, 15);
    } else {
        stream_n(param, 1);
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(conv2d_contest);
