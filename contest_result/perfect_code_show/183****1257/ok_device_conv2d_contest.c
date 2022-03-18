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
    int N, IC, OC, H, W;
    int kernel_h, kernel_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    unsigned long long output_addr;
    unsigned long long input_addr;
    unsigned long long kernel_addr;
} __attribute__((packed)) param_t;

// // split by N and no precision error
// void conduct_conv2d_1(param_t *param){

// }


void conv2d_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    const int IC_new = (param->IC + 1) / 2;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    dim4 output_stride, input_stride, kernel_stride, sys_input_stride, sys_output_stride, sys_kernel_stride;

    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    okk_continuous_stride(&sys_input_stride, &input_shape);
    okk_continuous_stride(&sys_output_stride, &output_shape);
    okk_continuous_stride(&sys_kernel_stride, &kernel_shape);

    // dim4 system_input_stride = {.n=input_shape.c*input_shape.h*input_shape.w, .c=input_shape.h*input_shape.w, .h=input_shape.w, .w=1};
    // dim4 system_output_stride = {.n=output_shape.c*output_shape.h*output_shape.w, .c=output_shape.h*output_shape.w, .h=output_shape.w, .w=1};

    long kernel_size = kernel_shape.n * kernel_stride.n * (int)sizeof(float);

    // cal max N
    int max_N = (LOCAL_MEM_SIZE - kernel_size) / ((input_stride.n + output_stride.n) * sizeof(float));

    // max_N = MIN(max_N, param->N);

    if(max_N > 1)
    {
        max_N /= 2;
        int channel_size = 512, channel_iter = 1;

        local_addr_t input_addr[2], output_addr[2], kernel_addr;
        output_addr[0] = 0;
        input_addr[0] = output_addr[0] + max_N * output_stride.n * sizeof(float);

        output_addr[1] = input_addr[0] + max_N * input_stride.n * sizeof(float);
        input_addr[1] = output_addr[1] + max_N * output_stride.n * sizeof(float);

        kernel_addr = input_addr[1] + max_N * input_stride.n * sizeof(float);
        OKKERNEL_ASSERT(output_addr[1] % 128 == 0);
        OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);

        if(param->OC > channel_size)
        {
            channel_iter = DIV_UP(param->OC, channel_size);
        }else
        {
            channel_size = MIN(param->OC, channel_size);
        }

        int channel_size_last = param->OC - (channel_iter - 1)*channel_size;

        dim4 new_input_shape = {.n=max_N, .c=input_shape.c, .h=input_shape.h, .w=input_shape.w};
        
        unsigned int input_tensor_size_global = max_N * input_shape.c * input_shape.h * input_shape.w * sizeof(float);
        unsigned int output_tensor_size_global = max_N * output_shape.c *output_shape.h *output_shape.w * sizeof(float);

        Padding padding = {
            .left=param->pad_left, .right=param->pad_right,
            .top=param->pad_top, .bottom=param->pad_bottom
        };
        dim2 stride = {.h = param->stride_h, .w = param->stride_w};
        dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

        int output_channel_skip_size = channel_size * output_h * output_w * sizeof(float);
        int kernel_skip_size = channel_size * kernel_shape.h * kernel_shape.w * sizeof(float);

        for(int c=0; c<channel_iter; c++)
        {
            dim4 new_output_shape = {.n=max_N, .c=(c==channel_iter-1?channel_size_last:channel_size), .h=output_shape.h, .w=output_shape.w};
            dim4 new_kernel_shape = {.n = IC_new, .c =(c==channel_iter-1?channel_size_last:channel_size), .h = kernel_shape.h, .w = kernel_shape.w};
            dim4 new_output_stride, new_kernel_stride;
            okk_128_byte_aligned_stride_for_32bit(&new_output_stride, 0, &new_output_shape);
            okk_compact_stride(&new_kernel_stride, 0, &new_kernel_shape);

            int iteration = DIV_UP(param->N, max_N);
            dim4 intput_shape_last = {.n=param->N - (iteration - 1)*max_N, .c=param->IC, .h=param->H, .w=param->W};
            dim4 output_shape_last = {.n=param->N - (iteration - 1)*max_N, .c=(c==channel_iter-1?channel_size_last:channel_size), .h=output_h, .w=output_w};

            dim4 last_output_stride;
            okk_128_byte_aligned_stride_for_32bit(&new_output_stride, 0, &new_output_shape);

            okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr + kernel_skip_size * c, &new_kernel_shape, &new_kernel_stride, &sys_kernel_stride);

            dim4 kernel_shape_2IC = {.n = IC_new, .c=(c==channel_iter-1?channel_size_last:channel_size), .h = param->kernel_h, .w = param->kernel_w};
            dim4 kernel_stride_2IC;
            okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

            for(int i=0;i<iteration+2;i++)
            {
                okk_parallel_start();

                if(i<iteration)
                    okk_gdma_32bit_cpy_S2L(input_addr[i%2], param->input_addr + i * input_tensor_size_global, i==iteration-1?&intput_shape_last:&new_input_shape, NO_USE, &sys_input_stride);

                if(i>0 && i<iteration+1)
                    okk_bdc_conv2d(output_addr[(i-1)%2], input_addr[(i-1)%2], kernel_addr, NO_USE, i == iteration?&intput_shape_last:&new_input_shape, kernel_shape_2IC.c, param->kernel_h, param->kernel_w, &input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);
                
                if(i>1)
                    okk_gdma_32bit_cpy_L2S(param->output_addr + (i-2)*output_tensor_size_global + c * output_channel_skip_size, output_addr[i%2], i == iteration+1? &output_shape_last:&new_output_shape, &sys_output_stride, NO_USE);

                okk_parallel_end();
            }
        }
    }else
    {
        // split by H

        int split_output_H = MIN(32, output_h);

        int iteration = DIV_UP(output_h, split_output_H);

        int split_input_H = (split_output_H - 1) * param->stride_h + kernel_h_ext;
        int last_output_H = output_h - (iteration-1) * split_output_H;
        int last_intput_H = (last_output_H - 1) * param->stride_h + kernel_h_ext;

        dim4 new_input_shape = {.n = param->N, .c=param->IC, .h=split_input_H, .w=param->W};
        dim4 new_output_shape = {.n = param->N, .c=param->OC, .h=split_output_H, .w=output_w};

        dim4 new_input_stride, new_output_stride;
        okk_128_byte_aligned_stride_for_32bit(&new_input_stride, 0, &new_input_shape);
        okk_128_byte_aligned_stride_for_32bit(&new_output_stride, 0, &new_output_shape);

        int max_N = MIN((LOCAL_MEM_SIZE - kernel_size) / (new_input_stride.n + new_output_stride.n) / sizeof(float), param->N);

        int N_iteration = param->N /max_N;

        int last_N = param->N - (N_iteration - 1)*max_N;

        dim2 stride = {.h = param->stride_h, .w = param->stride_w};
        dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
        dim4 kernel_shape_2IC = {.n = IC_new, .c=param->OC, .h = param->kernel_h, .w = param->kernel_w};
        dim4 kernel_stride_2IC;
        okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);

        int input_skip_N = max_N * param->IC * param->H * param->W * sizeof(float);
        int output_skip_N = max_N * param->OC * output_h * output_w * sizeof(float);

        local_addr_t input_addr, output_addr, kernel_addr;
        output_addr = 0;
        input_addr = output_addr + max_N * new_output_stride.n * sizeof(float);
        kernel_addr = input_addr + max_N * new_input_stride.n * sizeof(float);

        OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);

        // local_addr_t input_addr[2], output_addr[2], kernel_addr;
        // output_addr[0] = 0;
        // input_addr[0] = output_addr[0] + max_N * new_output_stride.n * sizeof(float);

        // output_addr[1] = input_addr[0] + max_N * new_input_stride.n * sizeof(float);
        // input_addr[1] = output_addr[1] + max_N * new_output_stride.n * sizeof(float);

        // kernel_addr = input_addr[1] + max_N * new_input_stride.n * sizeof(float);

        for(int n=0; n<N_iteration; n++)
        {
            new_input_shape.n = (n==N_iteration - 1)?last_N:max_N;
            new_output_shape.n = (n==N_iteration - 1)?last_N:max_N;

            dim4 last_output_shape = {.n = new_input_shape.n, .c=param->OC, .h=last_output_H, .w=output_w};
            dim4 last_output_stride;
            okk_128_byte_aligned_stride_for_32bit(&last_output_stride, 0, &last_output_shape);

            okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);

            int input_skip_size_per_H = param->W * sizeof(float);
            int output_skip_size_per_iter = split_output_H * output_w * sizeof(float);

            dim4 temp_shape={.n=new_input_shape.n, .c=param->IC, .h=new_input_shape.h, .w=param->W};
            dim4 temp_stride;
            okk_128_byte_aligned_stride_for_32bit(&temp_stride, 0, &temp_shape);

            for(int i =0; i<iteration; i++)
            {
                Padding padding = {
                    .left=param->pad_left, .right=param->pad_right,
                    .top=0, .bottom=0
                };
                bool use_temp=false;

                int input_skip_H = i*split_output_H*param->stride_h - param->pad_top;
                if(input_skip_H < 0)
                {
                    // 需要padding
                    padding.top = 0-input_skip_H;
                    temp_shape.h = new_input_shape.h - padding.top;
                    okk_128_byte_aligned_stride_for_32bit(&temp_stride, 0, &temp_shape);
                    input_skip_H = 0;

                    use_temp = true;
                }

                // 可能出现不需要那么多输入的情况，例如最后一次卷积完还剩下两个元素，无法完成下一次卷积
                int end_H = MIN(i*split_output_H*param->stride_h + kernel_h_ext+((i==iteration-1?last_output_H:split_output_H)-1)*param->stride_h - param->pad_top, param->H + param->pad_bottom);
                if(end_H > param->H)
                {
                    padding.bottom = end_H - param->H;
                    if(use_temp){
                        temp_shape.h -= padding.bottom;
                        okk_128_byte_aligned_stride_for_32bit(&temp_stride, 0, &temp_shape);
                    }else{
                        temp_shape.h = ((i==iteration-1)?last_intput_H:new_input_shape.h) - padding.bottom;
                        okk_128_byte_aligned_stride_for_32bit(&temp_stride, 0, &temp_shape);

                        use_temp = true;
                    }
                }

                okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + n*input_skip_N + input_skip_H * input_skip_size_per_H, use_temp?&temp_shape:&new_input_shape, use_temp?&temp_stride:&new_input_stride, &sys_input_stride);

                okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, use_temp?&temp_shape:&new_input_shape, param->OC, param->kernel_h, param->kernel_w, use_temp?&temp_stride:&new_input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

                okk_gdma_32bit_cpy_L2S(param->output_addr + n*output_skip_N + i * output_skip_size_per_iter, output_addr, (i==iteration-1)?&last_output_shape:&new_output_shape, &sys_output_stride, (i==iteration-1)?&last_output_stride:&new_output_stride);
            }

        }
    }
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(conv2d_contest);
