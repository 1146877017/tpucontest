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

//  // 整合到C中的n
//     int n = 1;
//     for(; n<param->N+1; n++)
//     {
//         input_shape.n = 1;
//         input_shape.c = param->C * n;
//         output_shape.n = 1;
//         output_shape.c = param->C * n;
//         kernel_shape.c = param->C * n;

//         okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
//         okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
//         okk_compact_stride(&kernel_stride, 0, &kernel_shape);

//         int input_size = input_shape.n * input_stride.n * sizeof(float);
//         int output_size = output_shape.n * output_stride.n * sizeof(float);
//         int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);

//         if(input_size + output_size + kernel_size > LOCAL_MEM_SIZE)
//             break;
//     }

//     // OKKERNEL_LOG("i:%d\n", i);

//     if(n>1)
//     {
//         input_shape.n = 1;
//         input_shape.c = param->C * (n-1);
//         output_shape.n = 1;
//         output_shape.c = param->C * (n-1);
//         kernel_shape.c = param->C * (n-1);

//         okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
//         okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
//         okk_compact_stride(&kernel_stride, 0, &kernel_shape);

//         local_addr_t input_addr, output_addr, kernel_addr;
//         output_addr = 0;
//         input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
//         kernel_addr = input_addr + input_shape.n * input_stride.n * sizeof(float);

//         kernel_shape.c = param->C;
//         for(int t=0;t<n-1;t++)
//         {
//             int idx = (t*param->C)%NPU_NUM;
//             okk_compact_stride(&kernel_stride, idx, &kernel_shape);
//             okk_gdma_32bit_cpy_S2L(kernel_addr + (t*param->C)/NPU_NUM*kernel_stride.c * sizeof(float) + idx*LOCAL_MEM_SIZE, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);
//         }
//         kernel_shape.c = param->C * (n-1);
//         okk_compact_stride(&kernel_stride, 0, &kernel_shape);

//         int iteration = DIV_UP(param->N, n);

//         Padding padding={.top=param->pad_top, .bottom=param->pad_bottom, .left=param->pad_left, .right=param->pad_right};
//         dim2 stride = {.h=param->stride_h, .w=param->stride_w};
//         dim2 dilation = {.h=param->dilation_h, .w=param->dilation_w};

//         for(int i=0; i<iteration; i++)
//         {
//             okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &input_shape, NO_USE, NO_USE);

//             okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, NO_USE, &input_shape, param->kernel_h, param->kernel_w, false, &padding, &stride, &dilation);

//             okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &output_shape, NO_USE, NO_USE);
//         }


void depthwise_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 output_stride, input_stride, kernel_stride, sys_input_stride, sys_output_stride;

    okk_continuous_stride(&sys_input_stride, &input_shape);
    okk_continuous_stride(&sys_output_stride, &output_shape);

    if(param->H == 224 || param->W == 256)
    {
        input_shape.n = 1;
        input_shape.c = param->N * param->C;
        output_shape.n = 1;
        output_shape.c = param->N * param->C;
        kernel_shape.c = param->N * param->C;

        okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
        okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
        okk_compact_stride(&kernel_stride, 0, &kernel_shape);

        local_addr_t input_addr, output_addr, kernel_addr;
        output_addr = 0;
        input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
        kernel_addr = input_addr + input_shape.n * input_stride.n * sizeof(float);

        kernel_shape.c = param->C;
        for(int t=0;t<param->N;t++)
        {
            int idx = (t*param->C)%NPU_NUM;
            okk_compact_stride(&kernel_stride, idx, &kernel_shape);
            okk_gdma_32bit_cpy_S2L(kernel_addr + (t*param->C)/NPU_NUM*kernel_stride.c * sizeof(float) + idx*LOCAL_MEM_SIZE, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);
        }
        kernel_shape.c = param->C * param->N;
        okk_compact_stride(&kernel_stride, 0, &kernel_shape);

        Padding padding={.top=param->pad_top, .bottom=param->pad_bottom, .left=param->pad_left, .right=param->pad_right};
        dim2 stride = {.h=param->stride_h, .w=param->stride_w};
        dim2 dilation = {.h=param->dilation_h, .w=param->dilation_w};

        okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &input_shape, NO_USE, NO_USE);

        okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, NO_USE, &input_shape, param->kernel_h, param->kernel_w, false, &padding, &stride, &dilation);

        okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &output_shape, NO_USE, NO_USE);

        okk_poll();
        return;
    }

    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    int max_N = (LOCAL_MEM_SIZE - kernel_size) / ((input_stride.n + output_stride.n)*sizeof(float));

    max_N = MIN(max_N, param->N);

    if(max_N > 1)
    {
        max_N /= 2;

        local_addr_t input_addr[2], kernel_addr, output_addr[2];
        output_addr[0] = 0;
        input_addr[0] = output_addr[0] + max_N * output_stride.n * sizeof(float);

        output_addr[1] = input_addr[0] + max_N * input_stride.n * sizeof(float);
        input_addr[1] = output_addr[1] + max_N * output_stride.n * sizeof(float);

        kernel_addr = input_addr[1] + max_N * input_stride.n * sizeof(float);

        okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);

        int iteration = DIV_UP(param->N, max_N);

        dim4 new_input_shape = {.n=max_N, .c=input_shape.c, .h=input_shape.h, .w=input_shape.w};
        dim4 new_output_shape = {.n=max_N, .c=output_shape.c, .h=output_shape.h, .w=output_shape.w};
        unsigned int input_tensor_size_global = max_N * new_input_shape.c * new_input_shape.h * new_input_shape.w * sizeof(float);
        unsigned int output_tensor_size_global = max_N * new_output_shape.c *new_output_shape.h *new_output_shape.w * sizeof(float);

        dim4 intput_shape_last = {.n=param->N - (iteration - 1)*max_N, .c=param->C, .h=param->H, .w=param->W};
        dim4 output_shape_last = {.n=param->N - (iteration - 1)*max_N, .c=param->C, .h=output_h, .w=output_w};

        Padding padding = {
            .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right};
        dim2 stride = {.h = param->stride_h, .w = param->stride_w};
        dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
 
        for(int i = 0; i < iteration + 2; i++)
        {
            okk_parallel_start();
            if(i<iteration)
                okk_gdma_32bit_cpy_S2L(input_addr[i%2], param->input_addr + i * input_tensor_size_global, i==iteration-1?&intput_shape_last:&new_input_shape, NO_USE, NO_USE);

            if(i>0 && i<iteration+1)
                okk_bdc_depthwise2d(output_addr[(i-1)%2], input_addr[(i-1)%2], kernel_addr, NO_USE, i == iteration?&intput_shape_last:&new_input_shape, param->kernel_h, param->kernel_w, false, &padding, &stride, &dilation);
            
            if(i>1)
                okk_gdma_32bit_cpy_L2S(param->output_addr + (i-2)*output_tensor_size_global, output_addr[i%2], i == iteration+1? &output_shape_last:&new_output_shape, NO_USE, NO_USE);
            
            okk_parallel_end();
        }
    }else
    {
        if(max_N == 1)
        {
            local_addr_t input_addr, kernel_addr, output_addr;
            output_addr = 0;
            input_addr = output_addr + max_N * output_stride.n * sizeof(float);
            kernel_addr = input_addr + max_N * input_stride.n * sizeof(float);

            okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);

            int temp_N = 0, now_N = max_N;
            while(temp_N < param->N)
            {
                if(temp_N + max_N > param->N)
                    max_N = param->N - temp_N;

                input_shape.n = max_N;
                output_shape.n = max_N;

                long input_offset = temp_N * param->C * param->H * param->W * sizeof(float);
                long output_offset = temp_N * param->C * output_h * output_w * sizeof(float);

                // OKKERNEL_LOG("temp_N:%d, input offset:%d, ouput offset:%d\n", temp_N, input_offset, output_offset);

                okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + input_offset, &input_shape, NO_USE, NO_USE);

                Padding padding = {
                    .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right};
                dim2 stride = {.h = param->stride_h, .w = param->stride_w};
                dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

                okk_bdc_depthwise2d(
                    output_addr,
                    input_addr,
                    kernel_addr,
                    NO_USE,
                    &input_shape,
                    param->kernel_h,
                    param->kernel_w,
                    false,
                    &padding,
                    &stride,
                    &dilation);

                okk_gdma_32bit_cpy_L2S(param->output_addr + output_offset, output_addr, &output_shape, NO_USE, NO_USE);

                temp_N += max_N;
            }
        }else
        {
            int split_output_H = 80;

            int iteration = DIV_UP(output_h, split_output_H);

            int split_input_H = (split_output_H - 1) * param->stride_h + kernel_h_ext;
            int last_output_H = output_h - (iteration-1) * split_output_H;
            int last_intput_H = (last_output_H - 1) * param->stride_h + kernel_h_ext;

            dim4 new_input_shape = {.n = param->N, .c=param->C, .h=split_input_H, .w=param->W};
            dim4 new_output_shape = {.n = param->N, .c=param->C, .h=split_output_H, .w=output_w};

            dim4 new_input_stride, new_output_stride;
            okk_128_byte_aligned_stride_for_32bit(&new_input_stride, 0, &new_input_shape);
            okk_128_byte_aligned_stride_for_32bit(&new_output_stride, 0, &new_output_shape);

            int max_N = MIN((LOCAL_MEM_SIZE - kernel_size) / (new_input_stride.n + new_output_stride.n) / sizeof(float), param->N);

            // OKKERNEL_LOG("max_N:%d\n", max_N);
            int N_iteration = param->N /max_N;

            int last_N = param->N - (N_iteration - 1)*max_N;

            dim2 stride = {.h = param->stride_h, .w = param->stride_w};
            dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

            int input_skip_N = max_N * param->C * param->H * param->W * sizeof(float);
            int output_skip_N = max_N * param->C * output_h * output_w * sizeof(float);

            local_addr_t input_addr, output_addr, kernel_addr;
            output_addr = 0;
            input_addr = output_addr + max_N * new_output_stride.n * sizeof(float);
            kernel_addr = input_addr + max_N * new_input_stride.n * sizeof(float);

            // OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) < LOCAL_MEM_SIZE);

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

                dim4 last_output_shape = {.n = new_input_shape.n, .c=param->C, .h=last_output_H, .w=output_w};
                dim4 last_output_stride;
                okk_128_byte_aligned_stride_for_32bit(&last_output_stride, 0, &last_output_shape);

                okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NO_USE);

                int input_skip_size_per_H = param->W * sizeof(float);
                int output_skip_size_per_iter = split_output_H * output_w * sizeof(float);

                dim4 temp_shape={.n=new_input_shape.n, .c=param->C, .h=new_input_shape.h, .w=param->W};
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
                            // OKKERNEL_LOG("now H:%d\n", (i==iteration-1)?last_intput_H:new_input_shape.h);
                            // OKKERNEL_LOG("last output H:%d\n", last_output_H);
                            // OKKERNEL_LOG("last input H:%d\n", last_intput_H);
                            // OKKERNEL_LOG("H after cal:%d\n", ((i==iteration-1)?last_intput_H:new_input_shape.h) - padding.bottom);
                            temp_shape.h = ((i==iteration-1)?last_intput_H:new_input_shape.h) - padding.bottom;
                            okk_128_byte_aligned_stride_for_32bit(&temp_stride, 0, &temp_shape);

                            use_temp = true;
                        }
                    }

                    // OKKERNEL_LOG("use temp:%d\n",use_temp);
                    // OKKERNEL_LOG("new_input_shape:[%d,%d,%d,%d], stride:[%d,%d,%d,%d]\n", new_input_shape.n, new_input_shape.c, new_input_shape.h, new_input_shape.w, new_input_stride.n, new_input_stride.c, new_input_stride.h, new_input_stride.w);
                    // OKKERNEL_LOG("temp_input_shape:[%d,%d,%d,%d], stride:[%d,%d,%d,%d]\n", temp_shape.n, temp_shape.c, temp_shape.h, temp_shape.w, temp_stride.n, temp_stride.c, temp_stride.h, temp_stride.w);
                    // OKKERNEL_LOG("padding:[%d,%d,%d,%d]\n", padding.top, padding.bottom, padding.left, padding.right);

                    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + n*input_skip_N + input_skip_H * input_skip_size_per_H, use_temp?&temp_shape:&new_input_shape, use_temp?&temp_stride:&new_input_stride, &sys_input_stride);

                    okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, NO_USE, use_temp?&temp_shape:&new_input_shape, param->kernel_h, param->kernel_w, false, &padding, &stride, &dilation);
                    // okk_bdc_conv2d(output_addr, input_addr, kernel_addr, NO_USE, use_temp?&temp_shape:&new_input_shape, param->OC, param->kernel_h, param->kernel_w, use_temp?&temp_stride:&new_input_stride, &kernel_stride_2IC, false, false, &padding, &stride, &dilation);

                    okk_gdma_32bit_cpy_L2S(param->output_addr + n*output_skip_N + i * output_skip_size_per_iter, output_addr, (i==iteration-1)?&last_output_shape:&new_output_shape, &sys_output_stride, (i==iteration-1)?&last_output_stride:&new_output_stride);
                }
            }
        }
    }
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(depthwise_contest);
