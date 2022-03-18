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

static void load_tensor(local_addr_t local_addr, system_addr_t system_addr, dim4 * shape)
{
    int n = shape->n;
    int c = shape->c;
    int h = shape->h;
    int w = shape->w;

    int npu_num = NPU_NUM;

    if( c % npu_num == 0 )
    {
        int channel_per_npu = c / npu_num;
        n = n * channel_per_npu;
        c = npu_num;
    }

    int rows = n;
    int cols = c * h * w;
    int cols_per_channel = h * w;

    if( cols < 65536 )
        okk_gdma_32bit_matrix_S2L(local_addr, system_addr, rows, cols, cols_per_channel, cols);
    else
        okk_gdma_32bit_cpy_S2L(local_addr, system_addr, shape, NULL, NULL);
}

static void save_tensor(system_addr_t system_addr, local_addr_t local_addr, dim4 * shape)
{
    int n = shape->n;
    int c = shape->c;
    int h = shape->h;
    int w = shape->w;

    int npu_num = NPU_NUM;

    if( c % npu_num == 0 )
    {
        int channel_per_npu = c / npu_num;
        n = n * channel_per_npu;
        c = npu_num;
    }

    int rows = n;
    int cols = c * h * w;
    int cols_per_channel = h * w;

    if( cols < 65536 )
        okk_gdma_32bit_matrix_L2S(system_addr, local_addr, rows, cols, cols_per_channel, cols);
    else
        okk_gdma_32bit_cpy_L2S(system_addr, local_addr, shape, NULL, NULL);
}

static void depthwise_demo(const param_t * param, int output_h, int output_w)
{
    dim4 output_shape = {.n = param->N, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    load_tensor(input_addr, param->input_addr, &input_shape);

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
            false, &padding, &stride, &dilation);

    save_tensor(param->output_addr, output_addr, &output_shape);
}

// 切分 n, 不作 pingpang 处理.
static void depthwise_splitn(const param_t * param, int output_h, int output_w)
{
    dim4 output_shape = {.n = 1, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    for(int n=0; n<param->N; n++)
    {
        load_tensor(input_addr, param->input_addr + n * param->C * param->H * param->W * sizeof(float), &input_shape);

        okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
                false, &padding, &stride, &dilation);

        save_tensor(param->output_addr + n * param->C * output_h * output_w * sizeof(float), output_addr, &output_shape);
    }
}

// 切分 n, 只有输入作 pingpang 处理.
static void depthwise_splitn_input(const param_t * param, int output_h, int output_w)
{
    dim4 output_shape = {.n = 1, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size * 2 + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr_1 = 0;
    local_addr_t input_addr_2 = input_addr_1 + input_size;
    local_addr_t output_addr = input_addr_2 + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    local_addr_t input_addr[2] = {input_addr_1, input_addr_2};

    Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    load_tensor(input_addr[0], param->input_addr, &input_shape);

    for(int n=0; n<param->N; n++)
    {
        okk_parallel_start();
        if( n + 1 < param->N )
            load_tensor(input_addr[(n+1)&1], param->input_addr + (n + 1) * param->C * param->H * param->W * sizeof(float), &input_shape);

        okk_bdc_depthwise2d(output_addr, input_addr[n&1], kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
                false, &padding, &stride, &dilation);
        okk_parallel_end();

        save_tensor(param->output_addr + n * param->C * output_h * output_w * sizeof(float), output_addr, &output_shape);
    }
}

// 切分 n, 输入输出都作 pingpang 处理.
static void depthwise_splitn_pingpang(const param_t * param, int output_h, int output_w)
{
    dim4 output_shape = {.n = 1, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size * 2 + output_size * 2 + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr_1 = 0;
    local_addr_t input_addr_2 = input_addr_1 + input_size;
    local_addr_t output_addr_1 = input_addr_2 + input_size;
    local_addr_t output_addr_2 = output_addr_1 + output_size;
    local_addr_t kernel_addr = output_addr_2 + output_size;

    local_addr_t input_addr[2] = {input_addr_1, input_addr_2};
    local_addr_t output_addr[2] = {output_addr_1, output_addr_2};

    Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    int n = 0;
    load_tensor(input_addr[n&1], param->input_addr + n * param->C * param->H * param->W * sizeof(float), &input_shape);

    for(n=1; n<param->N+1; n++)
    {
        int n1 = n + 1;
        okk_parallel_start();

        if( n > 1 )
            save_tensor(param->output_addr + (n - 2) * param->C * output_h * output_w * sizeof(float), output_addr[n&1], &output_shape);

        if( n < param->N )
            load_tensor(input_addr[n&1], param->input_addr + n * param->C * param->H * param->W * sizeof(float), &input_shape);

        okk_bdc_depthwise2d(output_addr[n1&1], input_addr[n1&1], kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
                false, &padding, &stride, &dilation);

        okk_parallel_end();
    }

    // n = param->N + 1
    save_tensor(param->output_addr + (n - 2) * param->C * output_h * output_w * sizeof(float), output_addr[n&1], &output_shape);
}

static void depthwise_splitn2_pingpang(const param_t * param, int output_h, int output_w)
{
    dim4 output_shape = {.n = 2, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = 2, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size * 2 + output_size * 2 + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr_1 = 0;
    local_addr_t input_addr_2 = input_addr_1 + input_size;
    local_addr_t output_addr_1 = input_addr_2 + input_size;
    local_addr_t output_addr_2 = output_addr_1 + output_size;
    local_addr_t kernel_addr = output_addr_2 + output_size;

    local_addr_t input_addr[2] = {input_addr_1, input_addr_2};
    local_addr_t output_addr[2] = {output_addr_1, output_addr_2};

    Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    //// param->N == 4
    load_tensor(input_addr[0], param->input_addr, &input_shape);

    okk_parallel_start();
    load_tensor(input_addr[1], param->input_addr + 2 * param->C * param->H * param->W * sizeof(float), &input_shape);

    okk_bdc_depthwise2d(output_addr[0], input_addr[0], kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w,
            false, &padding, &stride, &dilation);
    okk_parallel_end();

    okk_parallel_start();
    save_tensor(param->output_addr, output_addr[0], &output_shape);

    okk_bdc_depthwise2d(output_addr[1], input_addr[1], kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w,
            false, &padding, &stride, &dilation);
    okk_parallel_end();

    save_tensor(param->output_addr + 2 * param->C * output_h * output_w * sizeof(float), output_addr[1], &output_shape);
}

static void depthwise_n2c(const param_t * param, int output_h, int output_w)
{
    dim4 output_shape = {.n = 1, .c = param->N * param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->N * param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->N * param->C, .h = 1, .w = param->kernel_h * param->kernel_w};
    dim4 kernel_shape_single = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    load_tensor(input_addr, param->input_addr, &input_shape);

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape_single, &kernel_stride, NULL);
    for(int i=1; i<param->N; i++)
    {
        okk_gdma_32bit_cpy_L2L(kernel_addr + LOCAL_MEM_SIZE * i * param->C, kernel_addr, &kernel_shape_single, &kernel_stride, &kernel_stride);
    }

    okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
            false, &padding, &stride, &dilation);

    save_tensor(param->output_addr, output_addr, &output_shape);
}

// 切分 out_h, 不作 pingpang 处理. 
static void depthwise_splith(const param_t * param, int output_h, int output_w, int out_h_split)
{
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int in_h_split = (out_h_split - 1) * param->stride_h + kernel_h_ext;

    dim4 output_shape = {.n = param->N, .c = param->C, .h = out_h_split, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->C, .h = in_h_split, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    dim4 output_shape_all = {.n = param->N, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape_all = {.n = param->N, .c = param->C, .h = param->H, .w = param->W};
    dim4 input_stride_cont, output_stride_cont;
    okk_continuous_stride(&input_stride_cont, &input_shape_all);
    okk_continuous_stride(&output_stride_cont, &output_shape_all);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    int out_h_start;
    for(out_h_start=0; out_h_start<output_h; out_h_start += out_h_split)
    {
        int out_h_cur = MIN(output_h - out_h_start, out_h_split);
        output_shape.h = out_h_cur;

        int in_h_start = out_h_start * param->stride_h - param->pad_top;
        int in_h_cur = (out_h_cur - 1) * param->stride_h + kernel_h_ext;
        int in_h_end = in_h_start + in_h_cur;

        Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };

        if( in_h_start < 0 )
            in_h_start = 0;
        else
            padding.top = 0;

        if( in_h_end > param->H )
            in_h_end = param->H;
        else
            padding.bottom = 0;

        in_h_cur = in_h_end - in_h_start;  // 实际有效的输入高度
        input_shape.h = in_h_cur;

        okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + in_h_start * param->W * sizeof(float), &input_shape, NULL, &input_stride_cont);

        okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
                false, &padding, &stride, &dilation);

        okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start * output_w * sizeof(float), output_addr, &output_shape, &output_stride_cont, NULL);
    }
}

// 切分 h, 并且做 pingpang 处理.
static void depthwise_splith_pingpang(const param_t * param, int output_h, int output_w, int out_h_split)
{
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int in_h_split = (out_h_split - 1) * param->stride_h + kernel_h_ext;

    dim4 output_shape = {.n = param->N, .c = param->C, .h = out_h_split, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->C, .h = in_h_split, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = 1, .w = param->kernel_h * param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    dim4 output_shape_all = {.n = param->N, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape_all = {.n = param->N, .c = param->C, .h = param->H, .w = param->W};
    dim4 input_stride_cont, output_stride_cont;
    okk_continuous_stride(&input_stride_cont, &input_shape_all);
    okk_continuous_stride(&output_stride_cont, &output_shape_all);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size * 2 + output_size * 2 + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr_1 = 0;
    local_addr_t input_addr_2 = input_addr_1 + input_size;
    local_addr_t output_addr_1 = input_addr_2 + input_size;
    local_addr_t output_addr_2 = output_addr_1 + output_size;
    local_addr_t kernel_addr = output_addr_2 + output_size;

    local_addr_t input_addr[2] = {input_addr_1, input_addr_2};
    local_addr_t output_addr[2] = {output_addr_1, output_addr_2};

    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape, &kernel_stride, NULL);

    int index = 0;

    Padding padding_prev;
    dim4 input_shape_prev;
    dim4 output_shape_prev;
    dim4 output_shape_prev2;
    int out_h_start_prev = 0, out_h_start_prev2 = 0;

    int out_h_start;
    for(out_h_start=0; out_h_start<output_h; out_h_start += out_h_split)
    {
        int out_h_cur = MIN(output_h - out_h_start, out_h_split);
        output_shape.h = out_h_cur;

        int in_h_start = out_h_start * param->stride_h - param->pad_top;
        int in_h_cur = (out_h_cur - 1) * param->stride_h + kernel_h_ext;
        int in_h_end = in_h_start + in_h_cur;

        Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };

        if( in_h_start < 0 )
            in_h_start = 0;
        else
            padding.top = 0;

        if( in_h_end > param->H )
            in_h_end = param->H;
        else
            padding.bottom = 0;

        in_h_cur = in_h_end - in_h_start;  // 实际有效的输入高度
        input_shape.h = in_h_cur;
        int index1 = index + 1;

        okk_parallel_start();

        okk_gdma_32bit_cpy_S2L(input_addr[index&1], param->input_addr + in_h_start * param->W * sizeof(float), &input_shape, NULL, &input_stride_cont);

        if( out_h_start > 0 )
            okk_bdc_depthwise2d(output_addr[index1&1], input_addr[index1&1], kernel_addr, 0, &input_shape_prev, param->kernel_h, param->kernel_w,
                    false, &padding_prev, &stride, &dilation);

        if( out_h_start_prev > 0 )
            okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start_prev2 * output_w * sizeof(float), output_addr[index&1], &output_shape_prev2, &output_stride_cont, NULL);

        okk_parallel_end();

        padding_prev = padding;
        input_shape_prev = input_shape;
        output_shape_prev2 = output_shape_prev;
        output_shape_prev = output_shape;
        out_h_start_prev2 = out_h_start_prev;
        out_h_start_prev = out_h_start;
        index++;
    }

    {
        int index1 = index + 1;
        okk_parallel_start();
        okk_bdc_depthwise2d(output_addr[index1&1], input_addr[index1&1], kernel_addr, 0, &input_shape_prev, param->kernel_h, param->kernel_w,
                false, &padding_prev, &stride, &dilation);

        if( out_h_start_prev > 0 )
            okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start_prev2 * output_w * sizeof(float), output_addr[index&1], &output_shape_prev2, &output_stride_cont, NULL);
        okk_parallel_end();

        output_shape_prev2 = output_shape_prev;
        out_h_start_prev2 = out_h_start_prev;
        index++;
    }

    // last data.
    okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start_prev2 * output_w * sizeof(float), output_addr[index&1], &output_shape_prev2, &output_stride_cont, NULL);
}

static void depthwise_n2c_splith(const param_t * param, int output_h, int output_w, int out_h_split)
{
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int in_h_split = (out_h_split - 1) * param->stride_h + kernel_h_ext;

    dim4 output_shape = {.n = 1, .c = param->N * param->C, .h = out_h_split, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->N * param->C, .h = in_h_split, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->N * param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_shape_single = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    dim4 output_shape_all = {.n = 1, .c = param->N * param->C, .h = output_h, .w = output_w};
    dim4 input_shape_all = {.n = 1, .c = param->N * param->C, .h = param->H, .w = param->W};
    dim4 input_stride_cont, output_stride_cont;
    okk_continuous_stride(&input_stride_cont, &input_shape_all);
    okk_continuous_stride(&output_stride_cont, &output_shape_all);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape_single, &kernel_stride, NULL);
    for(int i=1; i<param->N; i++)
    {
        okk_gdma_32bit_cpy_L2L(kernel_addr + i * param->C * LOCAL_MEM_SIZE, kernel_addr, &kernel_shape_single, &kernel_stride, &kernel_stride);
    }

    int out_h_start;
    for(out_h_start=0; out_h_start<output_h; out_h_start += out_h_split)
    {
        int out_h_cur = MIN(output_h - out_h_start, out_h_split);
        output_shape.h = out_h_cur;

        int in_h_start = out_h_start * param->stride_h - param->pad_top;
        int in_h_cur = (out_h_cur - 1) * param->stride_h + kernel_h_ext;
        int in_h_end = in_h_start + in_h_cur;

        Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };

        if( in_h_start < 0 )
            in_h_start = 0;
        else
            padding.top = 0;

        if( in_h_end > param->H )
            in_h_end = param->H;
        else
            padding.bottom = 0;

        in_h_cur = in_h_end - in_h_start;  // 实际有效的输入高度
        input_shape.h = in_h_cur;

        okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr + in_h_start * param->W * sizeof(float), &input_shape, NULL, &input_stride_cont);

        okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w, 
                false, &padding, &stride, &dilation);

        okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start * output_w * sizeof(float), output_addr, &output_shape, &output_stride_cont, NULL);
    }
}

static void depthwise_n2c_splith_pingpang(const param_t * param, int output_h, int output_w, int out_h_split)
{
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int in_h_split = (out_h_split - 1) * param->stride_h + kernel_h_ext;

    dim4 output_shape = {.n = 1, .c = param->N * param->C, .h = out_h_split, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->N * param->C, .h = in_h_split, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->N * param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_shape_single = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    dim4 output_shape_all = {.n = 1, .c = param->N * param->C, .h = output_h, .w = output_w};
    dim4 input_shape_all = {.n = 1, .c = param->N * param->C, .h = param->H, .w = param->W};
    dim4 input_stride_cont, output_stride_cont;
    okk_continuous_stride(&input_stride_cont, &input_shape_all);
    okk_continuous_stride(&output_stride_cont, &output_shape_all);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size * 2 + output_size * 2 + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr_1 = 0;
    local_addr_t input_addr_2 = input_addr_1 + input_size;
    local_addr_t output_addr_1 = input_addr_2 + input_size;
    local_addr_t output_addr_2 = output_addr_1 + output_size;
    local_addr_t kernel_addr = output_addr_2 + output_size;

    local_addr_t input_addr[2] = {input_addr_1, input_addr_2};
    local_addr_t output_addr[2] = {output_addr_1, output_addr_2};

    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape_single, &kernel_stride, NULL);
    for(int i=1; i<param->N; i++)
    {
        okk_gdma_32bit_cpy_L2L(kernel_addr + i * param->C * LOCAL_MEM_SIZE, kernel_addr, &kernel_shape_single, &kernel_stride, &kernel_stride);
    }

    int index = 0;

    Padding padding_prev;
    dim4 input_shape_prev;
    dim4 output_shape_prev;
    dim4 output_shape_prev2;
    int out_h_start_prev = 0, out_h_start_prev2 = 0;

    int out_h_start;
    for(out_h_start=0; out_h_start<output_h; out_h_start += out_h_split)
    {
        int out_h_cur = MIN(output_h - out_h_start, out_h_split);
        output_shape.h = out_h_cur;

        int in_h_start = out_h_start * param->stride_h - param->pad_top;
        int in_h_cur = (out_h_cur - 1) * param->stride_h + kernel_h_ext;
        int in_h_end = in_h_start + in_h_cur;

        Padding padding = { .top = param->pad_top, .bottom = param->pad_bottom, .left = param->pad_left, .right = param->pad_right };

        if( in_h_start < 0 )
            in_h_start = 0;
        else
            padding.top = 0;

        if( in_h_end > param->H )
            in_h_end = param->H;
        else
            padding.bottom = 0;

        in_h_cur = in_h_end - in_h_start;  // 实际有效的输入高度
        input_shape.h = in_h_cur;
        int index1 = index + 1;

        okk_parallel_start();

        okk_gdma_32bit_cpy_S2L(input_addr[index&1], param->input_addr + in_h_start * param->W * sizeof(float), &input_shape, NULL, &input_stride_cont);

        if( out_h_start > 0 )
            okk_bdc_depthwise2d(output_addr[index1&1], input_addr[index1&1], kernel_addr, 0, &input_shape_prev, param->kernel_h, param->kernel_w,
                    false, &padding_prev, &stride, &dilation);

        if( out_h_start_prev > 0 )
            okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start_prev2 * output_w * sizeof(float), output_addr[index&1], &output_shape_prev2, &output_stride_cont, NULL);

        okk_parallel_end();

        padding_prev = padding;
        input_shape_prev = input_shape;
        output_shape_prev2 = output_shape_prev;
        output_shape_prev = output_shape;
        out_h_start_prev2 = out_h_start_prev;
        out_h_start_prev = out_h_start;
        index++;
    }

    {
        int index1 = index + 1;
        okk_parallel_start();
        okk_bdc_depthwise2d(output_addr[index1&1], input_addr[index1&1], kernel_addr, 0, &input_shape_prev, param->kernel_h, param->kernel_w,
                false, &padding_prev, &stride, &dilation);

        if( out_h_start_prev > 0 )
            okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start_prev2 * output_w * sizeof(float), output_addr[index&1], &output_shape_prev2, &output_stride_cont, NULL);
        okk_parallel_end();

        output_shape_prev2 = output_shape_prev;
        out_h_start_prev2 = out_h_start_prev;
        index++;
    }

    // last data.
    okk_gdma_32bit_cpy_L2S(param->output_addr + out_h_start_prev2 * output_w * sizeof(float), output_addr[index&1], &output_shape_prev2, &output_stride_cont, NULL);
}

static void depthwise_nh2c(const param_t * param, int output_h, int output_w, int out_h_split)
{
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int in_h_split = (out_h_split - 1) * param->stride_h + kernel_h_ext;

    int split_h_count = DIV_UP(output_h, out_h_split);

    dim4 output_shape = {.n = 1, .c = split_h_count * param->N * param->C, .h = out_h_split, .w = output_w};
    dim4 input_shape = {.n = 1, .c = split_h_count * param->N * param->C, .h = in_h_split, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = split_h_count * param->N * param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_shape_single = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};

    dim4 output_stride, input_stride, kernel_stride;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);

    dim4 output_shape_all = {.n = 1, .c = param->N * param->C, .h = output_h, .w = output_w};
    dim4 input_shape_all = {.n = 1, .c = param->N * param->C, .h = param->H, .w = param->W};
    dim4 input_stride_cont, output_stride_cont;
    okk_continuous_stride(&input_stride_cont, &input_shape_all);
    okk_continuous_stride(&output_stride_cont, &output_shape_all);

    unsigned int output_size = output_shape.n * output_stride.n * sizeof(float);
    unsigned int input_size = input_shape.n * input_stride.n * sizeof(float);
    unsigned int kernel_size = kernel_shape.n * kernel_stride.n * sizeof(float);
    unsigned int total_size = input_size + output_size + kernel_size;

    if( total_size > LOCAL_MEM_SIZE )
    {
        OKKERNEL_LOG("out of memory. %d\n", total_size);
        OKKERNEL_LOG("input: %d output: %d kernel: %d\n", input_size, output_size, kernel_size);
        return;
    }

    local_addr_t input_addr = 0;
    local_addr_t output_addr = input_addr + input_size;
    local_addr_t kernel_addr = output_addr + output_size;

    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};

    Padding padding = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };

    okk_gdma_32bit_cpy_S2L(kernel_addr, param->kernel_addr, &kernel_shape_single, &kernel_stride, NULL);
    for(int i=1; i<param->N; i++)
    {
        okk_gdma_32bit_cpy_L2L(kernel_addr + i * param->C * LOCAL_MEM_SIZE, kernel_addr, &kernel_shape_single, &kernel_stride, &kernel_stride);
    }
    kernel_shape_single.c = param->N * param->C;
    for(int i=1; i<split_h_count; i++)
    {
        okk_gdma_32bit_cpy_L2L(kernel_addr + i * param->C * param->N * LOCAL_MEM_SIZE, kernel_addr, &kernel_shape_single, &kernel_stride, &kernel_stride);
    }

    x32 zero; zero.fp32 = 0;
    okk_bdc_32bit_set_C(input_addr, zero, &input_shape, NULL);

    int out_h_start;
    int npu_idx = 0;
    for(out_h_start = 0; out_h_start < output_h; out_h_start += out_h_split)
    {
        int out_h_cur = MIN(output_h - out_h_start, out_h_split);

        int in_h_start = out_h_start * param->stride_h - param->pad_top;
        int in_h_cur = (out_h_cur - 1) * param->stride_h + kernel_h_ext;
        int in_h_end = in_h_start + in_h_cur;
        int pad_top = 0;

        if( in_h_start < 0 )
        {
            pad_top = - in_h_start;
            in_h_start = 0;
        }
        if( in_h_end > param->H )
        {
            in_h_end = param->H;
        }
        in_h_cur = in_h_end - in_h_start;  // 实际有效的输入高度

        dim4 shape_cur = {1, param->N * param->C, in_h_cur, param->W};
        okk_gdma_32bit_cpy_S2L(input_addr + npu_idx * LOCAL_MEM_SIZE + pad_top * input_stride.h * sizeof(float),
                param->input_addr + in_h_start * input_stride_cont.h * sizeof(float),
                &shape_cur, NULL, &input_stride_cont);

        npu_idx += param->N * param->C;
    }

    okk_bdc_depthwise2d(output_addr, input_addr, kernel_addr, 0, &input_shape, param->kernel_h, param->kernel_w,
            false, &padding, &stride, &dilation);

    npu_idx = 0;
    for(out_h_start = 0; out_h_start < output_h; out_h_start += out_h_split)
    {
        int out_h_cur = MIN(output_h - out_h_start, out_h_split);
        dim4 shape_cur = {1, param->N * param->C, out_h_cur, output_w};
        okk_gdma_32bit_cpy_L2S(param->output_addr + output_stride_cont.h * out_h_start * sizeof(float),
                output_addr + LOCAL_MEM_SIZE * npu_idx, &shape_cur, &output_stride_cont, NULL);
        npu_idx += param->N * param->C;
    }
}

void depthwise_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    (void)(param);
    // TODO
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    if( param->C == 3 && param->kernel_h == 11 )
    {   // 0
	    (void)depthwise_nh2c;
        // depthwise_n2c(param, output_h, output_w);
        depthwise_nh2c(param, output_h, output_w, 11);
    }
    else if( param->C == 3 && param->kernel_h == 7 )
    {   // 1
	  (void)depthwise_n2c;
        // depthwise_n2c(param, output_h, output_w);
	depthwise_nh2c(param, output_h, output_w, 32);
    }
    else if( param->C == 3 && param->H == 640 )
    {   // 2
	    (void)depthwise_n2c_splith;
	    (void)depthwise_n2c_splith_pingpang;
         depthwise_n2c_splith_pingpang(param, output_h, output_w, 40);
	// depthwise_nh2c(param, output_h, output_w, 64);
    }
    else if( param->H == 640 )
    {   // 2, slow mode.
        depthwise_splith(param, output_h, output_w, 8);
    }
    else if( param->C == 96 && param->H == 150 )
    {   // 3
        depthwise_splitn_pingpang(param, output_h, output_w);
    }
    else if( param->C == 144 && param->H == 75 )
    {   // 4
        depthwise_splitn_pingpang(param, output_h, output_w);
    }
    else if( param->C == 192 && param->H == 38 )
    {   // 5
	// depthwise_splith_pingpang(param, output_h, output_w, 10);
        depthwise_splitn2_pingpang(param, output_h, output_w);
    }
    else if( param->C == 336 && param->H == 29 )
    {   // 6
        // depthwise_splitn_pingpang(param, output_h, output_w);
	// depthwise_demo(param, output_h, output_w);
        depthwise_splitn2_pingpang(param, output_h, output_w);
    }
    else if( param->C == 512 && param->H == 14 )
    {   // 7
	depthwise_demo(param, output_h, output_w);
    }
    else if( param->C == 960 && param->H == 28 )
    {   // 8
        depthwise_splitn2_pingpang(param, output_h, output_w);
    }
    else if( param->C == 2048 && param->H == 33 )
    {   // 9, try split c.
        depthwise_splitn_input(param, output_h, output_w);
    }
    else if(param->N )
    {
        depthwise_splitn(param, output_h, output_w);
    }
    else
    {
        depthwise_demo(param, output_h, output_w);
	(void)depthwise_splith_pingpang;
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(depthwise_contest);


