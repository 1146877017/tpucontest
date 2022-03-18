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

static void softmax_x(param_t * param)
{
    system_addr_t l2sram_addr = okk_l2_sram_start_addr();
    dim4 shape_l2sram = {.n = 1, .c = 1, .h = 1, .w = param->N * param->C * param->H * param->W };
    okk_gdma_32bit_cpy_S2S(l2sram_addr, param->input_addr, &shape_l2sram, NULL, NULL);

    local_addr_t work_addr = 0;
    local_addr_t sum_addr = 0;
    local_addr_t max_addr = 0;
    local_addr_t input_addr = 0;
    local_addr_t output_addr = 0;

    dim4 shape_all;
    dim4 shape_one;
    dim4 align_stride;
    dim4 zero_stride;

    if( param->N == 1 )
    {
        shape_all.n = param->C;
        shape_all.c = param->H;
        shape_all.h = 1;
        shape_all.w = param->W;

        shape_one.n = 1;
        shape_one.c = param->H;
        shape_one.h = 1;
        shape_one.w = param->W;
    }
    else
    {
        return;
    }

    okk_128_byte_aligned_stride_for_32bit(&align_stride, 0, &shape_one);

    zero_stride.n = 0;
    zero_stride.c = align_stride.c;
    zero_stride.h = align_stride.h;
    zero_stride.w = align_stride.w;

    sum_addr = work_addr + shape_all.n * align_stride.n * sizeof(float);
    max_addr = sum_addr + align_stride.n * sizeof(float);
    input_addr = max_addr + align_stride.n * sizeof(float);
    output_addr = input_addr + shape_all.n * align_stride.n * sizeof(float);

    okk_gdma_32bit_cpy_S2L(input_addr, l2sram_addr, &shape_all, NULL, NULL);

    okk_bdc_max(max_addr, input_addr, input_addr+sizeof(float)*align_stride.n, &shape_one, NULL, NULL, NULL);
    for(int i=2; i<param->C; i++)
    {
        okk_bdc_max(max_addr, input_addr + sizeof(float) * align_stride.n * i, max_addr, &shape_one, NULL, NULL, NULL);
    }

    okk_bdc_sub(input_addr, input_addr, max_addr, &shape_all, NULL, NULL, &zero_stride);

    okk_bdc_exp(output_addr, input_addr, work_addr, &shape_all);

    okk_bdc_add(sum_addr, output_addr, output_addr + sizeof(float) * align_stride.n, &shape_one, NULL, NULL, NULL);
    for(int i=2; i<param->C; i++)
    {
        okk_bdc_add(sum_addr, output_addr + sizeof(float) * align_stride.n * i, sum_addr, &shape_one, NULL, NULL, NULL);
    }

    okk_bdc_reciprocal(sum_addr, sum_addr, &shape_one, NULL, NULL);

    okk_bdc_mul(output_addr, output_addr, sum_addr, &shape_all, NULL, NULL, &zero_stride);

    okk_gdma_32bit_cpy_L2S(l2sram_addr, output_addr, &shape_all, NULL, NULL);

    okk_gdma_32bit_cpy_S2S(param->output_addr, l2sram_addr, &shape_l2sram, NULL, NULL);
}

void softmax_0(param_t * param)
{
    int rows = 370 / 6;    // 61
    int rows_1 = rows + 1;  // 62
    int rows_all = 64;
    int cols = 13 * 13 * 6;
    int cols_per_channel = 16;
    int row_stride = cols;

    dim4 matrix_shape = {rows, 64, 1, 16};
    dim4 matrix_shape_1 = {rows_1, 64, 1, 16};
    dim4 matrix_shape_tmp = {0, 64, 1, 16};
    dim4 matrix_stride;
    okk_128_byte_aligned_stride_for_32bit(&matrix_stride, 0, &matrix_shape);
    dim4 matrix_stride_zero = {0, matrix_stride.c, matrix_stride.h, matrix_stride.w};

    dim4 remain_shape = {1, 1, 1, 4 * 13 * 13};
    dim4 padded_shape = {1, 1, 1, 14 * 13 * 13};

    dim4 tensor_shape_six = {6, 13, 1, 13};
    dim4 tensor_shape_three = {3, 13, 1, 13};
    dim4 tensor_shape_one = {1, 13, 1, 13};
    dim4 tensor_stride;
    okk_128_byte_aligned_stride_for_32bit(&tensor_stride, 0, &tensor_shape_six);
    dim4 tensor_stride_zero = {0, tensor_stride.c, tensor_stride.h, tensor_stride.w};

    system_addr_t l2sram_addr = okk_l2_sram_start_addr();

    local_addr_t sum_addr = 0;
    local_addr_t max_addr = 0;
    local_addr_t input_addr = 0;
    local_addr_t output_addr = 0;
    local_addr_t work_addr = 0;
    local_addr_t six_addr = 0;

    input_addr = max_addr + 32 * sizeof(float);
    output_addr = input_addr + 32 * rows_all * sizeof(float);
    work_addr = output_addr + 32 * rows_all * sizeof(float);
    six_addr = work_addr + 32 * rows_all * sizeof(float);

    x32 zero; 
    zero.fp32 = 0;

    // data transfer and pad.
    okk_gdma_32bit_matrix_S2L(input_addr, param->input_addr, rows, cols, cols_per_channel, row_stride);
#if 1
    okk_gdma_32bit_matrix_S2L(input_addr + sizeof(float) * 32 * rows, param->input_addr + sizeof(float) * rows * 6 * 13 * 13, 
            1, 4 * 13 * 13, cols_per_channel, row_stride);
#else    
    okk_gdma_32bit_cpy_S2S(l2sram_addr, param->input_addr + sizeof(float) * rows * 6 * 13 * 13, &remain_shape, NULL, NULL);
    okk_gdma_32bit_set_C_system(l2sram_addr + sizeof(float) * 4 * 13 * 13, zero, &padded_shape, NULL); // data padding for max step.
    okk_gdma_32bit_matrix_S2L(input_addr + sizeof(float) * 32 * rows, l2sram_addr, rows_all - rows, cols, cols_per_channel, row_stride);
#endif

#if 0  // It works without this max step.
    matrix_shape_tmp.n = 32;
    okk_bdc_max(work_addr, input_addr, input_addr + sizeof(float) * 32 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 16;
    okk_bdc_max(work_addr, work_addr, work_addr + sizeof(float) * 16 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 8;
    okk_bdc_max(work_addr, work_addr, work_addr + sizeof(float) * 8 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 4;
    okk_bdc_max(work_addr, work_addr, work_addr + sizeof(float) * 4 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 2;
    okk_bdc_max(work_addr, work_addr, work_addr + sizeof(float) * 2 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 1;
    okk_bdc_max(max_addr, work_addr, work_addr + sizeof(float) * 1 * 32, &matrix_shape_tmp, NULL, NULL, NULL);

    //// 
    okk_gdma_32bit_matrix_L2S(l2sram_addr, max_addr, 1, cols, cols_per_channel, row_stride);
    okk_gdma_32bit_cpy_S2L(six_addr, l2sram_addr, &tensor_shape_six, NULL, NULL);

    okk_bdc_max(work_addr, six_addr, six_addr + 32 * sizeof(float) * 3, &tensor_shape_three, NULL, NULL, NULL);
    okk_bdc_max(max_addr, work_addr, work_addr + 32 * sizeof(float), &tensor_shape_one, NULL, NULL, NULL);
    okk_bdc_max(max_addr, max_addr, work_addr + 32 * sizeof(float) * 2, &tensor_shape_one, NULL, NULL, NULL);

    okk_gdma_32bit_cpy_L2S(l2sram_addr, max_addr, &tensor_shape_six, NULL, &tensor_stride_zero);
    okk_gdma_32bit_matrix_S2L(max_addr, l2sram_addr, 1, cols, cols_per_channel, row_stride);

    okk_bdc_sub(input_addr, input_addr, max_addr, &matrix_shape_1, NULL, NULL, &matrix_stride_zero);
#endif
    // okk_bdc_exp(output_addr, input_addr, work_addr, &matrix_shape_1);
    okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &matrix_shape_1, 5);

#if 1
    // set pads to zeros.
    dim4 shape_t1 = {1, 1, 1, 12};
    okk_gdma_32bit_set_C_local(output_addr + sizeof(float) * 32 * rows + LOCAL_MEM_SIZE * 42 + 4 * sizeof(float), zero, &shape_t1, NULL);
    dim4 shape_t2 = {1, 21, 1, 16};
    okk_gdma_32bit_set_C_local(output_addr + sizeof(float) * 32 * rows + LOCAL_MEM_SIZE * 43, zero, &shape_t2, NULL);

    int left_cols_per_channel = 16;
    dim4 left_shape = {1, DIV_UP(rows_1, left_cols_per_channel), 1, left_cols_per_channel};

    x32 one;
    one.fp32 = 1.0f;

    okk_gdma_32bit_set_C_local(work_addr, one, &left_shape, NULL);
    okk_bdc_matmul(sum_addr, work_addr, output_addr, 0, 1, rows_1, cols, left_cols_per_channel, cols_per_channel, false, false);

#else
    // set pads to zeros.
    okk_gdma_32bit_matrix_L2S(l2sram_addr, output_addr + sizeof(float) * 32 * rows, 1, cols, cols_per_channel, row_stride);
    okk_gdma_32bit_set_C_system(l2sram_addr + sizeof(float) * 4 * 13 * 13, zero, &padded_shape, NULL);
    okk_gdma_32bit_matrix_S2L(output_addr + sizeof(float) * 32 * rows, l2sram_addr, rows_all - rows, cols, cols_per_channel, row_stride);

    matrix_shape_tmp.n = 32;
    okk_bdc_add(work_addr, output_addr, output_addr + sizeof(float) * 32 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 16;
    okk_bdc_add(work_addr, work_addr, work_addr + sizeof(float) * 16 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 8;
    okk_bdc_add(work_addr, work_addr, work_addr + sizeof(float) * 8 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 4;
    okk_bdc_add(work_addr, work_addr, work_addr + sizeof(float) * 4 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 2;
    okk_bdc_add(work_addr, work_addr, work_addr + sizeof(float) * 2 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
    matrix_shape_tmp.n = 1;
    okk_bdc_add(sum_addr, work_addr, work_addr + sizeof(float) * 1 * 32, &matrix_shape_tmp, NULL, NULL, NULL);
#endif

    ////// 
    okk_gdma_32bit_matrix_L2S(l2sram_addr, sum_addr, 1, cols, cols_per_channel, row_stride);
    okk_gdma_32bit_cpy_S2L(six_addr, l2sram_addr, &tensor_shape_six, NULL, NULL);

#if 1
    okk_bdc_matmul(sum_addr, work_addr, six_addr, 0, 1, 6, 13 * 13, left_cols_per_channel, 13, false, false);
#else
    okk_bdc_add(work_addr, six_addr, six_addr + 32 * sizeof(float) * 3, &tensor_shape_three, NULL, NULL, NULL);
    okk_bdc_add(sum_addr, work_addr, work_addr + 32 * sizeof(float), &tensor_shape_one, NULL, NULL, NULL);
    okk_bdc_add(sum_addr, sum_addr, work_addr + 32 * sizeof(float) * 2, &tensor_shape_one, NULL, NULL, NULL);
#endif

    okk_gdma_32bit_cpy_L2S(l2sram_addr, sum_addr, &tensor_shape_six, NULL, &tensor_stride_zero);
    okk_gdma_32bit_matrix_S2L(sum_addr, l2sram_addr, 1, cols, cols_per_channel, row_stride);

    okk_bdc_div(output_addr, output_addr, sum_addr, &matrix_shape_1, NULL, NULL, &matrix_stride_zero);

    // data output.
    okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, rows, cols, cols_per_channel, row_stride);
    okk_gdma_32bit_matrix_L2S(param->output_addr + sizeof(float) * rows * 6 * 13 * 13,
            output_addr + sizeof(float) * 32 * rows, 1, 4 * 13 * 13, cols_per_channel, row_stride);
}

static void softmax_1(param_t * param)
{
    int total_num = 1000;
    local_addr_t input_addr = 0;
    int cols_per_channel = 16; //32;

    okk_gdma_32bit_matrix_S2L(input_addr, param->input_addr, 1, total_num, cols_per_channel, total_num);

    local_addr_t output_addr = input_addr + 32 * sizeof(float);
    local_addr_t sum_addr = output_addr + 32 * sizeof(float);
    local_addr_t work_addr = sum_addr + 32 * sizeof(float);

    int channel = DIV_UP(total_num, cols_per_channel);
    dim4 matrix_shape = {1, channel, 1, cols_per_channel};
    dim4 matrix_2d_shape = {total_num, channel, 1, cols_per_channel};

    // okk_bdc_exp(output_addr, input_addr, work_addr, &matrix_shape);
    okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &matrix_shape, 6);

    x32 one;
    one.fp32 = 1;
    okk_bdc_32bit_set_C(work_addr, one, &matrix_2d_shape, NULL);
    okk_bdc_matmul(sum_addr, output_addr, work_addr, 0, 1, 1000, 1000, cols_per_channel, cols_per_channel, false, false);
    okk_bdc_div(output_addr, output_addr, sum_addr, &matrix_shape, NULL, NULL, NULL);

    // data output.
    okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, 1, 1000, cols_per_channel, 1000);
}

static void softmax_2(param_t * param)
{
    int batch_step = sizeof(float) * 2 * 157 * 283;
    int rows = 2;
    int cols = 157 * 283;
    int cols_per_channel = 64;
    int row_stride = cols;

    dim4 matrix_shape = {rows, 695, 1, 64};
    dim4 matrix_shape_one = {1, 695, 1, 64};
    dim4 matrix_stride;
    okk_128_byte_aligned_stride_for_32bit(&matrix_stride, 0, &matrix_shape);
    dim4 matrix_stride_zero = {0, matrix_stride.c, matrix_stride.h, matrix_stride.w};

    local_addr_t sum_addr = 0;
    local_addr_t max_addr = 0;
    local_addr_t input_addr = 0;
    local_addr_t middle_addr = 0;
    local_addr_t output_addr = 0;
    local_addr_t work_addr = 0;

    input_addr = max_addr + matrix_stride.n * sizeof(float);
    middle_addr = input_addr + matrix_stride.n * sizeof(float) * rows;
    output_addr = middle_addr + matrix_stride.n * sizeof(float) * rows;
    work_addr = output_addr + matrix_stride.n * sizeof(float) * rows;

    okk_gdma_32bit_matrix_S2L(input_addr, param->input_addr, rows, cols, cols_per_channel, row_stride);
    for(int b = 0; b<4; b++)
    {
#if 0
        okk_bdc_max(max_addr, input_addr, input_addr + sizeof(float) * matrix_stride.n, &matrix_shape_one, NULL, NULL, NULL);
        okk_bdc_sub(input_addr, input_addr, max_addr, &matrix_shape, NULL, NULL, &matrix_stride_zero);
#endif
        okk_parallel_start();
        // okk_bdc_exp(middle_addr, input_addr, work_addr, &matrix_shape);
        okk_bdc_exp_tunable(middle_addr, input_addr, work_addr, &matrix_shape, 5);
        if(b>0)
            okk_gdma_32bit_matrix_L2S(param->output_addr + (b-1) * batch_step, output_addr, rows, cols, cols_per_channel, row_stride);
        okk_parallel_end();

        okk_bdc_add(sum_addr, middle_addr, middle_addr + sizeof(float) * matrix_stride.n, &matrix_shape_one, NULL, NULL, NULL);

        okk_parallel_start();
        okk_bdc_div(output_addr, middle_addr, sum_addr, &matrix_shape, NULL, NULL, &matrix_stride_zero);
        if(b<3)
            okk_gdma_32bit_matrix_S2L(input_addr, param->input_addr + (b+1) * batch_step, rows, cols, cols_per_channel, row_stride);
        okk_parallel_end();
    }
    okk_gdma_32bit_matrix_L2S(param->output_addr + 3 * batch_step, output_addr, rows, cols, cols_per_channel, row_stride);
}

static void softmax_3(param_t * param)
{
    int batch_size = 79;
    int total_num = 4090;
    int channel = 64;
    int cols_per_channel = 64;

    dim4 matrix_shape = {batch_size, channel, 1, cols_per_channel};
    dim4 matrix_shape_one = {1, channel, 1, cols_per_channel};
    dim4 matrix_stride;
    okk_128_byte_aligned_stride_for_32bit(&matrix_stride, 0, &matrix_shape);

    local_addr_t input_addr = 0;
    okk_gdma_32bit_matrix_S2L(input_addr, param->input_addr, batch_size, total_num, cols_per_channel, total_num);

    local_addr_t output_addr = input_addr + batch_size * matrix_stride.n * sizeof(float);
    local_addr_t work_addr = output_addr + batch_size * matrix_stride.n * sizeof(float);
    local_addr_t sum_addr = work_addr + batch_size * matrix_stride.n * sizeof(float);
    system_addr_t dtcm_addr = okk_dtcm_start_addr();
    float * dtcm_fp32 = (float *)okk_dtcm_addr(dtcm_addr);

    // okk_bdc_exp(output_addr, input_addr, work_addr, &matrix_shape);
    okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &matrix_shape, 4);

    int pad_size = channel * cols_per_channel - total_num;
    if( pad_size )  // pad_size < cols_per_channel
    {
        x32 zero;
        zero.fp32 = 0;
        dim4 shape_t = {batch_size, 1, 1, pad_size};
        dim4 stride_t = {matrix_stride.n, 0, 0, 1};
        okk_gdma_32bit_set_C_local(output_addr + LOCAL_MEM_SIZE * (channel - 1) + sizeof(float) * (total_num%cols_per_channel), zero, &shape_t, &stride_t);
    }

    {
        int tmp = cols_per_channel / 2;
        dim4 shape_t = {batch_size, channel, 1, tmp};
#if 0
        dim4 stride_t = {matrix_stride.n, 0, 0, 1};
        okk_bdc_add(work_addr, output_addr, output_addr + tmp * sizeof(float), &shape_t, &stride_t, &stride_t, &stride_t);
#else
        okk_bdc_add(work_addr, output_addr, output_addr + tmp * sizeof(float), &shape_t, NULL, NULL, NULL);
#endif
    }
    for(int tmp = cols_per_channel / 4; tmp > 1; tmp /= 2 )
    {
        dim4 shape_t = {batch_size, channel, 1, tmp};
#if 0
        dim4 stride_t = {matrix_stride.n, 0, 0, 1};
        okk_bdc_add(work_addr, work_addr, work_addr + tmp * sizeof(float), &shape_t, &stride_t, &stride_t, &stride_t);
#else
        okk_bdc_add(work_addr, work_addr, work_addr + tmp * sizeof(float), &shape_t, NULL, NULL, NULL);
#endif
    }
#if 1
    {
        // batch_size, 64, 1, 1
        int tmp = 1;
        dim4 shape_t = {batch_size, channel, 1, tmp};
        okk_bdc_add(sum_addr, work_addr, work_addr + tmp * sizeof(float), &shape_t, NULL, NULL, NULL);
    }

    {
        x32 one;
        one.fp32 = 1;

        dim4 shape_t1 = {64, 1, 1, 1};
        okk_gdma_32bit_set_C_local(work_addr, one, &shape_t1, NULL);
        okk_bdc_matmul(input_addr, sum_addr, work_addr, 0, batch_size, 64, 1, 1, 1, false, false);

        dim4 shape_t2 = {1, DIV_UP(total_num, cols_per_channel), 1, cols_per_channel};
        okk_gdma_32bit_set_C_local(work_addr, one, &shape_t2, NULL);
        okk_bdc_matmul(sum_addr, input_addr, work_addr, 0, batch_size, 1, total_num, 1, cols_per_channel, false, false);

        okk_bdc_div(output_addr, output_addr, sum_addr, &matrix_shape, NULL, NULL, NULL);

        okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, batch_size, total_num, cols_per_channel, total_num);
    }
#else
    {
        // batch_size, 64, 1, 1
        int tmp = 1;
        dim4 shape_t = {batch_size, channel, 1, tmp};
#if 0
        dim4 stride_t = {matrix_stride.n, 0, 0, 1};
        okk_bdc_add(work_addr, work_addr, work_addr + tmp * sizeof(float), &shape_t, &stride_t, &stride_t, &stride_t);
        okk_gdma_32bit_cpy_L2S(dtcm_addr, work_addr, &shape_t, NULL, &stride_t);
#else
        okk_bdc_add(work_addr, work_addr, work_addr + tmp * sizeof(float), &shape_t, NULL, NULL, NULL);
        okk_gdma_32bit_cpy_L2S(dtcm_addr, work_addr, &shape_t, NULL, NULL);
#endif
        okk_poll();
    }

    {
        int i = 0;
        float sum_fp32 = 0;
        float * p = dtcm_fp32 + channel * i;
        for(int j=0; j<channel; j++)
        {
            sum_fp32 += p[j];
        }
        local_addr_t cur_addr = output_addr + matrix_stride.n * i * sizeof(float);
        okk_bdc_div_C(cur_addr, cur_addr, sum_fp32, &matrix_shape_one, NULL, NULL);
    }
    for(int i=1; i<batch_size; i++)
    {
        float sum_fp32 = 0;
        float * p = dtcm_fp32 + channel * i;
        for(int j=0; j<channel; j++)
        {
            sum_fp32 += p[j];
        }
        local_addr_t cur_addr = output_addr + matrix_stride.n * i * sizeof(float);
        okk_parallel_start();
        okk_bdc_div_C(cur_addr, cur_addr, sum_fp32, &matrix_shape_one, NULL, NULL);

        local_addr_t prev_addr = output_addr + matrix_stride.n * (i - 1) * sizeof(float);
        okk_gdma_32bit_matrix_L2S(param->output_addr + (i - 1) * total_num * sizeof(float), prev_addr, 1, total_num, cols_per_channel, total_num);
        okk_parallel_end();
    }
    {
        int i = batch_size;
        local_addr_t prev_addr = output_addr + matrix_stride.n * (i - 1) * sizeof(float);
        okk_gdma_32bit_matrix_L2S(param->output_addr + (i - 1) * total_num * sizeof(float), prev_addr, 1, total_num, cols_per_channel, total_num);
    }
#endif
}

static void softmax_4(param_t * param)
{
    local_addr_t input_addr = 0;
    int rows = 2;
    int cols = (6132 / 2) * 21;
    int cols_per_channel = 48 * 21;

    okk_gdma_32bit_matrix_S2L(input_addr, param->input_addr, rows, cols, cols_per_channel, cols);

    // n = 2, c = 64, h = 48, w = 21.
    dim4 shape_all = {2, 64, 48, 21};

    dim4 stride_all;
    okk_128_byte_aligned_stride_for_32bit(&stride_all, 0, &shape_all);

    int data_size = sizeof(float) * stride_all.n * shape_all.n;

    local_addr_t output_addr = input_addr + data_size;
    local_addr_t work_addr = output_addr + data_size;
    local_addr_t sum_addr = work_addr + data_size;

    dim4 shape_7 = {shape_all.n, shape_all.c, shape_all.h, 7};
    dim4 shape_1 = {shape_all.n, shape_all.c, shape_all.h, 1};
    dim4 shape_2 = {shape_all.n, shape_all.c, shape_all.h, 2};
    dim4 shape_3 = {shape_all.n, shape_all.c, shape_all.h, 3};

    dim4 stride_7;
    okk_128_byte_aligned_stride_for_32bit(&stride_7, 0, &shape_7);
    dim4 stride_zero = {stride_7.n, stride_7.c, stride_7.h, 0};

    // okk_bdc_exp(output_addr, input_addr, work_addr, &shape_all);
    okk_bdc_exp_tunable(output_addr, input_addr, work_addr, &shape_all, 5);

    okk_bdc_add(sum_addr, output_addr, output_addr + 7 * sizeof(float), &shape_7, &stride_7, &stride_all, &stride_all);
    okk_bdc_add(sum_addr, sum_addr, output_addr + 14 * sizeof(float), &shape_7, &stride_7, &stride_7, &stride_all);

    okk_bdc_add(sum_addr + sizeof(float), sum_addr + sizeof(float), sum_addr + 4 * sizeof(float), &shape_3, &stride_7, &stride_7, &stride_7);
    okk_bdc_add(sum_addr, sum_addr, sum_addr + 2 * sizeof(float), &shape_2, &stride_7, &stride_7, &stride_7);
    okk_bdc_add(sum_addr, sum_addr, sum_addr + 1 * sizeof(float), &shape_1, &stride_7, &stride_7, &stride_7);

    okk_bdc_div(output_addr, output_addr, sum_addr, &shape_all, &stride_all, &stride_all, &stride_zero);
    okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, rows, cols, cols_per_channel, cols);
}

void softmax_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    (void)(param);

    int N = param->N;
    int C = param->C;
    int H = param->H;
    int W = param->W;

    if( C == 1 )
    {
        // nothing.
    }
    else if( N == 1 && C == 370 && H == 13 && W == 13 )
    {
        softmax_0(param);
    }
    else if( N == 1 && C == 1000 && H == 1 && W == 1 )
    {
        softmax_1(param);
    }
    else if( N == 4 && C == 2 && H == 157 && W == 283 )
    {
        softmax_2(param);
    }
    else if( N == 79 && C == 4090 && H == 1 && W == 1 )
    {
        softmax_3(param);
    }
    else if( N == 6132 && C == 21 && H == 1 && W == 1 )
    {
        softmax_4(param);
    }
    else
    {
        softmax_x(param);
    }
    // TODO
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);

