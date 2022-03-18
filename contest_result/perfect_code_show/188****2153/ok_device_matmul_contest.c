//
// wp
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
    int left_rows, left_cols, right_cols;
    unsigned long long output_addr;
    unsigned long long left_addr;
    unsigned long long right_addr;
} __attribute__((packed)) param_t;

/*
 * param->left_cols <= 2048 && param->right_cols <= 2048
 */
void matmul_soso(param_t* param){

    const unsigned int left_rows_kernel_max = 256;

    const unsigned int left_rows_kernel_num = DIV_UP(param->left_rows, left_rows_kernel_max);
    const unsigned int left_rows_kernel_overage = param->left_rows - (left_rows_kernel_num-1)*left_rows_kernel_max;
    const unsigned int ddr_left_kernel_bytes = left_rows_kernel_max * param->left_cols * sizeof(float);
    const unsigned int ddr_output_kernel_bytes = left_rows_kernel_max * param->right_cols * sizeof(float);

    int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
    if (left_cols_per_channel > 128) left_cols_per_channel = 128; // ggg v1
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);
    if (right_cols_per_channel > 128) right_cols_per_channel = 128; // ggg v1

    okk_gdma_32bit_matrix_S2L(
            0,
            param->right_addr,
            param->left_cols,
            param->right_cols,
            right_cols_per_channel,
            param->right_cols);

    const dim4 left_shape = {.n = left_rows_kernel_max, .c = DIV_UP(param->left_cols, left_cols_per_channel),
            .h = 1, .w = left_cols_per_channel };

    const dim4 right_shape = {.n = param->left_cols, .c = DIV_UP(param->right_cols, right_cols_per_channel),
            .h = 1, .w = right_cols_per_channel};

    const dim4 output_shape = {.n = left_rows_kernel_max, .c = DIV_UP(param->right_cols, right_cols_per_channel),
            .h = 1, .w = right_cols_per_channel};

    dim4 left_stride, right_stride, output_stride;

    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    local_addr_t left_addr[2];
    local_addr_t output_addr[2];

    const int local_left_bytes = left_stride.n * left_shape.n * sizeof(float);
    const int local_output_bytes = output_stride.n * output_shape.n * sizeof(float);

    left_addr[0] = right_stride.n * right_shape.n * sizeof(float);
    left_addr[1] = left_addr[0] + local_left_bytes;

    output_addr[0] = left_addr[1] + local_left_bytes;
    output_addr[1] = output_addr[0] + local_output_bytes;

    if(output_addr[1] + local_output_bytes > LOCAL_MEM_SIZE){
        return;
    }
    if(output_addr[1] + local_output_bytes > LOCAL_MEM_SIZE){
        return;
    }

    const unsigned int S = left_rows_kernel_num;
    unsigned int flag = 0; // 0, 1
    for (unsigned int i = 0; i < S + 2; ++i) {
        okk_parallel_start();
        if (i < S)
            okk_gdma_32bit_matrix_S2L(
                    left_addr[flag],
                    param->left_addr + i * ddr_left_kernel_bytes,
                    i == S - 1 ? left_rows_kernel_overage : left_rows_kernel_max,
                    param->left_cols,
                    left_cols_per_channel,
                    param->left_cols);

        if (i > 0 && i < S + 1)
            okk_bdc_matmul(
                    output_addr[!flag],
                    left_addr[!flag],
                    0,
                    NO_USE,
                    i == S ? left_rows_kernel_overage : left_rows_kernel_max,
                    param->left_cols,
                    param->right_cols,
                    left_cols_per_channel,
                    right_cols_per_channel,
                    false,
                    false);

        if (i > 1)
            okk_gdma_32bit_matrix_L2S(
                    param->output_addr + (i - 2) * ddr_output_kernel_bytes,
                    output_addr[flag],
                    i == S + 1 ? left_rows_kernel_overage : left_rows_kernel_max,
                    param->right_cols,
                    right_cols_per_channel,
                    param->right_cols);
        flag = !flag;
        okk_parallel_end();
    }
}

/*
 * split left_col and right_col
 * left_row < sth
 * left_col direction
 */
void matmul_fight(param_t *param){
    int tmp = 1024;
    if(param->left_cols > 3072){
        tmp = 1;
    }

    const int left_col_max = (param->left_cols > tmp) ? tmp : param->left_cols;
    const int right_col_max = (param->right_cols > 2048) ? 2048 : param->right_cols;

    const int left_col_num = DIV_UP(param->left_cols, left_col_max);
    const int right_col_num = DIV_UP(param->right_cols, right_col_max);

    const int left_col_overage = param->left_cols - (left_col_num-1)*left_col_max;
    const int right_col_overage = param->right_cols - (right_col_num-1)*right_col_max;

    const int left_cols_per_channel = DIV_UP(left_col_max, NPU_NUM);
    const int right_cols_per_channel = DIV_UP(right_col_max, NPU_NUM);

    if (left_cols_per_channel > 128) return; // ggg v1
    if (right_cols_per_channel > 128) return; // ggg v1

    const dim4 left_shape = {.n = param->left_rows, .c = DIV_UP(left_col_max, left_cols_per_channel),
            .h = 1, .w = left_cols_per_channel };

    const dim4 right_shape = {.n = left_col_max, .c = DIV_UP(right_col_max, right_cols_per_channel),
            .h = 1, .w = right_cols_per_channel};

    const dim4 output_shape = {.n = param->left_rows, .c = DIV_UP(right_col_max, right_cols_per_channel),
            .h = 1, .w = right_cols_per_channel};

    dim4 left_stride, right_stride, output_stride;

    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    local_addr_t left_addr[2];
    local_addr_t right_addr[2];
    local_addr_t output_addr[2];

    const int local_left_bytes = left_shape.n*left_stride.n* sizeof(float);
    const int local_right_bytes = right_shape.n*right_stride.n* sizeof(float);
    const int local_output_bytes = output_shape.n*output_stride.n* sizeof(float);

    left_addr[0] = 0;
    left_addr[1] = local_left_bytes;

    right_addr[0] = left_addr[1] + local_left_bytes;
    right_addr[1] = right_addr[0] + local_right_bytes;

    output_addr[0] = right_addr[1] + local_right_bytes;
    output_addr[1] = output_addr[0] + local_output_bytes;

    if(output_addr[1] + right_col_num*local_output_bytes > LOCAL_MEM_SIZE) return;

    const int S = left_col_num * right_col_num;

    int left_col_id = 0;
    int right_col_id = 0;

    int left_col_id_1=0, left_col_id_2=0;
    int right_col_id_1=0, right_col_id_2=0;

    unsigned int flag = 0; // 0, 1
    for (int i = 0; i < S+2; ++i) {
        okk_parallel_start();
        if(i < S){
            okk_gdma_32bit_matrix_S2L(
                    left_addr[flag],
                    param->left_addr + left_col_id * left_col_max * sizeof(float),
                    param->left_rows,
                    (left_col_id == left_col_num-1) ? left_col_overage : left_col_max,
                    (left_col_id == left_col_num-1) ? DIV_UP(left_col_overage, NPU_NUM) : left_cols_per_channel,
                    param->left_cols);

            okk_gdma_32bit_matrix_S2L(
                    right_addr[flag],
                    param->right_addr + left_col_id * left_col_max * param->right_cols * sizeof(float) + right_col_id * right_col_max * sizeof(float),
                    left_col_id == (left_col_num-1) ? left_col_overage : left_col_max,
                    right_col_id == (right_col_num - 1) ? right_col_overage : right_col_max,
                    right_col_id == (right_col_num - 1) ? DIV_UP(right_col_overage, NPU_NUM) : right_cols_per_channel,
                    param->right_cols);
        }

        if(i > 0 && i < S+1){
            okk_bdc_matmul(
                    output_addr[right_col_id_1%2],
                    left_addr[!flag],
                    right_addr[!flag],
                    NO_USE,
                    param->left_rows,
                    left_col_id_1 == left_col_num-1 ? left_col_overage : left_col_max,
                    right_col_id_1 == right_col_num - 1 ? right_col_overage : right_col_max,
                    left_col_id_1 == left_col_num-1 ? DIV_UP(left_col_overage, NPU_NUM) : left_cols_per_channel,
                    right_col_id_1 == right_col_num - 1 ? DIV_UP(right_col_overage, NPU_NUM) : right_cols_per_channel,
                    false,
                    left_col_id_1 == 0 ? false : true);
        }

        if(i > 1 && (left_col_id_2 == left_col_num-1)) {
            okk_gdma_32bit_matrix_L2S(
                    param->output_addr + right_col_id_2 * right_col_max* sizeof(float),
                    output_addr[right_col_id_2%2],
                    param->left_rows,
                    right_col_id_2 == right_col_num - 1 ? right_col_overage : right_col_max,
                    right_col_id_2 == right_col_num - 1 ? DIV_UP(right_col_overage, NPU_NUM) : right_cols_per_channel,
                    param->right_cols);
        }
        flag = !flag;

        left_col_id_2 = left_col_id_1;
        left_col_id_1 = left_col_id;

        right_col_id_2 = right_col_id_1;
        right_col_id_1 = right_col_id;

        left_col_id++;

        if(left_col_id == left_col_num){
            right_col_id++;
            left_col_id = 0;
        }

        okk_parallel_end();
    }

}


void my_matmul_demo(param_t *param) {
    dim4 output_stride, left_stride, right_stride;
    int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
    if (left_cols_per_channel > 128)
        left_cols_per_channel = 128;
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);
    if (right_cols_per_channel > 128)
        right_cols_per_channel = 128;
    // Local left matrix tensor.
    local_addr_t left_addr = 0;
    dim4 left_shape = {
            .n = param->left_rows, .c = DIV_UP(param->left_cols, left_cols_per_channel),
            .h = 1, .w = left_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    local_addr_t right_addr = left_addr + left_stride.n * left_shape.n * sizeof(float);
    dim4 right_shape = {
            .n = param->left_cols, .c = DIV_UP(param->right_cols, right_cols_per_channel),
            .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    local_addr_t output_addr = right_addr + right_stride.n * right_shape.n * sizeof(float);
    dim4 output_shape = {
            .n = param->left_rows, .c = DIV_UP(param->right_cols, right_cols_per_channel),
            .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // Safe checking.
    if(output_addr + output_stride.n * output_shape.n * sizeof(float) > LOCAL_MEM_SIZE){
        return;
    }
    OKKERNEL_ASSERT(output_addr + output_stride.n * output_shape.n * sizeof(float) <= LOCAL_MEM_SIZE);
    // Copy global left matrix tensor to local left matrix tensor.
    okk_gdma_32bit_matrix_S2L(
            left_addr,
            param->left_addr,
            param->left_rows,
            param->left_cols,
            left_cols_per_channel,
            param->left_cols);
    // Copy global right matrix tensor to local right matrix tensor.
    okk_gdma_32bit_matrix_S2L(
            right_addr,
            param->right_addr,
            param->left_cols,
            param->right_cols,
            right_cols_per_channel,
            param->right_cols);
    // Matrix multiplication.
    okk_bdc_matmul(
            output_addr,
            left_addr,
            right_addr,
            NO_USE,
            param->left_rows,
            param->left_cols,
            param->right_cols,
            left_cols_per_channel,
            right_cols_per_channel,
            false,
            false);
    // Copy local output matrix tensor to global output matrix tensor.
    okk_gdma_32bit_matrix_L2S(
            param->output_addr,
            output_addr,
            param->left_rows,
            param->right_cols,
            right_cols_per_channel,
            param->right_cols);
}

void matmul_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    if(param->left_rows==300 && param->left_cols == 2048 && param->right_cols == 80){
        //case10
        matmul_fight(param);
    } else if(param->left_cols <= 2048 && param->right_cols <= 2048){
        matmul_soso(param);
    } else{
        matmul_fight(param);
    }
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(matmul_contest);
