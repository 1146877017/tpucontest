#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define ALIGN(x, n) (((x) + (n) - 1) / (n) * (n))
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
// #define LOCAL_MEM_SIZE 800
#define NPU_NUM okk_npu_num()
#define NO_USE 0
typedef struct {
    int left_rows, left_cols, right_cols;
    unsigned long long output_addr;
    unsigned long long left_addr;
    unsigned long long right_addr;
} __attribute__((packed)) param_t;

void matmul_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    if(param->left_cols != 4 && param->left_cols !=2 && !(param->left_cols==1024 && (param->right_cols == 1 || param->right_cols == 1024)) && param->left_cols < 9216)
    {
        int max_left_row;
        int max_left_col;
        int max_right_col;
        switch (param->left_cols)
        {
        case 100352:
            // case 0;
            max_left_row = 2;
            max_left_col = 1;
            max_right_col = 2048;
            break;
        case 1280:
            // case 1
            max_left_row = 2;
            max_left_col = 640;
            max_right_col = 1000;
            break;
        case 25088:
            // case 2;
            max_left_row = 2;
            max_left_col = 1;
            max_right_col = 4096;
            break;
        case 1024:
            // case 3
            max_left_row = 4;
            max_left_col = 512;
            max_right_col = 4096;
            break;
        case 2048:
            if(param->left_rows == 32)
            {
                // case 4
                max_left_row = 32;
                max_left_col = 512;
                max_right_col = 36;
            }else
            {
                // casr 10
                max_left_row = 300;
                max_left_col = 1024;
                max_right_col = 80;
            }
            break;
        case 9216:
            // case 5
            max_left_row = 64;
            max_left_col = 1;
            max_right_col = 4096;
            break;
        case 256:
            // case 6
            max_left_row = 79;
            max_left_col = 128;
            max_right_col = 4096;
            break;
        case 4096:
            // case 7
            max_left_row = 200;
            max_left_col = 256;
            max_right_col = 324;
            break;
        case 768:
            // case 8:
            max_left_row = 256;
            max_left_col = 256;
            max_right_col = 2048;
            break;
        case 3072:
            // case 9
            max_left_row = 256;
            max_left_col = 256;
            max_right_col = 768;
            break;
        case 4:
            // case 11 cha cha cha
            max_left_row = 512;
            max_left_col = 1024;
            max_right_col = 1024;
            break;
        case 2:
            // case 12 cha cha cha
            max_left_row = 2048;
            max_left_col = 4;
            max_right_col = 1024;
            break;
        default:
            max_left_row = MIN(8, param->left_rows);
            max_left_col = MIN(1024, param->left_cols);
            max_right_col = MIN(1024, param->right_cols);
            break;
        }

        int left_cols_per_channel = DIV_UP(max_left_col, NPU_NUM);
        int right_cols_per_channel = DIV_UP(max_right_col, NPU_NUM);
        dim4 left_shape = {.n=max_left_row, .c=DIV_UP(max_left_col, left_cols_per_channel), .h=1, .w=left_cols_per_channel};
        dim4 right_shape = {.n=max_left_col, .c=DIV_UP(max_right_col, right_cols_per_channel), .h=1, .w=right_cols_per_channel};
        dim4 output_shape = {.n=max_left_row, .c=right_shape.c, .h=1, .w=right_shape.w};

        dim4 left_stride, right_stride, output_stride;
        okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
        okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
        okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

        local_addr_t left_addr[2], right_addr[2], output_addr[2];
        output_addr[0] = 0;
        left_addr[0] = output_addr[0] + output_shape.n * output_stride.n * sizeof(float);
        right_addr[0] = left_addr[0] + left_shape.n * left_stride.n * sizeof(float);

        output_addr[1] = right_addr[0] + right_shape.n * right_stride.n * sizeof(float);
        left_addr[1] = output_addr[1] + output_shape.n * output_stride.n * sizeof(float);
        right_addr[1] = left_addr[1] + left_shape.n * left_stride.n * sizeof(float);

        int left_row_iteration = DIV_UP(param->left_rows, max_left_row);
        int left_col_iteration = DIV_UP(param->left_cols, max_left_col);
        int right_col_iteration = DIV_UP(param->right_cols, max_right_col);

        int cnt=0;

        int last_left_row = param->left_rows - (left_row_iteration - 1) * max_left_row;
        int last_left_col = param->left_cols - (left_col_iteration - 1) * max_left_col;
        int last_right_col = param->right_cols - (right_col_iteration - 1) * max_right_col;

        int last_left_col_per_channel = DIV_UP(last_left_col, NPU_NUM);
        int last_right_col_per_channel = DIV_UP(last_right_col, NPU_NUM);

        for(int i=0; i<left_row_iteration; i++)
        {
            for(int j=0; j<right_col_iteration;j++)
            {
                x32 C = {0.0};
                okk_bdc_32bit_set_C(output_addr[j%2], C, &output_shape, NO_USE);
                if(j!=0)
                    okk_parallel_end();
                for(int k=0; k<left_col_iteration + 1; k++)
                {
                    // if(k==0)
                    okk_parallel_start();
                    if(k<left_col_iteration)
                    {
                        okk_gdma_32bit_matrix_S2L(left_addr[k%2], param->left_addr + i*max_left_row * param->left_cols * sizeof(float) + k*max_left_col*sizeof(float), (i==left_row_iteration-1)?last_left_row:max_left_row, (k==left_col_iteration-1)?last_left_col:max_left_col, (k==left_col_iteration - 1)?last_left_col_per_channel:left_cols_per_channel, param->left_cols);

                        okk_gdma_32bit_matrix_S2L(right_addr[k%2], param->right_addr + k*max_left_col*param->right_cols*sizeof(float) + j*max_right_col * sizeof(float), (k==left_col_iteration-1)?last_left_col:max_left_col, (j==right_col_iteration - 1)?last_right_col:max_right_col, (j==right_col_iteration - 1)?last_right_col_per_channel:right_cols_per_channel, param->right_cols);
                    }

                    if(k>0)
                        okk_bdc_matmul(output_addr[j%2], left_addr[(k-1)%2], right_addr[(k-1)%2], NO_USE, (i==left_row_iteration-1)?last_left_row:max_left_row, (k==left_col_iteration)?last_left_col:max_left_col, (j==right_col_iteration - 1)?last_right_col:max_right_col, (k==left_col_iteration)?last_left_col_per_channel:left_cols_per_channel, (j==right_col_iteration - 1)?last_right_col_per_channel:right_cols_per_channel, false, true);

                    okk_parallel_end();
                }
                okk_parallel_start();
                okk_gdma_32bit_matrix_L2S(param->output_addr + i*max_left_row*param->right_cols * sizeof(float) + j * max_right_col * sizeof(float), output_addr[j%2], (i==left_row_iteration-1)?last_left_row:max_left_row, (j==right_col_iteration - 1)?last_right_col:max_right_col, (j==right_col_iteration - 1)?last_right_col_per_channel:right_cols_per_channel, param->right_cols);
            }
            okk_parallel_end();
        }
        okk_poll();
        return;
    }

    int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);
    
    dim4 left_shape = {.n=param->left_rows, .c=DIV_UP(param->left_cols, left_cols_per_channel), .h=1, .w=left_cols_per_channel};
    dim4 right_shape = {.n=param->left_cols, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};
    dim4 output_shape = {.n=param->left_rows, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};

    dim4 left_stride, right_stride, output_stride;

    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    // int left_size = left_shape.n * left_stride.n * sizeof(float);
    int right_size = right_shape.n * right_stride.n * sizeof(float);

    int max_left_row = 0;
    if ((long int)LOCAL_MEM_SIZE - right_size < 0)
        max_left_row = -1;
    else
        max_left_row = ((long int)LOCAL_MEM_SIZE - right_size) / ((left_stride.n + output_stride.n) * (int)sizeof(float));

    max_left_row = MIN(max_left_row, param->left_rows);

    // OKKERNEL_LOG("max_left_row:%d\n", max_left_row);

    // okk_poll();
    // return;

    if(max_left_row > 1)
    {
        // OKKERNEL_LOG("here\n");
        max_left_row /= 2;

        int attempt_total_size = (max_left_row * left_stride.n * 2 + right_shape.n * right_stride.n + output_shape.n * output_stride.n) * sizeof(float);

        if(attempt_total_size <= LOCAL_MEM_SIZE)
        {
            // OKKERNEL_LOG("here\n");
            // int output_size = output_shape.n * output_stride.n * sizeof(float);
            // max_left_row = ((long int)LOCAL_MEM_SIZE - right_size - output_size) / (left_stride.n * (int)sizeof(float));

            // OKKERNEL_LOG("max left row:%d\n", max_left_row);

            local_addr_t left_addr[2], right_addr, output_addr;
            left_addr[0] = 0;
            left_addr[1] = left_addr[0] + max_left_row * left_stride.n * sizeof(float);

            right_addr = ALIGN(left_addr[1] + max_left_row * left_stride.n * sizeof(float), LOCAL_MEM_SIZE / okk_local_mem_bank_per_npu());
            output_addr = right_addr + right_shape.n * right_stride.n * sizeof(float);

            int iteration = DIV_UP(param->left_rows, max_left_row);

            // OKKERNEL_LOG("iteration:%d\n", iteration);

            unsigned int left_tensor_size_global = max_left_row * param->left_cols * sizeof(float);
            unsigned int output_tensor_size_local = max_left_row * output_stride.n * sizeof(float);

            int last_left_row = param->left_rows - (iteration - 1) * max_left_row;

            for(int i = 0; i<iteration + 1; i++)
            {
                okk_parallel_start();

                if(i == 0)
                    okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, param->left_cols, param->right_cols, right_cols_per_channel, param->right_cols);

                if(i<iteration)
                    okk_gdma_32bit_matrix_S2L(left_addr[i%2], param->left_addr + i * left_tensor_size_global, i==iteration-1?last_left_row:max_left_row, param->left_cols, left_cols_per_channel, param->left_cols);
                
                if(i>0 && i<iteration + 1)
                    okk_bdc_matmul(output_addr + (i-1) * output_tensor_size_local, left_addr[(i-1)%2], right_addr, NO_USE, i==iteration?last_left_row:max_left_row, param->left_cols, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

                okk_parallel_end();
            }

            okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows, param->right_cols, right_cols_per_channel, param->right_cols);
        }else
        {
            local_addr_t left_addr[2], right_addr, output_addr[2];
            left_addr[0] = 0;
            output_addr[0] = left_addr[0] + max_left_row * left_stride.n * sizeof(float);

            left_addr[1] = output_addr[0] + max_left_row * right_stride.n * sizeof(float);
            output_addr[1] = left_addr[1] + max_left_row * left_stride.n * sizeof(float);

            right_addr = output_addr[1] + max_left_row * right_stride.n * sizeof(float);

            okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr, param->left_cols, param->right_cols, right_cols_per_channel, param->right_cols);

            int iteration = DIV_UP(param->left_rows, max_left_row);

            unsigned int left_tensor_size_global = max_left_row * param->left_cols * sizeof(float);
            unsigned int output_tensor_size_global = max_left_row * param->right_cols * sizeof(float);

            int last_left_row = param->left_rows - (iteration - 1) * max_left_row;

            for(int i = 0; i<iteration + 2; i++){
                okk_parallel_start();

                if(i < iteration)
                    okk_gdma_32bit_matrix_S2L(left_addr[i%2], param->left_addr + i * left_tensor_size_global, i==iteration-1?last_left_row:max_left_row, param->left_cols, left_cols_per_channel, param->left_cols);
                
                if(i>0 && i<iteration + 1)
                    okk_bdc_matmul(output_addr[(i-1)%2], left_addr[(i-1)%2], right_addr, NO_USE, i==iteration?last_left_row:max_left_row, param->left_cols, param->right_cols, left_cols_per_channel, right_cols_per_channel, false, false);

                if(i>1)
                    okk_gdma_32bit_matrix_L2S(param->output_addr + (i-2)*output_tensor_size_global, output_addr[i%2], i==iteration+1?last_left_row:max_left_row, param->right_cols, right_cols_per_channel, param->right_cols);
                
                okk_parallel_end();
            }
        }
    }else{
        // OKKERNEL_LOG("not here\n");
        // okk_poll();
        // return;
        // may be left row not be largest
        if(param->left_cols > param->right_cols)
        {
            // left cols largest
            // OKKERNEL_LOG("here\n");
            int left_col = MIN(1024, param->left_cols);
            if(param->left_cols == 100352 || param->left_cols == 25088 || param->left_cols == 9216 || param->left_cols == 4096)
                left_col = 1;
            if(param->left_cols == 3072)
                left_col = 256;
            int output_size = output_shape.n * output_stride.n * sizeof(float);

            left_cols_per_channel = DIV_UP(left_col, NPU_NUM);

            dim4 new_left_shape = {.n=param->left_rows, .c=DIV_UP(left_col, left_cols_per_channel), .h=1, .w=left_cols_per_channel};
            dim4 new_right_shape = {.n=left_col, .c=DIV_UP(param->right_cols, right_cols_per_channel), .h=1, .w=right_cols_per_channel};

            dim4 temp_left_stride;

            okk_128_byte_aligned_stride_for_32bit(&temp_left_stride, 0, &new_left_shape);

            local_addr_t left_addr, right_addr, output_addr;
            left_addr = 0;
            right_addr = left_addr + new_left_shape.n * temp_left_stride.n * sizeof(float);
            output_addr = right_addr + new_right_shape.n * right_stride.n * sizeof(float);

            int iteration = DIV_UP(param->left_cols, left_col);

            int last_left_col = param->left_cols - (iteration - 1)*left_col;
            int last_left_col_per_channel = DIV_UP(last_left_col, NPU_NUM);

            int left_skip_size = left_col * sizeof(float);
            int right_skip_size = left_col * param->right_cols * sizeof(float);

            x32 C = {0.0};
            okk_bdc_32bit_set_C(output_addr, C, &output_shape, &output_stride);

            for(int i = 0; i<iteration; i++)
            {
                okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr + i*left_skip_size, param->left_rows, i==iteration-1?last_left_col:left_col, i==iteration-1?last_left_col_per_channel:left_cols_per_channel, param->left_cols);

                okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + i*right_skip_size, i==iteration-1?last_left_col:left_col, param->right_cols, right_cols_per_channel, param->right_cols);

                okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, param->left_rows, i==iteration-1?last_left_col:left_col, param->right_cols, i==iteration-1?last_left_col_per_channel:left_cols_per_channel, right_cols_per_channel, false, true);
            }

            okk_gdma_32bit_matrix_L2S(param->output_addr, output_addr, param->left_rows, param->right_cols, right_cols_per_channel, param->right_cols);
        }else
        {
            // OKKERNEL_LOG("here\n");
            // right cols largest
            int right_col = 4096;

            int iteration = DIV_UP(param->right_cols, right_col);
            
            right_cols_per_channel = DIV_UP(right_col, NPU_NUM);
            dim4 new_right_shape = {.n=param->left_cols, .c=DIV_UP(right_col, right_cols_per_channel), .h=1, .w=right_cols_per_channel};
            dim4 new_output_shape = {.n=param->left_rows, .c=new_right_shape.c, .h=1, .w=new_right_shape.w};

            dim4 new_right_stride, new_output_stride;
            okk_128_byte_aligned_stride_for_32bit(&new_right_stride, 0, &new_right_shape);
            okk_128_byte_aligned_stride_for_32bit(&new_output_stride, 0, &new_output_shape);

            local_addr_t left_addr, right_addr, output_addr;
            left_addr = 0;
            right_addr = left_addr + left_shape.n * left_stride.n * sizeof(float);
            output_addr = right_addr + new_right_shape.n * new_right_stride.n * sizeof(float);

            int right_skip_global = right_col * sizeof(float);
            int last_right_col = param->right_cols - (iteration - 1) * right_col;
            int last_right_cols_per_channel = DIV_UP(last_right_col, NPU_NUM);

            okk_gdma_32bit_matrix_S2L(left_addr, param->left_addr, param->left_rows, param->left_cols, left_cols_per_channel, param->left_cols);

            for(int i = 0; i< iteration; i++)
            {
                okk_gdma_32bit_matrix_S2L(right_addr, param->right_addr + i * right_skip_global, param->left_cols, i==iteration-1?last_right_col:right_col, i==iteration-1?last_right_cols_per_channel:right_cols_per_channel, param->right_cols);

                okk_bdc_matmul(output_addr, left_addr, right_addr, NO_USE, param->left_rows, param->left_cols, i==iteration-1?last_right_col:right_col, left_cols_per_channel, i==iteration-1?last_right_cols_per_channel:right_cols_per_channel, false, false);

                okk_gdma_32bit_matrix_L2S(param->output_addr + i * right_skip_global, output_addr, param->left_rows, i==iteration-1?last_right_col:right_col, i==iteration-1?last_right_cols_per_channel:right_cols_per_channel, param->right_cols);
            }
        }
    }

    okk_poll();
}
OKKERNEL_FUNC_REGISTER(matmul_contest);
