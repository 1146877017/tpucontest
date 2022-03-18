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

void matmul_tilingH(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 output_stride, left_stride, right_stride;
    bool small = param->left_rows < 256;
    int Sh = small ? DIV_UP(param->left_rows, 4) : 256;
    if (param->left_cols < 128 && param->left_rows > 10240) Sh = 896;
    int Th = param->left_rows / Sh;
    int Thh = Th - 1;
    int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);
    if (param->right_cols % 32 > 1) right_cols_per_channel  = 32; 
    // Local left matrix tensor.
    local_addr_t left_addr = 0;
    dim4 left_shape = {
        .n = Sh, .c = DIV_UP(param->left_cols, left_cols_per_channel),
        .h = 1, .w = left_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    local_addr_t left_addr2 = left_addr + left_stride.n * left_shape.n * sizeof(float);
    local_addr_t right_addr = left_addr2 + left_stride.n * left_shape.n * sizeof(float);
    dim4 right_shape = {
        .n = param->left_cols, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    local_addr_t output_addr = right_addr + right_stride.n * right_shape.n * sizeof(float);
    dim4 output_shape = {
        .n = Sh, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    local_addr_t output_addr2 = output_addr + output_stride.n * output_shape.n * sizeof(float);
    if (output_addr2 + output_stride.n * output_shape.n * sizeof(float) > LOCAL_MEM_SIZE) {
        return;
    }
    local_addr_t left_addrs[2] = { left_addr, left_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    okk_gdma_32bit_matrix_S2L(
        right_addr,
        param->right_addr,
        param->left_cols,
     	param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_gdma_32bit_matrix_S2L(
        left_addr,
        param->left_addr,
        Sh,
        param->left_cols,
        left_cols_per_channel,
        param->left_cols);
    okk_parallel_start();
    okk_gdma_32bit_matrix_S2L(
        left_addr2,
        param->left_addr + Sh * param->left_cols * sizeof(float),
        Sh,
        param->left_cols,
        left_cols_per_channel,
        param->left_cols);
    okk_bdc_matmul(
        output_addr,
        left_addr,
        right_addr,
        NO_USE,
        Sh,
        param->left_cols,
        param->right_cols,
        left_cols_per_channel,
        right_cols_per_channel,
        false,
        false);
    okk_parallel_end();
    int j = 1;
    for (; j < Thh; j++) {
    okk_parallel_start();
    okk_gdma_32bit_matrix_S2L(
        left_addrs[(j+1)%2],
        param->left_addr + (j+1) * Sh * param->left_cols * sizeof(float),
        Sh,
        param->left_cols,
        left_cols_per_channel,
        param->left_cols);
    okk_bdc_matmul(
        output_addrs[j%2],
        left_addrs[j%2],
        right_addr,
        NO_USE,
        Sh,
        param->left_cols,
        param->right_cols,
        left_cols_per_channel,
        right_cols_per_channel,
        false,
        false);
    okk_gdma_32bit_matrix_L2S(
        param->output_addr + ((j-1) * Sh * param->right_cols) * sizeof(float),
        output_addrs[(j-1)%2],
        Sh,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_matmul(
        output_addrs[j%2],
        left_addrs[j%2],
        right_addr,
        NO_USE,
        Sh,
        param->left_cols,
        param->right_cols,
        left_cols_per_channel,
        right_cols_per_channel,
        false,
        false);
    okk_gdma_32bit_matrix_L2S(
        param->output_addr + ((j-1) * Sh * param->right_cols) * sizeof(float),
        output_addrs[(j-1)%2],
        Sh,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_parallel_end();
    okk_gdma_32bit_matrix_L2S(
        param->output_addr + (j * Sh * param->right_cols) * sizeof(float),
        output_addrs[j%2],
        Sh,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_poll();
}


void matmul_tilingL(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 output_stride, left_stride, right_stride;

    int Sl = param->left_cols > 512 ? 512 : 128;
    int Tl = DIV_UP(param->left_cols, Sl);
    int Tll = Tl - 1;
    int left_cols_per_channel = 64;
    int right_cols_per_channel = param->right_cols > 2048 ? 64 : 16;

    // Local left matrix tensor.
    local_addr_t left_addr = 0;
    dim4 left_shape = {
        .n = param->left_rows, .c = DIV_UP(Sl, left_cols_per_channel),
        .h = 1, .w = left_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    local_addr_t left_addr2 = left_addr + left_stride.n * left_shape.n * sizeof(float);
    local_addr_t right_addr = left_addr2 + left_stride.n * left_shape.n * sizeof(float);
    dim4 right_shape = {
        .n = Sl, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    local_addr_t right_addr2 = right_addr + right_stride.n * right_shape.n * sizeof(float);
    local_addr_t output_addr = right_addr2 + right_stride.n * right_shape.n * sizeof(float);
    dim4 output_shape = {
        .n = param->left_rows, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    local_addr_t left_addrs[2] = {left_addr, left_addr2};
    local_addr_t right_addrs[2] = {right_addr, right_addr2};
    okk_gdma_32bit_matrix_S2L(
		    left_addr,
		    param->left_addr,
		    param->left_rows,
		    Sl,
		    left_cols_per_channel,
		    param->left_cols);
    okk_gdma_32bit_matrix_S2L(
		    right_addr,
		    param->right_addr,
		    Sl,
		    param->right_cols,
		    right_cols_per_channel,
		    param->right_cols);
    for (int i = 0; i < Tll; i++) {
            okk_parallel_start();
            okk_gdma_32bit_matrix_S2L(
                            left_addrs[(i+1)%2],
                            param->left_addr + (i+1) * Sl * sizeof(float),
                            param->left_rows,
                            Sl,
                            left_cols_per_channel,
                            param->left_cols);
            okk_gdma_32bit_matrix_S2L(
                            right_addrs[(i+1)%2],
                            param->right_addr + (i+1) * Sl * param->right_cols * sizeof(float),
                            Sl,
                            param->right_cols,
                            right_cols_per_channel,
                            param->right_cols);
            okk_bdc_matmul(
                            output_addr,
                            left_addrs[i%2],
                            right_addrs[i%2],
                            NO_USE,
                            param->left_rows,
                            Sl,
                            param->right_cols,
                            left_cols_per_channel,
                            right_cols_per_channel,
                            false,
                            i);
            okk_parallel_end();
    }
    okk_bdc_matmul(
		    output_addr,
		    left_addrs[Tll%2],
		    right_addrs[Tll%2],
		    NO_USE,
		    param->left_rows,
		    Sl,
		    param->right_cols,
		    left_cols_per_channel,
		    right_cols_per_channel,
		    false,
		    true);
    okk_gdma_32bit_matrix_L2S(
        param->output_addr,
        output_addr,
        param->left_rows,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_poll();
}

void matmul_tilingE(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 output_stride, left_stride, right_stride;

    bool halfBlock = param->right_cols % 1024;
    int Se = 1024;
    int Te = DIV_UP(param->right_cols, Se);
    int Tee = Te - 1;
    int right_cols_per_channel = halfBlock ? 32 : 16;
    int left_cols_per_channel = halfBlock ? 16 : 64;
    int Re = param->right_cols - Tee * Se;

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
        .n = param->left_cols, .c = DIV_UP(Se, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    int right_size = right_stride.n * right_shape.n * sizeof(float);
    // Local output matrix tensor.
    local_addr_t output_addr = right_addr + right_size;
    dim4 output_shape = {
        .n = left_shape.n, .c = right_shape.c,
        .h = 1, .w = right_shape.w
    };
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);

    local_addr_t right_addr_2 = output_addr + output_stride.n * output_shape.n * sizeof(float);
    local_addr_t output_addr_2 = right_addr_2 + right_size;
    //if (output_addr_2 + output_stride.n * output_shape.n * sizeof(float) > LOCAL_MEM_SIZE) {
    //    return;
    //}
    local_addr_t right_addrs[2] = {right_addr, right_addr_2};
    local_addr_t output_addrs[2] = {output_addr, output_addr_2};
    // load left
    okk_gdma_32bit_matrix_S2L(
                    left_addr,
                    param->left_addr,
                    param->left_rows,
                    param->left_cols,
                    left_cols_per_channel,
                    param->left_cols);
    int i = 0;
    okk_gdma_32bit_matrix_S2L(
                    right_addrs[(i)%2],
                    param->right_addr + (i) * Se * sizeof(float),
                    param->left_cols,
                    Se,
                    right_cols_per_channel,
                    param->right_cols);
    okk_parallel_start();
    okk_gdma_32bit_matrix_S2L(
                    right_addrs[(i+1)%2],
                    param->right_addr + (i+1) * Se * sizeof(float),
                    param->left_cols,
                    Se,
                    right_cols_per_channel,
                    param->right_cols);
    // compute i
    okk_bdc_matmul(
                    output_addrs[i%2],
                    left_addr,
                    right_addrs[i%2],
                    NO_USE,
                    param->left_rows,
                    param->left_cols,
                    Se,
                    left_cols_per_channel,
                    right_cols_per_channel,
                    false,
                    false);
    okk_parallel_end();
    i++;
    for (; i < Tee; i++) {
            okk_parallel_start();
            // load right i + 1
            okk_gdma_32bit_matrix_S2L(
                            right_addrs[(i+1)%2],
                            param->right_addr + (i+1) * Se * sizeof(float),
                            param->left_cols,
                            i == Tee - 1 ? Re : Se,
                            right_cols_per_channel,
                            param->right_cols);

            // compute i
            okk_bdc_matmul(
                            output_addrs[i%2],
                            left_addr,
                            right_addrs[i%2],
                            NO_USE,
                            param->left_rows,
                            param->left_cols,
                            Se,
                            left_cols_per_channel,
                            right_cols_per_channel,
                            false,
                            false);

            // store i - 1
            okk_gdma_32bit_matrix_L2S(
                            param->output_addr + (i-1) * Se * sizeof(float),
                            output_addrs[(i-1)%2],
                            param->left_rows,
                            Se,
                            right_cols_per_channel,
                            param->right_cols);
            okk_parallel_end();
    }
    okk_parallel_start();
    // compute i
    okk_bdc_matmul(
                    output_addrs[i%2],
                    left_addr,
                    right_addrs[i%2],
                    NO_USE,
                    param->left_rows,
                    param->left_cols,
                    Re,
                    left_cols_per_channel,
                    right_cols_per_channel,
                    false,
                    false);
    // store i - 1
    okk_gdma_32bit_matrix_L2S(
                    param->output_addr + (i-1) * Se * sizeof(float),
                    output_addrs[(i-1)%2],
                    param->left_rows,
                    Se,
                    right_cols_per_channel,
                    param->right_cols);
    okk_parallel_end();
    okk_gdma_32bit_matrix_L2S(
                    param->output_addr + (i) * Se * sizeof(float),
                    output_addrs[i%2],
                    param->left_rows,
                    Re,
                    right_cols_per_channel,
                    param->right_cols);
    okk_poll();
}
void matmul_cpu(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 output_stride, left_stride, right_stride;

    int Tl = param->left_cols;
    int Sl = 1;
    int Block = 64;
    int left_cols_per_channel = Sl;
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);

    // Local left matrix tensor.
    local_addr_t left_addr = 0;
    dim4 left_shape = {
        .n = param->left_rows, .c = Block, .h = 1, .w = 1
    };
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    local_addr_t left_addr2 = left_addr + left_stride.n * left_shape.n * sizeof(float);
    local_addr_t right_addr = left_addr2 + left_stride.n * left_shape.n * sizeof(float);
    dim4 right_shape = {
        .n = Block, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    local_addr_t right_addr2 = right_addr + right_stride.n * right_shape.n * sizeof(float);
    local_addr_t output_addr = right_addr2 + right_stride.n * right_shape.n * sizeof(float);
    dim4 output_shape = {
        .n = param->left_rows, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    if (output_addr + output_stride.n * output_shape.n * sizeof(float) > LOCAL_MEM_SIZE) {
        return;
    }
    local_addr_t left_addrs[2] = { left_addr, left_addr2 };
    local_addr_t right_addrs[2] = { right_addr, right_addr2 };
    int i = 0;
    okk_gdma_32bit_matrix_S2L(
                    right_addrs[i%2],
                    param->right_addr + i * Block * param->right_cols * sizeof(float),
                    Block,
                    param->right_cols,
                    right_cols_per_channel,
                    param->right_cols);
    okk_gdma_32bit_matrix_S2L(
                    left_addrs[i%2],
                    param->left_addr + i * Block * sizeof(float),
                    param->left_rows,
                    Block,
                    left_cols_per_channel,
                    param->left_cols);
#define MM_ITEM(x)\
    okk_bdc_matmul(\
		    output_addr,\
		    left_addrs[i%2] + LOCAL_MEM_SIZE * (x),\
		    right_addrs[i%2] + (x) * right_stride.n * sizeof(float),\
		    NO_USE,\
		    param->left_rows,\
		    Sl,\
		    param->right_cols,\
		    left_cols_per_channel,\
		    right_cols_per_channel,\
		    false,\
		    true);
    for (; i < Tl/Block-1; i++) {
	    okk_parallel_start();
            okk_gdma_32bit_matrix_S2L(
                            right_addrs[(i+1)%2],
                            param->right_addr + (i+1) * Block * param->right_cols * sizeof(float),
                            Block,
                            param->right_cols,
                            right_cols_per_channel,
                            param->right_cols);
            okk_gdma_32bit_matrix_S2L(
                            left_addrs[(i+1)%2],
                            param->left_addr + (i+1) * Block * sizeof(float),
                            param->left_rows,
                            Block,
                            left_cols_per_channel,
                            param->left_cols);
	    okk_bdc_matmul(
			    output_addr,
			    left_addrs[i%2],
			    right_addrs[i%2],
			    NO_USE,
			    param->left_rows,
			    Sl,
			    param->right_cols,
			    left_cols_per_channel,
			    right_cols_per_channel,
			    false,
			    i);
	    // loop unroll is fast, reduce arm9 core -> npu call
	    MM_ITEM(1)
	    MM_ITEM(2)
	    MM_ITEM(3)
	    MM_ITEM(4)
	    MM_ITEM(5)
	    MM_ITEM(6)
	    MM_ITEM(7)
	    MM_ITEM(8)
	    MM_ITEM(9)
	    MM_ITEM(10)
	    MM_ITEM(11)
	    MM_ITEM(12)
	    MM_ITEM(13)
	    MM_ITEM(14)
	    MM_ITEM(15)
	    MM_ITEM(16)
	    MM_ITEM(17)
	    MM_ITEM(18)
	    MM_ITEM(19)
	    MM_ITEM(20)
	    MM_ITEM(21)
	    MM_ITEM(22)
	    MM_ITEM(23)
	    MM_ITEM(24)
	    MM_ITEM(25)
	    MM_ITEM(26)
	    MM_ITEM(27)
	    MM_ITEM(28)
	    MM_ITEM(29)
	    MM_ITEM(30)
	    MM_ITEM(31)
	    MM_ITEM(32)
	    MM_ITEM(33)
	    MM_ITEM(34)
	    MM_ITEM(35)
	    MM_ITEM(36)
	    MM_ITEM(37)
	    MM_ITEM(38)
	    MM_ITEM(39)
	    MM_ITEM(40)
	    MM_ITEM(41)
	    MM_ITEM(42)
	    MM_ITEM(43)
	    MM_ITEM(44)
	    MM_ITEM(45)
	    MM_ITEM(46)
	    MM_ITEM(47)
	    MM_ITEM(48)
	    MM_ITEM(49)
	    MM_ITEM(50)
	    MM_ITEM(51)
	    MM_ITEM(52)
	    MM_ITEM(53)
	    MM_ITEM(54)
	    MM_ITEM(55)
	    MM_ITEM(56)
	    MM_ITEM(57)
	    MM_ITEM(58)
	    MM_ITEM(59)
	    MM_ITEM(60)
	    MM_ITEM(61)
	    MM_ITEM(62)
	    MM_ITEM(63)
	    okk_parallel_end();
    }
	    MM_ITEM(0)
	    MM_ITEM(1)
	    MM_ITEM(2)
	    MM_ITEM(3)
	    MM_ITEM(4)
	    MM_ITEM(5)
	    MM_ITEM(6)
	    MM_ITEM(7)
	    MM_ITEM(8)
	    MM_ITEM(9)
	    MM_ITEM(10)
	    MM_ITEM(11)
	    MM_ITEM(12)
	    MM_ITEM(13)
	    MM_ITEM(14)
	    MM_ITEM(15)
	    MM_ITEM(16)
	    MM_ITEM(17)
	    MM_ITEM(18)
	    MM_ITEM(19)
	    MM_ITEM(20)
	    MM_ITEM(21)
	    MM_ITEM(22)
	    MM_ITEM(23)
	    MM_ITEM(24)
	    MM_ITEM(25)
	    MM_ITEM(26)
	    MM_ITEM(27)
	    MM_ITEM(28)
	    MM_ITEM(29)
	    MM_ITEM(30)
	    MM_ITEM(31)
	    MM_ITEM(32)
	    MM_ITEM(33)
	    MM_ITEM(34)
	    MM_ITEM(35)
	    MM_ITEM(36)
	    MM_ITEM(37)
	    MM_ITEM(38)
	    MM_ITEM(39)
	    MM_ITEM(40)
	    MM_ITEM(41)
	    MM_ITEM(42)
	    MM_ITEM(43)
	    MM_ITEM(44)
	    MM_ITEM(45)
	    MM_ITEM(46)
	    MM_ITEM(47)
	    MM_ITEM(48)
	    MM_ITEM(49)
	    MM_ITEM(50)
	    MM_ITEM(51)
	    MM_ITEM(52)
	    MM_ITEM(53)
	    MM_ITEM(54)
	    MM_ITEM(55)
	    MM_ITEM(56)
	    MM_ITEM(57)
	    MM_ITEM(58)
	    MM_ITEM(59)
	    MM_ITEM(60)
	    MM_ITEM(61)
	    MM_ITEM(62)
	    MM_ITEM(63)
    okk_gdma_32bit_matrix_L2S(
        param->output_addr,
        output_addr,
        param->left_rows,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_poll();
}

void matmul_depthwise(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;

    int C = 64;
    int Sn = 49;
    Sn = 56;
    int W = 1024;
    int H = 1;
    int N = 1568;
    int Tn = N / Sn;
    dim4 output_shape = {.n = Sn, .c = C, .h = 1, .w = 1};
    dim4 input_shape = {.n = Sn, .c = C, .h = H, .w = W};
    dim4 kernel_shape = {.n = 1, .c = C, .h = H, .w = W};
    dim4 output_stride, input_stride, kernel_stride;
    dim4 kstride = { .n = 1, .c = 0, .h = W, .w = 1 };

    int inputBlock = Sn * C * H * W * sizeof(float);
    int outputBlock = Sn * C * sizeof(float);
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t output_addr2 = output_addr + output_shape.n * output_stride.n * sizeof(float);
    local_addr_t input_addr = output_addr2 + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t input_addr2 = input_addr + input_shape.n * input_stride.n * sizeof(float);
    local_addr_t kernel_addr = input_addr2 + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    /*
    if (kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) >  LOCAL_MEM_SIZE) {
      OKKERNEL_LOG("Memory big!\n");
      return;
    }
    */
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->right_addr,
        &kernel_shape,
        &kernel_stride,
        &kstride);
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->left_addr,
        &input_shape,
        NULL,
        NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
        input_addr2,
        param->left_addr + inputBlock,
        &input_shape,
        NULL,
        NULL);
    okk_bdc_depthwise2d(
        output_addr,
        input_addr,
        kernel_addr,
        NO_USE,
        &input_shape,
        H,
        W,
        false,
        NULL,
        NULL,
        NULL);
    okk_parallel_end();
    int i = 1;
    for (; i < Tn-1; i++) {
      okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(
                            input_addrs[(i+1)%2],
                            param->left_addr + (i+1) * inputBlock,
                            &input_shape,
                            NULL,
                            NULL);
            okk_bdc_depthwise2d(
                            output_addrs[i%2],
                            input_addrs[i%2],
                            kernel_addr,
                            NO_USE,
                            &input_shape,
                            H,
                            W,
                            false,
                            NULL,
                            NULL,
                            NULL);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            NULL,
                            NULL);
      okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
        output_addrs[i%2],
        input_addrs[i%2],
        kernel_addr,
        NO_USE,
        &input_shape,
        H,
        W,
        false,
        NULL,
        NULL,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * outputBlock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * outputBlock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void matmul_contest(const void *args) {
    param_t *param = (param_t *)args;
    if (param->right_cols == 1) {
	matmul_depthwise(args); 
    } else if (param->left_cols >= 9216) {
        matmul_cpu(args);
    } else if (param->left_cols >= 2048 || param->left_rows %2) {
        matmul_tilingL(args);
    } else if (param->right_cols >= 3072) {
        matmul_tilingE(args);
    } else {
        matmul_tilingH(args);
    }
}
OKKERNEL_FUNC_REGISTER(matmul_contest);
