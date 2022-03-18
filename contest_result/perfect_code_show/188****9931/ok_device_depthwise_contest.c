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

#if 0
void depthwise_tilingH(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 is, os;
    is.w = 1;
    is.h = param->W;
    is.c = param->H * is.h;
    is.n = param->C * is.c;
    os.w = 1;
    os.h = output_w;
    os.c = output_h * os.h;
    os.n = param->C * os.c;
    int Sh = 40;
    if (param->W < 256) Sh = 27;
    int Th = DIV_UP(output_h, Sh);
    int Rh = output_h - (Th-1)*Sh;
    int maxIH = (Sh - 1) * param->stride_h + param->kernel_h;
    int lastIH = (Rh - 1) * param->stride_h + param->kernel_h;
    dim4 output_shape = {.n = 1, .c = param->N * param->C, .h = Sh, .w = output_w};
    dim4 input_shape = {.n = 1, .c = param->N * param->C, .h = maxIH, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 output_stride, input_stride, kernel_stride;
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
    dim4 input_shape2 = {.n = input_shape.n, .c = input_shape.c, .h = input_shape.h, .w = input_shape.w };
    dim4 input_stride2;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    dim4* input_shapes[2] = { &input_shape, &input_shape2 };
    dim4* input_strides[2] = { &input_stride, &input_stride2 };
    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_S2L(
        kernel_addr + LOCAL_MEM_SIZE * 3,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_S2L(
        kernel_addr + LOCAL_MEM_SIZE * 6,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_S2L(
        kernel_addr + LOCAL_MEM_SIZE * 9,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    // depthwise
    Padding padding = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
    Padding padding2 = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
    Padding* paddings[2] = { &padding, &padding2 };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    padding.top = param->pad_top;
    input_shape.h = maxIH - padding.top;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    input_shape2.h = maxIH;
    okk_128_byte_aligned_stride_for_32bit(&input_stride2, 0, &input_shape2);
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr,
        &input_shape,
        NULL,
        &is);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
        input_addr2,
        param->input_addr + (Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
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
    okk_parallel_end();
    padding.top = 0;
    int i = 1;
    for (; i < Th-1; i++) {
    okk_parallel_start();
    int ih = (i + 1) * Sh * param->stride_h - param->pad_top;
    int IH = i == Th-2 ? lastIH : maxIH;
    if (ih + IH > param->H) {
	input_shapes[(i+1)%2]->h = param->H - ih;
	paddings[(i+1)%2]->bottom = IH - input_shapes[(i+1)%2]->h;
    } else {
	input_shapes[(i+1)%2]->h = IH;
    }
    okk_128_byte_aligned_stride_for_32bit(input_strides[(i+1)%2], 0, input_shapes[(i+1)%2]);
    okk_gdma_32bit_cpy_S2L(
        input_addrs[(i+1)%2],
        param->input_addr + ((i+1) * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        input_shapes[(i+1)%2],
        NULL,
        &is);
    okk_bdc_depthwise2d(
        output_addrs[i%2],
        input_addrs[i%2],
        kernel_addr,
        NO_USE,
        input_shapes[i%2],
        param->kernel_h,
        param->kernel_w,
        false,
        paddings[i%2],
        &stride,
        &dilation);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * Sh * output_w * sizeof(float),
        output_addrs[(i-1)%2],
        &output_shape,
        &os,
        NULL);
    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
        output_addrs[i%2],
        input_addrs[i%2],
        kernel_addr,
        NO_USE,
        input_shapes[i%2],
        param->kernel_h,
        param->kernel_w,
        false,
        paddings[i%2],
        &stride,
        &dilation);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * Sh * output_w * sizeof(float),
        output_addrs[(i-1)%2],
        &output_shape,
        &os,
        NULL);
    okk_parallel_end();
    output_shape.h = Rh;
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * Sh * output_w * sizeof(float),
        output_addrs[i%2],
        &output_shape,
        &os,
        NULL);
    okk_poll();
}

void depthwise_tilingC(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    int Sc = param->C / 4;
    if (param->C == 336) Sc = 168;
    if (param->C == 2048) Sc = 256;
    int Tc = param->C / Sc;
    dim4 output_shape = {.n = param->N, .c = Sc, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = Sc, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = Sc, .h = param->kernel_h, .w = param->kernel_w};
    dim4 output_stride, input_stride, kernel_stride;

    int inputBlock =  Sc * param->H * param->W * sizeof(float);
    int outputBlock = Sc * output_h * output_w * sizeof(float);
    int kernelBlock = Sc * kernel_shape.h * kernel_shape.w * sizeof(float);
    dim4 istride = { .n = param->C*param->H*param->W, .c = param->H*param->W, .h = param->W, .w = 1 };
    dim4 ostride = { .n = param->C*output_h*output_w, .c = output_h*output_w, .h = output_w, .w = 1 };
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
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &istride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kernelBlock,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    &istride);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tc-1; i++) {
            okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(i+1)%2],
			    param->kernel_addr + (i+1) * kernelBlock,
			    &kernel_shape,
			    &kernel_stride,
			    NULL);
	    okk_gdma_32bit_cpy_S2L(
			    input_addrs[(i+1)%2],
			    param->input_addr + (i+1) * inputBlock,
			    &input_shape,
			    NULL,
			    &istride);
	    okk_bdc_depthwise2d(
			    output_addrs[i%2],
			    input_addrs[i%2],
			    kernel_addrs[i%2],
			    NO_USE,
			    &input_shape,
			    param->kernel_h,
			    param->kernel_w,
			    false,
			    &padding,
			    &stride,
			    &dilation);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            &ostride,
                            NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
		    output_addrs[i%2],
		    input_addrs[i%2],
		    kernel_addrs[i%2],
		    NO_USE,
		    &input_shape,
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + (i-1) * outputBlock,
		    output_addrs[(i-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + i * outputBlock,
		    output_addrs[i%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}
void depthwise_tilingN(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;

    int Sn = 1;
    int Tn = param->N / Sn;
    dim4 output_shape = {.n = Sn, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = Sn, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 output_stride, input_stride, kernel_stride;

    int inputBlock = Sn * param->C * param->H * param->W * sizeof(float);
    int outputBlock = Sn * param->C * output_h * output_w * sizeof(float);
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
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    if (kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) >  LOCAL_MEM_SIZE) {
	//OKKERNEL_LOG("Memory big!\n");
 	return;
    }
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
	NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    NULL);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tn-1; i++) {
	    okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(
                            input_addrs[(i+1)%2],
                            param->input_addr + (i+1) * inputBlock,
                            &input_shape,
                            NULL,
                            NULL);
            okk_bdc_depthwise2d(
                            output_addrs[i%2],
                            input_addrs[i%2],
                            kernel_addr,
                            NO_USE,
                            &input_shape,
                            param->kernel_h,
                            param->kernel_w,
                            false,
                            &padding,
                            &stride,
                            &dilation);
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
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
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
#endif

void depthwise_0(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int output_w = 55;
    int Sh = 16;
    int Rh = 7;
    dim4 is = { .n = 150528, .c = 50176, .h = 224, .w = 1 };
    dim4 os = { .n = 9075, .c = 3025, .h = 55, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 12, .h = 16, .w = 55 };
    dim4 input_shape = { .n = 1, .c = 12, .h = 69, .w = 224 };
    dim4 input_shape2 = { .n = 1, .c = 12, .h = 71, .w = 224 };
    dim4 input_shape3 = { .n = 1, .c = 24, .h = 71, .w = 224 };
    dim4 kernel_shape = { .n = 1, .c = 3, .h = 11, .w = 11 };
    dim4 input_stride = { .n = 15456, .c = 15456, .h = 224, .w = 1 };
    dim4 kernel_stride = { .n = 121, .c = 121, .h = 11, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 3584;
    local_addr_t input_addr = 7168;
    local_addr_t input_addr2 = 68992;
    local_addr_t kernel_addr = 132608;
    dim4 kshape = { .n = 1, .c = 12, .h = 11, .w = 11 };
    dim4 kstride = { .n = 121, .c = 121, .h = 11, .w = 1 };
    dim4 kshape2 = { .n = 1, .c = 6, .h = 11, .w = 11 };
    dim4 kstride2 = { .n = 121, .c = 121, .h = 11, .w = 1 };

    Padding padding = { .top = param->pad_top, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_S2L(
        kernel_addr + LOCAL_MEM_SIZE * 3,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 6,
	kernel_addr,
	&kshape2,
	NULL,
	&kstride2);
    // depthwise
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr,
        &input_shape,
        NULL,
        &is);
    okk_parallel_start();
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
    // copy kernel
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 12,
	kernel_addr,
	&kshape,
	NULL,
	&kstride);
    // copy input
    okk_gdma_32bit_cpy_S2L(
        input_addr2,
        param->input_addr + (Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_gdma_32bit_cpy_S2L(
        input_addr2 + 12 * LOCAL_MEM_SIZE,
        param->input_addr + (2 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_parallel_end();
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        &os,
        NULL);
    padding.top = 0;
    padding.bottom = 0;
    okk_bdc_depthwise2d(
        output_addr2,
        input_addr2,
        kernel_addr,
        NO_USE,
        &input_shape3,
        param->kernel_h,
        param->kernel_w,
        false,
        &padding,
        &stride,
        &dilation);
    padding.bottom = 1;
    input_shape.h = 34;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr + (3 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape,
        NULL,
        &is);
    okk_parallel_end();
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + Sh * output_w * sizeof(float),
        output_addr2,
        &output_shape,
        &os,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 2 * Sh * output_w * sizeof(float),
        output_addr2 + 12 * LOCAL_MEM_SIZE,
        &output_shape,
        &os,
        NULL);
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
    okk_parallel_end();
    output_shape.h = Rh;
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 3 * Sh * output_w * sizeof(float),
        output_addr,
        &output_shape,
        &os,
        NULL);
    okk_poll();
}

void depthwise_1(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int output_w = 128;
    int Sh = 40;
    int Rh = 8;
    dim4 is = { .n = 196608, .c = 65536, .h = 256, .w = 1 };
    dim4 os = { .n = 49152, .c = 16384, .h = 128, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 12, .h = 40, .w = 128 };
    dim4 input_shape = { .n = 1, .c = 12, .h = 82, .w = 256 };
    dim4 input_shape2 = { .n = 1, .c = 12, .h = 85, .w = 256 };
    dim4 input_shape3 = { .n = 1, .c = 24, .h = 85, .w = 256 };
    dim4 kernel_shape = { .n = 1, .c = 3, .h = 7, .w = 7 };
    dim4 input_stride = { .n = 20992, .c = 20992, .h = 256, .w = 1 };
    dim4 kernel_stride = { .n = 49, .c = 49, .h = 7, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 20480;
    local_addr_t input_addr = 40960;
    local_addr_t input_addr2 = 124928;
    local_addr_t kernel_addr = 211968;
    dim4 kshape = { .n = 1, .c = 12, .h = 7, .w = 7 };
    dim4 kstride = { .n = 49, .c = 49, .h = 7, .w = 1 };
    dim4 kshape2 = { .n = 1, .c = 6, .h = 7, .w = 7 };
    dim4 kstride2 = { .n = 49, .c = 49, .h = 7, .w = 1 };
    Padding padding = { .top = param->pad_top, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_S2L(
        kernel_addr + LOCAL_MEM_SIZE * 3,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 6,
	kernel_addr,
	&kshape2,
	NULL,
	&kstride2);
    // depthwise
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr,
        &input_shape,
        NULL,
        &is);
    okk_parallel_start();
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
    // copy kernel
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 12,
	kernel_addr,
	&kshape,
	NULL,
	&kstride);
    // copy input
    okk_gdma_32bit_cpy_S2L(
        input_addr2,
        param->input_addr + (Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_gdma_32bit_cpy_S2L(
        input_addr2 + 12 * LOCAL_MEM_SIZE,
        param->input_addr + (2 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_parallel_end();
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        &os,
        NULL);
    padding.top = 0;
    padding.bottom = 0;
    okk_bdc_depthwise2d(
        output_addr2,
        input_addr2,
        kernel_addr,
        NO_USE,
        &input_shape3,
        param->kernel_h,
        param->kernel_w,
        false,
        &padding,
        &stride,
        &dilation);
    padding.bottom = 2;
    input_shape.h = 19;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr + (3 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape,
        NULL,
        &is);
    okk_parallel_end();
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + Sh * output_w * sizeof(float),
        output_addr2,
        &output_shape,
        &os,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 2 * Sh * output_w * sizeof(float),
        output_addr2 + 12 * LOCAL_MEM_SIZE,
        &output_shape,
        &os,
        NULL);
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
    okk_parallel_end();
    output_shape.h = Rh;
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 3 * Sh * output_w * sizeof(float),
        output_addr,
        &output_shape,
        &os,
        NULL);
    okk_poll();
}

void depthwise_2(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int output_w = 320;
    int Sh = 40;
    int Rh = 40;
    dim4 is = { .n = 1228800, .c = 409600, .h = 640, .w = 1 };
    dim4 os = { .n = 307200, .c = 102400, .h = 320, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 12, .h = 40, .w = 320 };
    dim4 input_shape = { .n = 1, .c = 12, .h = 81, .w = 640 };
    dim4 input_shape2 = { .n = 1, .c = 12, .h = 81, .w = 640 };
    dim4 input_shape3 = { .n = 1, .c = 24, .h = 81, .w = 640 };
    dim4 kernel_shape = { .n = 1, .c = 3, .h = 3, .w = 3 };
    dim4 input_stride = { .n = 51840, .c = 51840, .h = 640, .w = 1 };
    dim4 kernel_stride = { .n = 9, .c = 9, .h = 3, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 51200;
    local_addr_t input_addr = 102400;
    local_addr_t input_addr2 = 309760;
    local_addr_t kernel_addr = 517120;
    dim4 kshape = { .n = 1, .c = 12, .h = 3, .w = 3 };
    dim4 kstride = { .n = 9, .c = 9, .h = 3, .w = 1 };
    dim4 kshape1 = { .n = 1, .c = 6, .h = 3, .w = 3 };
    dim4 kstride1 = { .n = 9, .c = 9, .h = 3, .w = 1 };
    dim4 kshape2 = { .n = 1, .c = 24, .h = 3, .w = 3 };
    dim4 kstride2 = { .n = 9, .c = 9, .h = 3, .w = 1 };
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_S2L(
        kernel_addr + LOCAL_MEM_SIZE * 3,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 6,
	kernel_addr,
	&kshape1,
	NULL,
	&kstride1);
    // depthwise
    Padding padding = { .top = param->pad_top, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    input_shape.h = 80;
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr,
        &input_shape,
        NULL,
        &is);
    okk_parallel_start();
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
    // copy kernel
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 12,
	kernel_addr,
	&kshape,
	NULL,
	&kstride);
    // copy input
    okk_gdma_32bit_cpy_S2L(
        input_addr2,
        param->input_addr + (Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_gdma_32bit_cpy_S2L(
        input_addr2 + 12 * LOCAL_MEM_SIZE,
        param->input_addr + (2 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_parallel_end();
    padding.top = 0;
    padding.bottom = 0;
    // 1
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        &os,
        NULL);
    okk_bdc_depthwise2d(
        output_addr2,
        input_addr2,
        kernel_addr,
        NO_USE,
        &input_shape3,
        param->kernel_h,
        param->kernel_w,
        false,
        &padding,
        &stride,
        &dilation);
    // copy kernel
    okk_gdma_32bit_cpy_L2L(
	kernel_addr + LOCAL_MEM_SIZE * 24,
	kernel_addr,
	&kshape2,
	NULL,
	&kstride2);
    // copy input
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr + (3 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_gdma_32bit_cpy_S2L(
        input_addr + 12 * LOCAL_MEM_SIZE,
        param->input_addr + (4 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_gdma_32bit_cpy_S2L(
        input_addr + 24 * LOCAL_MEM_SIZE,
        param->input_addr + (5 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_gdma_32bit_cpy_S2L(
        input_addr + 36 * LOCAL_MEM_SIZE,
        param->input_addr + (6 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_parallel_end();
    input_shape3.c = 48;
    // 2
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + Sh * output_w * sizeof(float),
        output_addr2,
        &output_shape,
        &os,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 2 * Sh * output_w * sizeof(float),
        output_addr2 + 12 * LOCAL_MEM_SIZE,
        &output_shape,
        &os,
        NULL);
    okk_bdc_depthwise2d(
        output_addr,
        input_addr,
        kernel_addr,
        NO_USE,
        &input_shape3,
        param->kernel_h,
        param->kernel_w,
        false,
        &padding,
        &stride,
        &dilation);
    okk_gdma_32bit_cpy_S2L(
        input_addr2,
        param->input_addr + (7 * Sh  * param->stride_h - param->pad_top) * param->W * sizeof(float),
        &input_shape2,
        NULL,
        &is);
    okk_parallel_end();
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 3 * Sh * output_w * sizeof(float),
        output_addr,
        &output_shape,
        &os,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 4 * Sh * output_w * sizeof(float),
        output_addr + 12 * LOCAL_MEM_SIZE,
        &output_shape,
        &os,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 5 * Sh * output_w * sizeof(float),
        output_addr + 24 * LOCAL_MEM_SIZE,
        &output_shape,
        &os,
        NULL);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 6 * Sh * output_w * sizeof(float),
        output_addr + 36 * LOCAL_MEM_SIZE,
        &output_shape,
        &os,
        NULL);
    okk_bdc_depthwise2d(
        output_addr2,
        input_addr2,
        kernel_addr,
        NO_USE,
        &input_shape2,
        param->kernel_h,
        param->kernel_w,
        false,
        &padding,
        &stride,
        &dilation);
    okk_parallel_end();
    output_shape.h = Rh;
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + 7 * Sh * output_w * sizeof(float),
        output_addr2,
        &output_shape,
        &os,
        NULL);
    okk_poll();
}

void depthwise_3(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int Tn = 4;
    int inputBlock = 8640000;
    int outputBlock = 2160000;
    dim4 output_shape = { .n = 1, .c = 96, .h = 75, .w = 75 };
    dim4 input_shape = { .n = 1, .c = 96, .h = 150, .w = 150 };
    dim4 kernel_shape = { .n = 1, .c = 96, .h = 3, .w = 3 };
    dim4 kernel_stride = { .n = 18, .c = 9, .h = 3, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 45056;
    local_addr_t input_addr = 90112;
    local_addr_t input_addr2 = 270336;
    local_addr_t kernel_addr = 450560;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
	NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    NULL);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tn-1; i++) {
	    okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(
                            input_addrs[(i+1)%2],
                            param->input_addr + (i+1) * inputBlock,
                            &input_shape,
                            NULL,
                            NULL);
            okk_bdc_depthwise2d(
                            output_addrs[i%2],
                            input_addrs[i%2],
                            kernel_addr,
                            NO_USE,
                            &input_shape,
                            param->kernel_h,
                            param->kernel_w,
                            false,
                            &padding,
                            &stride,
                            &dilation);
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
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
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

void depthwise_4(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int Tc = 4;
    int inputBlock = 810000;
    int outputBlock = 810000;
    int kernelBlock = 1296;
    dim4 output_shape = { .n = 4, .c = 36, .h = 75, .w = 75 };
    dim4 input_shape = { .n = 4, .c = 36, .h = 75, .w = 75 };
    dim4 kernel_shape = { .n = 1, .c = 36, .h = 3, .w = 3 };
    dim4 kernel_stride = { .n = 9, .c = 9, .h = 3, .w = 1 };
    dim4 istride = { .n = 810000, .c = 5625, .h = 75, .w = 1 };
    dim4 ostride = { .n = 810000, .c = 5625, .h = 75, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 90112;
    local_addr_t input_addr = 180224;
    local_addr_t kernel_addr = 360448;
    local_addr_t kernel_addr2 = 360484;
    local_addr_t input_addr2 = 270336;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &istride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kernelBlock,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    &istride);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tc-1; i++) {
            okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(i+1)%2],
			    param->kernel_addr + (i+1) * kernelBlock,
			    &kernel_shape,
			    &kernel_stride,
			    NULL);
	    okk_gdma_32bit_cpy_S2L(
			    input_addrs[(i+1)%2],
			    param->input_addr + (i+1) * inputBlock,
			    &input_shape,
			    NULL,
			    &istride);
	    okk_bdc_depthwise2d(
			    output_addrs[i%2],
			    input_addrs[i%2],
			    kernel_addrs[i%2],
			    NO_USE,
			    &input_shape,
			    param->kernel_h,
			    param->kernel_w,
			    false,
			    &padding,
			    &stride,
			    &dilation);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            &ostride,
                            NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
		    output_addrs[i%2],
		    input_addrs[i%2],
		    kernel_addrs[i%2],
		    NO_USE,
		    &input_shape,
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + (i-1) * outputBlock,
		    output_addrs[(i-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + i * outputBlock,
		    output_addrs[i%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}

void depthwise_5(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int Tc = 4;
    int inputBlock = 277248;
    int outputBlock = 69312;
    int kernelBlock = 1728;
    dim4 output_shape = { .n = 4, .c = 48, .h = 19, .w = 19 };
    dim4 input_shape = { .n = 4, .c = 48, .h = 38, .w = 38 };
    dim4 kernel_shape = { .n = 1, .c = 48, .h = 3, .w = 3 };
    dim4 kernel_stride = { .n = 9, .c = 9, .h = 3, .w = 1 };
    dim4 istride = { .n = 277248, .c = 1444, .h = 38, .w = 1 };
    dim4 ostride = { .n = 69312, .c = 361, .h = 19, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 6144;
    local_addr_t input_addr = 12288;
    local_addr_t kernel_addr = 59392;
    local_addr_t kernel_addr2 = 59428;
    local_addr_t input_addr2 = 35840;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &istride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kernelBlock,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    &istride);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tc-1; i++) {
            okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(i+1)%2],
			    param->kernel_addr + (i+1) * kernelBlock,
			    &kernel_shape,
			    &kernel_stride,
			    NULL);
	    okk_gdma_32bit_cpy_S2L(
			    input_addrs[(i+1)%2],
			    param->input_addr + (i+1) * inputBlock,
			    &input_shape,
			    NULL,
			    &istride);
	    okk_bdc_depthwise2d(
			    output_addrs[i%2],
			    input_addrs[i%2],
			    kernel_addrs[i%2],
			    NO_USE,
			    &input_shape,
			    param->kernel_h,
			    param->kernel_w,
			    false,
			    &padding,
			    &stride,
			    &dilation);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            &ostride,
                            NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
		    output_addrs[i%2],
		    input_addrs[i%2],
		    kernel_addrs[i%2],
		    NO_USE,
		    &input_shape,
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + (i-1) * outputBlock,
		    output_addrs[(i-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + i * outputBlock,
		    output_addrs[i%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}
void depthwise_6(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int Tc = 2;
    int inputBlock = 565152;
    int outputBlock = 151200;
    int kernelBlock = 16800;
    dim4 output_shape = { .n = 4, .c = 168, .h = 15, .w = 15 };
    dim4 input_shape = { .n = 4, .c = 168, .h = 29, .w = 29 };
    dim4 kernel_shape = { .n = 1, .c = 168, .h = 5, .w = 5 };
    dim4 kernel_stride = { .n = 75, .c = 25, .h = 5, .w = 1 };
    dim4 istride = { .n = 282576, .c = 841, .h = 29, .w = 1 };
    dim4 ostride = { .n = 75600, .c = 225, .h = 15, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 12288;
    local_addr_t input_addr = 24576;
    local_addr_t kernel_addr = 107520;
    local_addr_t kernel_addr2 = 107820;
    local_addr_t input_addr2 = 66048;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &istride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kernelBlock,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    &istride);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tc-1; i++) {
            okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(i+1)%2],
			    param->kernel_addr + (i+1) * kernelBlock,
			    &kernel_shape,
			    &kernel_stride,
			    NULL);
	    okk_gdma_32bit_cpy_S2L(
			    input_addrs[(i+1)%2],
			    param->input_addr + (i+1) * inputBlock,
			    &input_shape,
			    NULL,
			    &istride);
	    okk_bdc_depthwise2d(
			    output_addrs[i%2],
			    input_addrs[i%2],
			    kernel_addrs[i%2],
			    NO_USE,
			    &input_shape,
			    param->kernel_h,
			    param->kernel_w,
			    false,
			    &padding,
			    &stride,
			    &dilation);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            &ostride,
                            NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
		    output_addrs[i%2],
		    input_addrs[i%2],
		    kernel_addrs[i%2],
		    NO_USE,
		    &input_shape,
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + (i-1) * outputBlock,
		    output_addrs[(i-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + i * outputBlock,
		    output_addrs[i%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}
void depthwise_7(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int inputBlock = 401408;
    int outputBlock = 100352;
    dim4 output_shape = { .n = 1, .c = 512, .h = 7, .w = 7 };
    dim4 input_shape = { .n = 1, .c = 512, .h = 14, .w = 14 };
    dim4 kernel_shape = { .n = 1, .c = 512, .h = 3, .w = 3 };
    dim4 kernel_stride = { .n = 72, .c = 9, .h = 3, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 2048;
    local_addr_t input_addr = 4096;
    local_addr_t input_addr2 = 11264;
    local_addr_t kernel_addr = 18432;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
	NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    NULL);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    NULL);
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
    okk_parallel_end();
    int i = 1;
    for (; i < 3; i++) {
	    okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(
                            input_addrs[(i+1)%2],
                            param->input_addr + (i+1) * inputBlock,
                            &input_shape,
                            NULL,
                            NULL);
            okk_bdc_depthwise2d(
                            output_addrs[i%2],
                            input_addrs[i%2],
                            kernel_addr,
                            NO_USE,
                            &input_shape,
                            param->kernel_h,
                            param->kernel_w,
                            false,
                            &padding,
                            &stride,
                            &dilation);
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
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
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
void depthwise_8(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int Tc = 4;
    int inputBlock = 752640;
    int outputBlock = 752640;
    int kernelBlock = 8640;
    dim4 output_shape = { .n = 4, .c = 240, .h = 28, .w = 28 };
    dim4 input_shape = { .n = 4, .c = 240, .h = 28, .w = 28 };
    dim4 kernel_shape = { .n = 1, .c = 240, .h = 3, .w = 3 };
    dim4 kernel_stride = { .n = 36, .c = 9, .h = 3, .w = 1 };
    dim4 istride = { .n = 752640, .c = 784, .h = 28, .w = 1 };
    dim4 ostride = { .n = 752640, .c = 784, .h = 28, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 51200;
    local_addr_t input_addr = 102400;
    local_addr_t kernel_addr = 204800;
    local_addr_t kernel_addr2 = 204944;
    local_addr_t input_addr2 = 153600;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &istride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kernelBlock,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    &istride);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tc-1; i++) {
            okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(i+1)%2],
			    param->kernel_addr + (i+1) * kernelBlock,
			    &kernel_shape,
			    &kernel_stride,
			    NULL);
	    okk_gdma_32bit_cpy_S2L(
			    input_addrs[(i+1)%2],
			    param->input_addr + (i+1) * inputBlock,
			    &input_shape,
			    NULL,
			    &istride);
	    okk_bdc_depthwise2d(
			    output_addrs[i%2],
			    input_addrs[i%2],
			    kernel_addrs[i%2],
			    NO_USE,
			    &input_shape,
			    param->kernel_h,
			    param->kernel_w,
			    false,
			    &padding,
			    &stride,
			    &dilation);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            &ostride,
                            NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
		    output_addrs[i%2],
		    input_addrs[i%2],
		    kernel_addrs[i%2],
		    NO_USE,
		    &input_shape,
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + (i-1) * outputBlock,
		    output_addrs[(i-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + i * outputBlock,
		    output_addrs[i%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}
void depthwise_9(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int Tc = 8;
    int inputBlock = 1115136;
    int outputBlock = 1115136;
    int kernelBlock = 9216;
    dim4 output_shape = { .n = 4, .c = 256, .h = 33, .w = 33 };
    dim4 input_shape = { .n = 4, .c = 256, .h = 33, .w = 33 };
    dim4 kernel_shape = { .n = 1, .c = 256, .h = 3, .w = 3 };
    dim4 kernel_stride = { .n = 36, .c = 9, .h = 3, .w = 1 };
    dim4 istride = { .n = 2230272, .c = 1089, .h = 33, .w = 1 };
    dim4 ostride = { .n = 2230272, .c = 1089, .h = 33, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 71680;
    local_addr_t input_addr = 143360;
    local_addr_t kernel_addr = 286720;
    local_addr_t kernel_addr2 = 286864;
    local_addr_t input_addr2 = 215040;
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &istride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kernelBlock,
		    &kernel_shape,
		    &kernel_stride,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    input_addr2,
		    param->input_addr + inputBlock,
		    &input_shape,
		    NULL,
		    &istride);
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
    okk_parallel_end();
    int i = 1;
    for (; i < Tc-1; i++) {
            okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(i+1)%2],
			    param->kernel_addr + (i+1) * kernelBlock,
			    &kernel_shape,
			    &kernel_stride,
			    NULL);
	    okk_gdma_32bit_cpy_S2L(
			    input_addrs[(i+1)%2],
			    param->input_addr + (i+1) * inputBlock,
			    &input_shape,
			    NULL,
			    &istride);
	    okk_bdc_depthwise2d(
			    output_addrs[i%2],
			    input_addrs[i%2],
			    kernel_addrs[i%2],
			    NO_USE,
			    &input_shape,
			    param->kernel_h,
			    param->kernel_w,
			    false,
			    &padding,
			    &stride,
			    &dilation);
            okk_gdma_32bit_cpy_L2S(
                            param->output_addr + (i-1) * outputBlock,
                            output_addrs[(i-1)%2],
                            &output_shape,
                            &ostride,
                            NULL);
    	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_depthwise2d(
		    output_addrs[i%2],
		    input_addrs[i%2],
		    kernel_addrs[i%2],
		    NO_USE,
		    &input_shape,
		    param->kernel_h,
		    param->kernel_w,
		    false,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + (i-1) * outputBlock,
		    output_addrs[(i-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + i * outputBlock,
		    output_addrs[i%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}
void depthwise_contest(const void *args) {
	param_t *param = (param_t *)args;
	switch (param->W) {
		case 224: depthwise_0(args); return;
		case 256: depthwise_1(args); return;
		case 640: depthwise_2(args); return;
		case 150: depthwise_3(args); return;
		case 75: depthwise_4(args); return;
		case 38: depthwise_5(args); return;
		case 29: depthwise_6(args); return;
		case 14: depthwise_7(args); return;
		case 28: depthwise_8(args); return;
		case 33: depthwise_9(args); return;
	}
}
OKKERNEL_FUNC_REGISTER(depthwise_contest);
