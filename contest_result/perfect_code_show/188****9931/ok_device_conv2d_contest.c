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

#if 0
void conv2d_tilingN(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int N = param->N;
    param->N = 1;
    const int IC_new = (param->IC + 1) / 2;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    dim4 output_stride, input_stride, kernel_stride;
    const int iblock = input_shape.c * input_shape.h * input_shape.w * sizeof(float);
    const int oblock = output_shape.c * output_shape.h * output_shape.w * sizeof(float);
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
    if (kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) > LOCAL_MEM_SIZE) {
        return;
    }
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
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
        param->input_addr + iblock,
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
    okk_parallel_end();
    int i = 1;
    for (; i < N - 1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            input_addrs[(i+1)%2],
            param->input_addr + (i+1) *iblock,
            &input_shape,
            NULL,
            NULL);
        okk_bdc_conv2d(
            output_addrs[i%2],
            input_addrs[i%2],
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
            param->output_addr + (i-1) * oblock,
            output_addrs[(i-1)%2],
            &output_shape,
            NULL,
            NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * oblock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_bdc_conv2d(
        output_addrs[i%2],
        input_addrs[i%2],
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
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * oblock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}

void conv2d_tilingNOC(const void *args, bool tilingN) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int origin_N = param->N;
    const int origin_OC = param->OC;
    if (tilingN) {
    	param->N = 1;
    }
    param->OC = 64;
    const int N = origin_N / param->N;
    const int OC = origin_OC / param->OC;
    const int IC_new = (param->IC + 1) / 2;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
    dim4 output_stride, input_stride, kernel_stride, kstride, ostride;
    const int iblock = input_shape.c * input_shape.h * input_shape.w * sizeof(float);
    const int oblock = param->OC * output_shape.h * output_shape.w * sizeof(float);
    const int kblock = param->OC * kernel_shape.h * kernel_shape.w * sizeof(float);
    kstride.w = 1;
    kstride.h = kernel_shape.w;
    kstride.c = kernel_shape.h * kernel_shape.w;
    kstride.n = kstride.c * origin_OC;
    ostride.w = 1;
    ostride.h = output_shape.w;
    ostride.c = output_shape.h * ostride.h;
    ostride.n = origin_OC * ostride.c;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t output_addr2 = output_addr + output_shape.n * output_stride.n * sizeof(float);
    local_addr_t input_addr = output_addr2 + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t kernel_addr = input_addr + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    for (int i = 0; i < N; i++) {
        okk_gdma_32bit_cpy_S2L(
            input_addr,
            param->input_addr + i * iblock,
            &input_shape,
            NULL,
            NULL);
        okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            &kstride);
        okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            kernel_addr2,
            param->kernel_addr + kblock,
            &kernel_shape,
            &kernel_stride,
            &kstride);
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
        okk_parallel_end();
        int j = 1;
        for (; j < OC-1; j++) {
            okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(
                kernel_addrs[(j+1)%2],
                param->kernel_addr + (j+1) * kblock,
                &kernel_shape,
                &kernel_stride,
                &kstride);
            okk_bdc_conv2d(
                output_addrs[j%2],
                input_addr,
                kernel_addrs[j%2],
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
                param->output_addr + (i * OC + j-1) * oblock,
                output_addrs[(j-1)%2],
                &output_shape,
                &ostride,
                NULL);
            okk_parallel_end();
        }
        okk_parallel_start();
        okk_bdc_conv2d(
            output_addrs[j%2],
            input_addr,
            kernel_addrs[j%2],
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
            param->output_addr + (i * OC + j-1) * oblock,
            output_addrs[(j-1)%2],
            &output_shape,
            &ostride,
            NULL);
        okk_parallel_end();
        okk_gdma_32bit_cpy_L2S(
            param->output_addr + (i * OC + j) * oblock,
            output_addrs[j%2],
            &output_shape,
            &ostride,
            NULL);
    }
    okk_poll();
}

#define DUMP_DIM4(x) \
OKKERNEL_LOG("dim4 " #x " = { .n = %d, .c = %d, .h = %d, .w = %d };\n", x.n, x.c, x.h, x.w);
#define DUMP_ADDR(x) \
OKKERNEL_LOG("local_addr_t " #x " = %d;\n", x);
#define DUMP_INT(x) \
OKKERNEL_LOG("int " #x " = %d;\n", x);
void conv2d_tilingH(const void *args) {
	okk_initialize();
	param_t *param = (param_t *)args;
	const int IC_new = (param->IC + 1) / 2;
	const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
	const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
	const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
	const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
	dim4 is, os;
	is.w = 1;
	is.h = param->W;
	is.c = param->H * is.h;
	is.n = param->IC * is.c;
	os.w = 1;
	os.h = output_w;
	os.c = output_h * os.h;
	os.n = param->OC * os.c;
	int OH = 9;
	if (param->W > 1024) OH = 2;
	int Th = DIV_UP(output_h, OH);
	int RH = output_h - (Th-1)*OH;
	int maxIH = (OH - 1) * param->stride_h + param->kernel_h;
	int lastIH = (RH - 1) * param->stride_h + param->kernel_h;
	dim4 output_shape = {.n = param->N, .c = param->OC, .h = OH, .w = output_w};
	dim4 input_shape = {.n = param->N, .c = param->IC, .h = maxIH, .w = param->W};
	dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w * 2};
	dim4 output_stride, input_stride, kernel_stride;
	local_addr_t output_addr = 0;
	okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
	local_addr_t output_addr2 = output_addr + output_shape.n * output_stride.n * sizeof(float);
	local_addr_t input_addr = output_addr2 + output_shape.n * output_stride.n * sizeof(float);
	okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
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
	dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
	dim4 kernel_stride_2IC;
	okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
	dim2 stride = {.h = param->stride_h, .w = param->stride_w};
	dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
	Padding padding = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
	Padding padding2 = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
	Padding* paddings[2] = { &padding, &padding2 };
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
				param->input_addr + (OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
				&input_shape2,
				NULL,
				&is);
		// comp
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
		okk_parallel_end();
		padding.top = 0;
		int oh = 1;
		for (; oh < Th-1; oh++) {
			okk_parallel_start();
			int ih = (oh + 1) * OH  * param->stride_h - param->pad_top;
			int IH = oh == Th-2 ? lastIH : maxIH;
			if (ih + IH > param->H) {
				input_shapes[(oh+1)%2]->h = param->H - ih;
				paddings[(oh+1)%2]->bottom = IH - input_shapes[(oh+1)%2]->h;
			} else {
				input_shapes[(oh+1)%2]->h = IH;
			}
			okk_128_byte_aligned_stride_for_32bit(input_strides[(oh+1)%2], 0, input_shapes[(oh+1)%2]);
			//OKKERNEL_LOG("oh = %d, ih = %d, h = %d, pad = {%d, %d}\n", oh+1, ih, input_shapes[(oh+1)%2]->h, paddings[(oh+1)%2]->top, paddings[(oh+1)%2]->bottom);
			okk_gdma_32bit_cpy_S2L(
					input_addrs[(oh+1)%2],
					param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
					input_shapes[(oh+1)%2],
					NULL,
					&is);
			// comp
			okk_bdc_conv2d(
					output_addrs[oh%2],
					input_addrs[oh%2],
					kernel_addr,
					NO_USE,
					input_shapes[oh%2],
					param->OC,
					param->kernel_h,
					param->kernel_w,
					input_strides[oh%2],
					&kernel_stride_2IC,
					false,
					false,
					paddings[oh%2],
					&stride,
					&dilation);
			// store i-1
			okk_gdma_32bit_cpy_L2S(
					param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
					output_addrs[(oh-1)%2],
					&output_shape,
					&os,
					NULL);
			okk_parallel_end();
		}
		okk_parallel_start();
		okk_bdc_conv2d(
				output_addrs[oh%2],
				input_addrs[oh%2],
				kernel_addr,
				NO_USE,
				input_shapes[oh%2],
				param->OC,
				param->kernel_h,
				param->kernel_w,
				input_strides[oh%2],
				&kernel_stride_2IC,
				false,
				false,
				paddings[oh%2],
				&stride,
				&dilation);
		okk_gdma_32bit_cpy_L2S(
				param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
				output_addrs[(oh-1)%2],
				&output_shape,
				&os,
				NULL);
		okk_parallel_end();
		output_shape.h = RH;
		okk_gdma_32bit_cpy_L2S(
				param->output_addr + (oh * OH * output_w) * sizeof(float),
				output_addrs[oh%2],
				&output_shape,
				&os,
				NULL);
	okk_poll();
}
void conv2d_cpu(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int IC_new = (param->IC + 1) / 2;
    int Tc = IC_new;
    IC_new = 1;
    int IC = param->IC / Tc;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = param->kernel_h * param->kernel_w, .w = 2};
    dim4 output_stride, input_stride, kernel_stride;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t input_addr2 = input_addr + input_shape.n * input_stride.n * sizeof(float);
    local_addr_t kernel_addr = input_addr2 + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    if (kernel_addr2 + kernel_shape.n * kernel_stride.n * sizeof(float) > LOCAL_MEM_SIZE) {
        OKKERNEL_LOG("Memory big!\n");
        return;
    }
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    dim4 i_stride = { .n = param->IC * param->W * param->H, .c = param->W * param->H, .h = param->W, .w = 1 };
    dim4 k_stride = { .n = param->OC * param->kernel_h * param->kernel_w * 2, .c = param->kernel_h * param->kernel_w * 2, .h = 2, .w = 1 };
    dim4 k_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    x32 zero = { .fp32 = 0 };
    for (int hw = 0; hw < param->kernel_h * param->kernel_w; hw++) {
        for (int c = 0; c < Tc; c++) {
            if (c == 0) {
                okk_gdma_32bit_cpy_S2L(
                    input_addr,
                    param->input_addr + c * IC * param->H * param->W * sizeof(float),
                    &input_shape,
                    NULL,
                    &i_stride);
                okk_gdma_32bit_set_C_local(kernel_addr, zero, &kernel_shape, &kernel_stride);
                okk_gdma_32bit_cpy_S2L(
                    kernel_addr + hw * 2 * sizeof(float),
                    param->kernel_addr + (c * param->OC * param->kernel_h * param->kernel_w + hw) * 2 * sizeof(float),
                    &k_shape,
                    &kernel_stride,
                    &k_stride);
            }
            okk_gdma_32bit_set_C_local(kernel_addr2, zero, &kernel_shape, &kernel_stride);
            okk_parallel_start();
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
                (hw>0 || c>0),
                &padding,
                &stride,
                &dilation);
            okk_gdma_32bit_cpy_S2L(
                kernel_addr2 + (hw * 2 + 1) * sizeof(float),
                param->kernel_addr + ((c * param->OC * param->kernel_h * param->kernel_w + hw) * 2 + 1) * sizeof(float),
                &k_shape,
                &kernel_stride,
                &k_stride);
            okk_parallel_end();
            okk_gdma_32bit_set_C_local(kernel_addr, zero, &kernel_shape, &kernel_stride);
            okk_parallel_start();
            okk_bdc_conv2d(
                output_addr,
                input_addr,
                kernel_addr2,
                NO_USE,
                &input_shape,
                param->OC,
                param->kernel_h,
                param->kernel_w,
                &input_stride,
                &kernel_stride_2IC,
                false,
                true,
                &padding,
                &stride,
                &dilation);
            if (c < Tc - 1) {
                okk_gdma_32bit_cpy_S2L(
                    kernel_addr + hw * 2 * sizeof(float),
                    param->kernel_addr + ((c + 1) * param->OC * param->kernel_h * param->kernel_w + hw) * 2 * sizeof(float),
                    &k_shape,
                    &kernel_stride,
                    &k_stride);
                okk_gdma_32bit_cpy_S2L(
                    input_addr2,
                    param->input_addr + (c + 1) * IC * param->H * param->W * sizeof(float),
                    &input_shape,
                    NULL,
                    &i_stride);
            }
            okk_parallel_end();
            local_addr_t tmp = input_addr2;
            input_addr2 = input_addr;
            input_addr = tmp;
        }
    }
    // copy output from local memory to global memory
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
#endif

void conv2d_0(const void *args) {
	okk_initialize();
	param_t *param = (param_t *)args;
	int output_w = 320;
	int OH = 9;
	int Th = 36;
	int RH = 5;
	int maxIH = 19;
	dim4 is = { .n = 1228800, .c = 409600, .h = 640, .w = 1 };
	dim4 os = { .n = 1638400, .c = 102400, .h = 320, .w = 1 };
	dim4 output_shape = { .n = 4, .c = 16, .h = 9, .w = 320 };
	dim4 input_shape = { .n = 4, .c = 3, .h = 19, .w = 640 };
	dim4 input_shape2 = { .n = 4, .c = 3, .h = 19, .w = 640 };
	dim4 kernel_shape = { .n = 2, .c = 16, .h = 3, .w = 6 };
	dim4 input_stride = { .n = 12160, .c = 12160, .h = 640, .w = 1 };
	dim4 kernel_stride = { .n = 18, .c = 18, .h = 6, .w = 1 };
	dim4 kernel_stride_2IC = { .n = 9, .c = 9, .h = 3, .w = 1 };
	local_addr_t output_addr = 0;
	local_addr_t output_addr2 = 46080;
	local_addr_t input_addr = 92160;
	local_addr_t input_addr2 = 286720;
	local_addr_t kernel_addr = 481280;
	dim4 input_stride2;
	local_addr_t input_addrs[2] = { input_addr, input_addr2 };
	local_addr_t output_addrs[2] = { output_addr, output_addr2 };
	// copy kernel from global memory to local memory
	okk_gdma_32bit_cpy_S2L(
			kernel_addr,
			param->kernel_addr,
			&kernel_shape,
			&kernel_stride,
			NULL);
	dim2 stride = {.h = param->stride_h, .w = param->stride_w};
	dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
	Padding padding = { .top = param->pad_top, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
	input_shape2.h = maxIH - padding.top;
	okk_128_byte_aligned_stride_for_32bit(&input_stride2, 0, &input_shape2);
	input_shape.h = 19;
	okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
	okk_gdma_32bit_cpy_S2L(
			input_addr,
			param->input_addr,
			&input_shape2,
			NULL,
			&is);
	okk_parallel_start();
	okk_gdma_32bit_cpy_S2L(
			input_addr2,
			param->input_addr + (OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape,
			NULL,
			&is);
	// comp
	okk_bdc_conv2d(
			output_addr,
			input_addr,
			kernel_addr,
			NO_USE,
			&input_shape2,
			param->OC,
			param->kernel_h,
			param->kernel_w,
			&input_stride2,
			&kernel_stride_2IC,
			false,
			false,
			&padding,
			&stride,
			&dilation);
	okk_parallel_end();
	padding.top = 0;
	int oh = 1;
	for (; oh < Th-2; oh++) {
		okk_parallel_start();
		okk_gdma_32bit_cpy_S2L(
				input_addrs[(oh+1)%2],
				param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
				&input_shape,
				NULL,
				&is);
		// comp
		okk_bdc_conv2d(
				output_addrs[oh%2],
				input_addrs[oh%2],
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
		// store i-1
		okk_gdma_32bit_cpy_L2S(
				param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
				output_addrs[(oh-1)%2],
				&output_shape,
				&os,
				NULL);
		okk_parallel_end();
	}
	okk_parallel_start();
	input_shape2.h = 11;
	okk_128_byte_aligned_stride_for_32bit(&input_stride2, 0, &input_shape2);
	okk_gdma_32bit_cpy_S2L(
			input_addrs[(oh+1)%2],
			param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape2,
			NULL,
			&is);
	// comp
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
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
	// store i-1
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	oh++;
	okk_parallel_start();
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
			kernel_addr,
			NO_USE,
			&input_shape2,
			param->OC,
			param->kernel_h,
			param->kernel_w,
			&input_stride2,
			&kernel_stride_2IC,
			false,
			false,
			&padding,
			&stride,
			&dilation);
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	output_shape.h = RH;
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + (oh * OH * output_w) * sizeof(float),
			output_addrs[oh%2],
			&output_shape,
			&os,
			NULL);
	okk_poll();
}
void conv2d_1(const void *args) {
	okk_initialize();
	param_t *param = (param_t *)args;
	int output_w = 960;
	int OH = 5;
	int Th = 103;
	int RH = 2;
	int maxIH = 11;
	dim4 is = { .n = 1474560, .c = 491520, .h = 960, .w = 1 };
	dim4 os = { .n = 7864320, .c = 491520, .h = 960, .w = 1 };
	dim4 output_shape = { .n = 4, .c = 16, .h = 5, .w = 960 };
	dim4 input_shape = { .n = 4, .c = 3, .h = 11, .w = 960 };
	dim4 input_shape2 = { .n = 4, .c = 3, .h = 11, .w = 960 };
	dim4 kernel_shape = { .n = 2, .c = 16, .h = 7, .w = 14 };
	dim4 input_stride = { .n = 10560, .c = 10560, .h = 960, .w = 1 };
	dim4 kernel_stride = { .n = 98, .c = 98, .h = 14, .w = 1 };
	dim4 kernel_stride_2IC = { .n = 49, .c = 49, .h = 7, .w = 1 };
	local_addr_t output_addr = 0;
	local_addr_t output_addr2 = 76800;
	local_addr_t input_addr = 153600;
	local_addr_t input_addr2 = 322560;
	local_addr_t kernel_addr = 491520;
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
	dim2 stride = {.h = param->stride_h, .w = param->stride_w};
	dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
	Padding padding = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
	Padding padding2 = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
	Padding* paddings[2] = { &padding, &padding2 };
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
			param->input_addr + (OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape2,
			NULL,
			&is);
	// comp
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
	okk_parallel_end();
	padding.top = 0;
	padding.bottom = 0;
	input_shape.h = 11;
	okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
	int oh = 1;
	for (; oh < Th-3; oh++) {
		okk_parallel_start();
		okk_gdma_32bit_cpy_S2L(
				input_addrs[(oh+1)%2],
				param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
				&input_shape,
				NULL,
				&is);
		// comp
		okk_bdc_conv2d(
				output_addrs[oh%2],
				input_addrs[oh%2],
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
		// store i-1
		okk_gdma_32bit_cpy_L2S(
				param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
				output_addrs[(oh-1)%2],
				&output_shape,
				&os,
				NULL);
		okk_parallel_end();
	}
	okk_parallel_start();
	input_shapes[(oh+1)%2]->h = 10;
	paddings[(oh+1)%2]->bottom = 1;
	okk_128_byte_aligned_stride_for_32bit(input_strides[(oh+1)%2], 0, input_shapes[(oh+1)%2]);
	okk_gdma_32bit_cpy_S2L(
			input_addrs[(oh+1)%2],
			param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			input_shapes[(oh+1)%2],
			NULL,
			&is);
	// comp
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
			kernel_addr,
			NO_USE,
			input_shapes[oh%2],
			param->OC,
			param->kernel_h,
			param->kernel_w,
			input_strides[oh%2],
			&kernel_stride_2IC,
			false,
			false,
			paddings[oh%2],
			&stride,
			&dilation);
	// store i-1
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	oh++;
	okk_parallel_start();
	input_shapes[(oh+1)%2]->h = 5;
	paddings[(oh+1)%2]->bottom = 3;
	okk_128_byte_aligned_stride_for_32bit(input_strides[(oh+1)%2], 0, input_shapes[(oh+1)%2]);
	okk_gdma_32bit_cpy_S2L(
			input_addrs[(oh+1)%2],
			param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			input_shapes[(oh+1)%2],
			NULL,
			&is);
	// comp
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
			kernel_addr,
			NO_USE,
			input_shapes[oh%2],
			param->OC,
			param->kernel_h,
			param->kernel_w,
			input_strides[oh%2],
			&kernel_stride_2IC,
			false,
			false,
			paddings[oh%2],
			&stride,
			&dilation);
	// store i-1
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	oh++;
	okk_parallel_start();
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
			kernel_addr,
			NO_USE,
			input_shapes[oh%2],
			param->OC,
			param->kernel_h,
			param->kernel_w,
			input_strides[oh%2],
			&kernel_stride_2IC,
			false,
			false,
			paddings[oh%2],
			&stride,
			&dilation);
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	output_shape.h = RH;
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + (oh * OH * output_w) * sizeof(float),
			output_addrs[oh%2],
			&output_shape,
			&os,
			NULL);
	okk_poll();
}
void conv2d_2(const void *args) {
	okk_initialize();
	param_t *param = (param_t *)args;
	int output_w = 1920;
	int OH = 2;
	int Th = 540;
	int RH = 2;
	int maxIH = 4;
	dim4 is = { .n = 6220800, .c = 2073600, .h = 1920, .w = 1 };
	dim4 os = { .n = 33177600, .c = 2073600, .h = 1920, .w = 1 };
	dim4 output_shape = { .n = 4, .c = 16, .h = 2, .w = 1920 };
	dim4 input_shape = { .n = 4, .c = 3, .h = 4, .w = 1920 };
	dim4 input_shape2 = { .n = 4, .c = 3, .h = 4, .w = 1920 };
	dim4 kernel_shape = { .n = 2, .c = 16, .h = 3, .w = 6 };
	dim4 input_stride = { .n = 7680, .c = 7680, .h = 1920, .w = 1 };
	dim4 kernel_stride = { .n = 18, .c = 18, .h = 6, .w = 1 };
	dim4 kernel_stride_2IC = { .n = 9, .c = 9, .h = 3, .w = 1 };
	local_addr_t output_addr = 0;
	local_addr_t output_addr2 = 61440;
	local_addr_t input_addr = 122880;
	local_addr_t input_addr2 = 245760;
	local_addr_t kernel_addr = 368640;
	dim4 input_stride2;
	local_addr_t input_addrs[2] = { input_addr, input_addr2 };
	local_addr_t output_addrs[2] = { output_addr, output_addr2 };
	// copy kernel from global memory to local memory
	okk_gdma_32bit_cpy_S2L(
			kernel_addr,
			param->kernel_addr,
			&kernel_shape,
			&kernel_stride,
			NULL);
	dim2 stride = {.h = param->stride_h, .w = param->stride_w};
	dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
	Padding padding = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
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
			param->input_addr + (OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape2,
			NULL,
			&is);
	// comp
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
	okk_parallel_end();
	padding.top = 0;
	int oh = 1;
	for (; oh < Th-2; oh++) {
		okk_parallel_start();
		okk_gdma_32bit_cpy_S2L(
				input_addrs[(oh+1)%2],
				param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
				&input_shape2,
				NULL,
				&is);
		// comp
		okk_bdc_conv2d(
				output_addrs[oh%2],
				input_addrs[oh%2],
				kernel_addr,
				NO_USE,
				&input_shape2,
				param->OC,
				param->kernel_h,
				param->kernel_w,
				&input_stride2,
				&kernel_stride_2IC,
				false,
				false,
				&padding,
				&stride,
				&dilation);
		// store i-1
		okk_gdma_32bit_cpy_L2S(
				param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
				output_addrs[(oh-1)%2],
				&output_shape,
				&os,
				NULL);
		okk_parallel_end();
	}
	okk_parallel_start();
	input_shape.h = 3;
	okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
	okk_gdma_32bit_cpy_S2L(
			input_addrs[(oh+1)%2],
			param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape,
			NULL,
			&is);
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
			kernel_addr,
			NO_USE,
			&input_shape2,
			param->OC,
			param->kernel_h,
			param->kernel_w,
			&input_stride2,
			&kernel_stride_2IC,
			false,
			false,
			&padding,
			&stride,
			&dilation);
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	oh++;
	okk_parallel_start();
	padding.bottom = 1;
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
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
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	output_shape.h = RH;
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + (oh * OH * output_w) * sizeof(float),
			output_addrs[oh%2],
			&output_shape,
			&os,
			NULL);
	okk_poll();
}
void conv2d_3(const void *args) {
	okk_initialize();
	param_t *param = (param_t *)args;
	int output_w = 96;
	int OH = 9;
	int Th = 11;
	int RH = 6;
	int maxIH = 39;
	dim4 is = { .n = 442368, .c = 147456, .h = 384, .w = 1 };
	dim4 os = { .n = 221184, .c = 9216, .h = 96, .w = 1 };
	dim4 output_shape = { .n = 4, .c = 24, .h = 9, .w = 96 };
	dim4 input_shape = { .n = 4, .c = 3, .h = 39, .w = 384 };
	dim4 input_shape2 = { .n = 4, .c = 3, .h = 39, .w = 384 };
	dim4 kernel_shape = { .n = 2, .c = 24, .h = 7, .w = 14 };
	dim4 input_stride = { .n = 14976, .c = 14976, .h = 384, .w = 1 };
	dim4 kernel_stride = { .n = 98, .c = 98, .h = 14, .w = 1 };
	dim4 kernel_stride_2IC = { .n = 49, .c = 49, .h = 7, .w = 1 };
	local_addr_t output_addr = 0;
	local_addr_t output_addr2 = 13824;
	local_addr_t input_addr = 27648;
	local_addr_t input_addr2 = 267264;
	local_addr_t kernel_addr = 506880;
	dim4 input_stride2;
	local_addr_t input_addrs[2] = { input_addr, input_addr2 };
	local_addr_t output_addrs[2] = { output_addr, output_addr2 };
	// copy kernel from global memory to local memory
	okk_gdma_32bit_cpy_S2L(
			kernel_addr,
			param->kernel_addr,
			&kernel_shape,
			&kernel_stride,
			NULL);
	dim2 stride = {.h = param->stride_h, .w = param->stride_w};
	dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
	Padding padding = { .top = 0, .bottom = 0, .left = param->pad_left, .right = param->pad_right };
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
			param->input_addr + (OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape2,
			NULL,
			&is);
	// comp
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
	okk_parallel_end();
	padding.top = 0;
	padding.bottom = 0;
	int oh = 1;
	for (; oh < Th-2; oh++) {
		okk_parallel_start();
		okk_gdma_32bit_cpy_S2L(
				input_addrs[(oh+1)%2],
				param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
				&input_shape2,
				NULL,
				&is);
		// comp
		okk_bdc_conv2d(
				output_addrs[oh%2],
				input_addrs[oh%2],
				kernel_addr,
				NO_USE,
				&input_shape2,
				param->OC,
				param->kernel_h,
				param->kernel_w,
				&input_stride2,
				&kernel_stride_2IC,
				false,
				false,
				&padding,
				&stride,
				&dilation);
		// store i-1
		okk_gdma_32bit_cpy_L2S(
				param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
				output_addrs[(oh-1)%2],
				&output_shape,
				&os,
				NULL);
		okk_parallel_end();
	}
	okk_parallel_start();
	input_shape.h = 25;
	okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
	okk_gdma_32bit_cpy_S2L(
			input_addrs[(oh+1)%2],
			param->input_addr + ((oh + 1) * OH  * param->stride_h - param->pad_top) * param->W * sizeof(float),
			&input_shape,
			NULL,
			&is);
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
			kernel_addr,
			NO_USE,
			&input_shape2,
			param->OC,
			param->kernel_h,
			param->kernel_w,
			&input_stride2,
			&kernel_stride_2IC,
			false,
			false,
			&padding,
			&stride,
			&dilation);
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	oh++;
	okk_parallel_start();
	padding.bottom = 2;
	okk_bdc_conv2d(
			output_addrs[oh%2],
			input_addrs[oh%2],
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
			param->output_addr + ((oh-1) * OH * output_w) * sizeof(float),
			output_addrs[(oh-1)%2],
			&output_shape,
			&os,
			NULL);
	okk_parallel_end();
	output_shape.h = RH;
	okk_gdma_32bit_cpy_L2S(
			param->output_addr + (oh * OH * output_w) * sizeof(float),
			output_addrs[oh%2],
			&output_shape,
			&os,
			NULL);
	okk_poll();
}
void conv2d_4(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->N = 1;
    int N = 4;
    int iblock = 618348;
    int oblock = 1161600;
    dim4 output_shape = { .n = 1, .c = 96, .h = 55, .w = 55 };
    dim4 input_shape = { .n = 1, .c = 3, .h = 227, .w = 227 };
    dim4 kernel_shape = { .n = 2, .c = 96, .h = 11, .w = 22 };
    dim4 input_stride = { .n = 51552, .c = 51552, .h = 227, .w = 1 };
    dim4 kernel_stride = { .n = 484, .c = 242, .h = 22, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 242, .c = 121, .h = 11, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 24320;
    local_addr_t input_addr = 48640;
    local_addr_t input_addr2 = 254848;
    local_addr_t kernel_addr = 461056;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
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
        param->input_addr + iblock,
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
    okk_parallel_end();
    int i = 1;
    for (; i < N - 1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            input_addrs[(i+1)%2],
            param->input_addr + (i+1) *iblock,
            &input_shape,
            NULL,
            NULL);
        okk_bdc_conv2d(
            output_addrs[i%2],
            input_addrs[i%2],
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
            param->output_addr + (i-1) * oblock,
            output_addrs[(i-1)%2],
            &output_shape,
            NULL,
            NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * oblock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_bdc_conv2d(
        output_addrs[i%2],
        input_addrs[i%2],
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
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * oblock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_5(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->N = 1;
    int N = 4;
    int iblock = 193548;
    int oblock = 2673408;
    dim4 output_shape = { .n = 1, .c = 192, .h = 59, .w = 59 };
    dim4 input_shape = { .n = 1, .c = 3, .h = 127, .w = 127 };
    dim4 kernel_shape = { .n = 2, .c = 192, .h = 11, .w = 22 };
    dim4 input_stride = { .n = 16160, .c = 16160, .h = 127, .w = 1 };
    dim4 kernel_stride = { .n = 726, .c = 242, .h = 22, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 363, .c = 121, .h = 11, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 41856;
    local_addr_t input_addr = 83712;
    local_addr_t input_addr2 = 148352;
    local_addr_t kernel_addr = 212992;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
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
        param->input_addr + iblock,
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
    okk_parallel_end();
    int i = 1;
    for (; i < N - 1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            input_addrs[(i+1)%2],
            param->input_addr + (i+1) *iblock,
            &input_shape,
            NULL,
            NULL);
        okk_bdc_conv2d(
            output_addrs[i%2],
            input_addrs[i%2],
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
            param->output_addr + (i-1) * oblock,
            output_addrs[(i-1)%2],
            &output_shape,
            NULL,
            NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * oblock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_bdc_conv2d(
        output_addrs[i%2],
        input_addrs[i%2],
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
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * oblock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_6(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->N = 1;
    int N = 4;
    int iblock = 3538944;
    int oblock = 1769472;
    dim4 output_shape = { .n = 1, .c = 36, .h = 96, .w = 128 };
    dim4 input_shape = { .n = 1, .c = 18, .h = 192, .w = 256 };
    dim4 kernel_shape = { .n = 9, .c = 36, .h = 3, .w = 6 };
    dim4 input_stride = { .n = 49152, .c = 49152, .h = 256, .w = 1 };
    dim4 kernel_stride = { .n = 18, .c = 18, .h = 6, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 9, .c = 9, .h = 3, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 49152;
    local_addr_t input_addr = 98304;
    local_addr_t input_addr2 = 294912;
    local_addr_t kernel_addr = 491520;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
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
        param->input_addr + iblock,
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
    okk_parallel_end();
    int i = 1;
    for (; i < N - 1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            input_addrs[(i+1)%2],
            param->input_addr + (i+1) *iblock,
            &input_shape,
            NULL,
            NULL);
        okk_bdc_conv2d(
            output_addrs[i%2],
            input_addrs[i%2],
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
            param->output_addr + (i-1) * oblock,
            output_addrs[(i-1)%2],
            &output_shape,
            NULL,
            NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * oblock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_bdc_conv2d(
        output_addrs[i%2],
        input_addrs[i%2],
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
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * oblock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_7(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->N = 1;
    int N = 4;
    int iblock = 7372800;
    int oblock = 1843200;
    dim4 output_shape = { .n = 1, .c = 72, .h = 80, .w = 80 };
    dim4 input_shape = { .n = 1, .c = 72, .h = 160, .w = 160 };
    dim4 kernel_shape = { .n = 36, .c = 72, .h = 3, .w = 6 };
    dim4 input_stride = { .n = 51200, .c = 25600, .h = 160, .w = 1 };
    dim4 kernel_stride = { .n = 36, .c = 18, .h = 6, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 18, .c = 9, .h = 3, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 51200;
    local_addr_t input_addr = 102400;
    local_addr_t input_addr2 = 307200;
    local_addr_t kernel_addr = 512000;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
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
        param->input_addr + iblock,
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
    okk_parallel_end();
    int i = 1;
    for (; i < N - 1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            input_addrs[(i+1)%2],
            param->input_addr + (i+1) *iblock,
            &input_shape,
            NULL,
            NULL);
        okk_bdc_conv2d(
            output_addrs[i%2],
            input_addrs[i%2],
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
            param->output_addr + (i-1) * oblock,
            output_addrs[(i-1)%2],
            &output_shape,
            NULL,
            NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * oblock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_bdc_conv2d(
        output_addrs[i%2],
        input_addrs[i%2],
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
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * oblock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_8(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->N = 1;
    int N = 4;
    int iblock = 1280000;
    int oblock = 2560000;
    dim4 output_shape = { .n = 1, .c = 256, .h = 50, .w = 50 };
    dim4 input_shape = { .n = 1, .c = 128, .h = 50, .w = 50 };
    dim4 kernel_shape = { .n = 64, .c = 256, .h = 3, .w = 6 };
    dim4 input_stride = { .n = 5056, .c = 2528, .h = 50, .w = 1 };
    dim4 kernel_stride = { .n = 72, .c = 18, .h = 6, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 36, .c = 9, .h = 3, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 40448;
    local_addr_t input_addr = 80896;
    local_addr_t input_addr2 = 101120;
    local_addr_t kernel_addr = 121344;
    local_addr_t input_addrs[2] = { input_addr, input_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
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
        param->input_addr + iblock,
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
    okk_parallel_end();
    int i = 1;
    for (; i < N - 1; i++) {
    	okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            input_addrs[(i+1)%2],
            param->input_addr + (i+1) *iblock,
            &input_shape,
            NULL,
            NULL);
        okk_bdc_conv2d(
            output_addrs[i%2],
            input_addrs[i%2],
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
            param->output_addr + (i-1) * oblock,
            output_addrs[(i-1)%2],
            &output_shape,
            NULL,
            NULL);
    	okk_parallel_end();
    }
    okk_parallel_start();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + (i-1) * oblock,
        output_addrs[(i-1)%2],
        &output_shape,
        NULL,
        NULL);
    okk_bdc_conv2d(
        output_addrs[i%2],
        input_addrs[i%2],
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
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr + i * oblock,
        output_addrs[i%2],
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_9(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->OC = 64;
    int OC = 3;
    int oblock = 230400;
    int kblock = 3584;
    dim4 kstride = { .n = 2688, .c = 14, .h = 2, .w = 1 };
    dim4 ostride = { .n = 172800, .c = 900, .h = 30, .w = 1 };
    dim4 output_shape = { .n = 4, .c = 64, .h = 30, .w = 30 };
    dim4 input_shape = { .n = 4, .c = 160, .h = 30, .w = 30 };
    dim4 kernel_shape = { .n = 80, .c = 64, .h = 7, .w = 2 };
    dim4 input_stride = { .n = 2784, .c = 928, .h = 30, .w = 1 };
    dim4 kernel_stride = { .n = 14, .c = 14, .h = 2, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 7, .c = 7, .h = 1, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 14848;
    local_addr_t input_addr = 29696;
    local_addr_t kernel_addr = 74240;
    local_addr_t kernel_addr2 = 78720;
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    NULL);
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &kernel_shape,
		    &kernel_stride,
		    &kstride);
    okk_parallel_start();
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2,
		    param->kernel_addr + kblock,
		    &kernel_shape,
		    &kernel_stride,
		    &kstride);
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
    okk_parallel_end();
    int j = 1;
    for (; j < OC-1; j++) {
	    okk_parallel_start();
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addrs[(j+1)%2],
			    param->kernel_addr + (j+1) * kblock,
			    &kernel_shape,
			    &kernel_stride,
			    &kstride);
	    okk_bdc_conv2d(
			    output_addrs[j%2],
			    input_addr,
			    kernel_addrs[j%2],
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
			    param->output_addr + (j-1) * oblock,
			    output_addrs[(j-1)%2],
			    &output_shape,
			    &ostride,
			    NULL);
	    okk_parallel_end();
    }
    okk_parallel_start();
    okk_bdc_conv2d(
		    output_addrs[j%2],
		    input_addr,
		    kernel_addrs[j%2],
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
		    param->output_addr + (j-1) * oblock,
		    output_addrs[(j-1)%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
		    param->output_addr + j * oblock,
		    output_addrs[j%2],
		    &output_shape,
		    &ostride,
		    NULL);
    okk_poll();
}
void conv2d_10(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    param->N = 1;
    param->OC = 64;
    int OC = 8;
    int N = 4;
    int iblock = 16777216;
    int oblock = 1048576;
    int kblock = 512;
    dim4 kstride = { .n = 1024, .c = 2, .h = 2, .w = 1 };
    dim4 ostride = { .n = 2097152, .c = 4096, .h = 64, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 64, .h = 64, .w = 64 };
    dim4 input_shape = { .n = 1, .c = 256, .h = 128, .w = 128 };
    dim4 kernel_shape = { .n = 128, .c = 64, .h = 1, .w = 2 };
    dim4 input_stride = { .n = 65536, .c = 16384, .h = 128, .w = 1 };
    dim4 kernel_stride = { .n = 2, .c = 2, .h = 2, .w = 1 };
    dim4 kernel_stride_2IC = { .n = 1, .c = 1, .h = 1, .w = 1 };
    local_addr_t output_addr = 0;
    local_addr_t output_addr2 = 16384;
    local_addr_t input_addr = 32768;
    local_addr_t kernel_addr = 294912;
    local_addr_t kernel_addr2 = 295936;
    local_addr_t kernel_addrs[2] = { kernel_addr, kernel_addr2 };
    local_addr_t output_addrs[2] = { output_addr, output_addr2 };
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    for (int i = 0; i < N; i++) {
        okk_gdma_32bit_cpy_S2L(
            input_addr,
            param->input_addr + i * iblock,
            &input_shape,
            NULL,
            NULL);
        okk_gdma_32bit_cpy_S2L(
            kernel_addr,
            param->kernel_addr,
            &kernel_shape,
            &kernel_stride,
            &kstride);
        okk_parallel_start();
        okk_gdma_32bit_cpy_S2L(
            kernel_addr2,
            param->kernel_addr + kblock,
            &kernel_shape,
            &kernel_stride,
            &kstride);
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
        okk_parallel_end();
        int j = 1;
        for (; j < OC-1; j++) {
            okk_parallel_start();
            okk_gdma_32bit_cpy_S2L(
                kernel_addrs[(j+1)%2],
                param->kernel_addr + (j+1) * kblock,
                &kernel_shape,
                &kernel_stride,
                &kstride);
            okk_bdc_conv2d(
                output_addrs[j%2],
                input_addr,
                kernel_addrs[j%2],
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
                param->output_addr + (i * OC + j-1) * oblock,
                output_addrs[(j-1)%2],
                &output_shape,
                &ostride,
                NULL);
            okk_parallel_end();
        }
        okk_parallel_start();
        okk_bdc_conv2d(
            output_addrs[j%2],
            input_addr,
            kernel_addrs[j%2],
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
            param->output_addr + (i * OC + j-1) * oblock,
            output_addrs[(j-1)%2],
            &output_shape,
            &ostride,
            NULL);
        okk_parallel_end();
        okk_gdma_32bit_cpy_L2S(
            param->output_addr + (i * OC + j) * oblock,
            output_addrs[j%2],
            &output_shape,
            &ostride,
            NULL);
    }
    okk_poll();
}
#define CONV_ITEM(flag, x, h, w)\
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);\
    okk_gdma_32bit_cpy_S2L(\
		    input_addr,\
		    param->input_addr + (h * param->W + w) * sizeof(float),\
		    &input_shape,\
		    &input_stride,\
		    &i_stride);\
    okk_gdma_32bit_cpy_S2L(\
		    kernel_addr,\
		    param->kernel_addr + x * 8,\
		    &k_shape,\
		    &kernel_stride,\
		    &k_stride);\
    for (c = 0; c < Tc-2; c+=2) {\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr,\
			    kernel_addr,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    flag,\
			    &padding,\
			    &stride,\
			    &dilation);\
	    okk_gdma_32bit_cpy_S2L(\
			    kernel_addr2 + sizeof(float),\
			    param->kernel_addr + ((c * param->OC * param->kernel_h * param->kernel_w + x) * 2 + 1) * sizeof(float),\
			    &k_shape,\
			    &kernel_stride,\
			    &k_stride);\
	    okk_parallel_end();\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr,\
			    kernel_addr2,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);\
		    okk_gdma_32bit_cpy_S2L(\
				    kernel_addr,\
				    param->kernel_addr + ((c + 1) * param->OC * param->kernel_h * param->kernel_w + x) * 2 * sizeof(float),\
				    &k_shape,\
				    &kernel_stride,\
				    &k_stride);\
		    okk_gdma_32bit_cpy_S2L(\
				    input_addr2,\
				    param->input_addr + (((c + 1) * IC * param->H + h) * param->W + w) * sizeof(float),\
				    &input_shape,\
				    NULL,\
				    &i_stride);\
	    okk_parallel_end();\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr2,\
			    kernel_addr,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);\
	    okk_gdma_32bit_cpy_S2L(\
			    kernel_addr2 + sizeof(float),\
			    param->kernel_addr + (((c+1) * param->OC * param->kernel_h * param->kernel_w + x) * 2 + 1) * sizeof(float),\
			    &k_shape,\
			    &kernel_stride,\
			    &k_stride);\
	    okk_parallel_end();\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr2,\
			    kernel_addr2,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);\
		    okk_gdma_32bit_cpy_S2L(\
				    kernel_addr,\
				    param->kernel_addr + ((c + 2) * param->OC * param->kernel_h * param->kernel_w + x) * 2 * sizeof(float),\
				    &k_shape,\
				    &kernel_stride,\
				    &k_stride);\
		    okk_gdma_32bit_cpy_S2L(\
				    input_addr,\
				    param->input_addr + (((c + 2) * IC * param->H + h) * param->W + w) * sizeof(float),\
				    &input_shape,\
				    NULL,\
				    &i_stride);\
	    okk_parallel_end();\
    }\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr,\
			    kernel_addr,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);\
	    okk_gdma_32bit_cpy_S2L(\
			    kernel_addr2 + sizeof(float),\
			    param->kernel_addr + ((c * param->OC * param->kernel_h * param->kernel_w + x) * 2 + 1) * sizeof(float),\
			    &k_shape,\
			    &kernel_stride,\
			    &k_stride);\
	    okk_parallel_end();\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr,\
			    kernel_addr2,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);\
		    okk_gdma_32bit_cpy_S2L(\
				    kernel_addr,\
				    param->kernel_addr + ((c + 1) * param->OC * param->kernel_h * param->kernel_w + x) * 2 * sizeof(float),\
				    &k_shape,\
				    &kernel_stride,\
				    &k_stride);\
		    okk_gdma_32bit_cpy_S2L(\
				    input_addr2,\
				    param->input_addr + (((c + 1) * IC * param->H + h) * param->W + w) * sizeof(float),\
				    &input_shape,\
				    NULL,\
				    &i_stride);\
	    okk_parallel_end();\
	    okk_parallel_start();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr2,\
			    kernel_addr,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);\
	    okk_gdma_32bit_cpy_S2L(\
			    kernel_addr2 + sizeof(float),\
			    param->kernel_addr + (((c+1) * param->OC * param->kernel_h * param->kernel_w + x) * 2 + 1) * sizeof(float),\
			    &k_shape,\
			    &kernel_stride,\
			    &k_stride);\
	    okk_parallel_end();\
	    okk_bdc_conv2d(\
			    output_addr,\
			    input_addr2,\
			    kernel_addr2,\
			    NO_USE,\
			    &input_shape,\
			    param->OC,\
			    1,\
			    1,\
			    &input_stride,\
			    &kernel_stride_2IC,\
			    false,\
			    true,\
			    &padding,\
			    &stride,\
			    &dilation);

void conv2d_11(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int IC_new = (param->IC + 1) / 2;
    int Tc = IC_new;
    IC_new = 1;
    int IC = param->IC / Tc;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 2};
    dim4 output_stride, input_stride, kernel_stride;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t input_addr2 = input_addr + input_shape.n * input_stride.n * sizeof(float);
    local_addr_t kernel_addr = input_addr2 + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    dim4 i_stride = { .n = param->IC * param->W * param->H, .c = param->W * param->H, .h = param->W, .w = 1 };
    dim4 k_stride = { .n = param->OC * param->kernel_h * param->kernel_w * 2, .c = param->kernel_h * param->kernel_w * 2, .h = 2, .w = 1 };
    dim4 k_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    x32 zero = { .fp32 = 0 };
    okk_gdma_32bit_set_C_local(kernel_addr, zero, &kernel_shape, &kernel_stride);
    okk_gdma_32bit_set_C_local(kernel_addr2, zero, &kernel_shape, &kernel_stride);
    int c = 0;
    // kh = 0, kw = 0
    input_shape.h = 24;
    input_shape.w = 24;
    padding.bottom = 0;
    padding.right = 0;
    //okk_parallel_start();
    CONV_ITEM(c, 0, 0, 0)
    // kh = 0, kw = 1
    input_shape.h = 24;
    padding.top = 4;
    padding.bottom = 0;
    input_shape.w = 28;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 1, 0, 0)
    // kh = 0, kw = 2
    input_shape.h = 24;
    padding.top = 4;
    padding.bottom = 0;
    input_shape.w = 24;
    padding.left = 0;
    padding.right = 4;
    CONV_ITEM(true, 2, 0, 4)
    // kh = 1, kw = 0
    input_shape.h = 28;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 24;
    padding.left = 4;
    padding.right = 0;
    CONV_ITEM(true, 3, 0, 0)
    // kh = 1, kw = 1
    input_shape.h = 28;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 28;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 4, 0, 0)
    // kh = 1, kw = 2
    input_shape.h = 28;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 24;
    padding.left = 0;
    padding.right = 4;
    CONV_ITEM(true, 5, 0, 4)
    // kh = 2, kw = 0
    input_shape.h = 24;
    padding.top = 0;
    padding.bottom = 4;
    input_shape.w = 24;
    padding.left = 4;
    padding.right = 0;
    CONV_ITEM(true, 6, 4, 0)
    // kh = 2, kw = 1
    input_shape.h = 24;
    padding.top = 0;
    padding.bottom = 4;
    input_shape.w = 28;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 7, 4, 0)
    // kh = 2, kw = 2
    input_shape.h = 24;
    padding.top = 0;
    padding.bottom = 4;
    input_shape.w = 24;
    padding.left = 0;
    padding.right = 4;
    CONV_ITEM(true, 8, 4, 4)
    //okk_parallel_end();
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_12(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int IC_new = (param->IC + 1) / 2;
    int Tc = IC_new;
    IC_new = 1;
    int IC = param->IC / Tc;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 2};
    dim4 output_stride, input_stride, kernel_stride;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t input_addr2 = input_addr + input_shape.n * input_stride.n * sizeof(float);
    local_addr_t kernel_addr = input_addr2 + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    dim4 i_stride = { .n = param->IC * param->W * param->H, .c = param->W * param->H, .h = param->W, .w = 1 };
    dim4 k_stride = { .n = param->OC * param->kernel_h * param->kernel_w * 2, .c = param->kernel_h * param->kernel_w * 2, .h = 2, .w = 1 };
    dim4 k_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    x32 zero = { .fp32 = 0 };
    okk_gdma_32bit_set_C_local(kernel_addr, zero, &kernel_shape, &kernel_stride);
    okk_gdma_32bit_set_C_local(kernel_addr2, zero, &kernel_shape, &kernel_stride);
    int c = 0;
    // kh = 0, kw = 0
    input_shape.h = 9;
    padding.bottom = 0;
    input_shape.w = 9;
    padding.right = 0;
    //okk_parallel_start();
    CONV_ITEM(c, 0, 0, 0)
    // kh = 0, kw = 1
    input_shape.h = 9;
    input_shape.w = 10;
    padding.left = 0;
    CONV_ITEM(true, 1, 0, 0)
    // kh = 0, kw = 2
    input_shape.h = 9;
    input_shape.w = 9;
    padding.left = 0;
    padding.right = 1;
    CONV_ITEM(true, 2, 0, 1)
    // kh = 1, kw = 0
    input_shape.h = 10;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 9;
    padding.left = 1;
    padding.right = 0;
    CONV_ITEM(true, 3, 0, 0)
    // kh = 1, kw = 1
    input_shape.h = 10;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 10;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 4, 0, 0)
    // kh = 1, kw = 2
    input_shape.h = 10;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 9;
    padding.left = 0;
    padding.right = 1;
    CONV_ITEM(true, 5, 0, 1)
    // kh = 2, kw = 0
    input_shape.h = 9;
    padding.top = 0;
    padding.bottom = 1;
    input_shape.w = 9;
    padding.left = 1;
    padding.right = 0;
    CONV_ITEM(true, 6, 1, 0)
    // kh = 2, kw = 1
    input_shape.h = 9;
    padding.top = 0;
    padding.bottom = 1;
    input_shape.w = 10;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 7, 1, 0)
    // kh = 2, kw = 2
    input_shape.h = 9;
    padding.top = 0;
    padding.bottom = 1;
    input_shape.w = 9;
    padding.left = 0;
    padding.right = 1;
    CONV_ITEM(true, 8, 1, 1)
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_13(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int IC_new = (param->IC + 1) / 2;
    int Tc = IC_new;
    IC_new = 1;
    int IC = param->IC / Tc;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 2};
    dim4 output_stride, input_stride, kernel_stride;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t input_addr2 = input_addr + input_shape.n * input_stride.n * sizeof(float);
    local_addr_t kernel_addr = input_addr2 + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    dim4 i_stride = { .n = param->IC * param->W * param->H, .c = param->W * param->H, .h = param->W, .w = 1 };
    dim4 k_stride = { .n = param->OC * param->kernel_h * param->kernel_w * 2, .c = param->kernel_h * param->kernel_w * 2, .h = 2, .w = 1 };
    dim4 k_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    x32 zero = { .fp32 = 0 };
    okk_gdma_32bit_set_C_local(kernel_addr, zero, &kernel_shape, &kernel_stride);
    okk_gdma_32bit_set_C_local(kernel_addr2, zero, &kernel_shape, &kernel_stride);
    int c = 0;
    // kh = 0, kw = 0
    input_shape.h = 16;
    input_shape.w = 16;
    padding.bottom = 0;
    padding.right = 0;
    //okk_parallel_start();
    CONV_ITEM(c, 0, 0, 0)
    // kh = 0, kw = 1
    input_shape.h = 16;
    input_shape.w = 28;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 1, 0, 0)
    // kh = 0, kw = 2
    input_shape.h = 16;
    input_shape.w = 16;
    padding.left = 0;
    padding.right = 12;
    CONV_ITEM(true, 2, 0, 12)
    // kh = 1, kw = 0
    input_shape.h = 28;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 16;
    padding.left = 12;
    padding.right = 0;
    CONV_ITEM(true, 3, 0, 0)
    // kh = 1, kw = 1
    input_shape.h = 28;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 28;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 4, 0, 0)
    // kh = 1, kw = 2
    input_shape.h = 28;
    padding.top = 0;
    padding.bottom = 0;
    input_shape.w = 16;
    padding.left = 0;
    padding.right = 12;
    CONV_ITEM(true, 5, 0, 12)
    // kh = 2, kw = 0
    input_shape.h = 16;
    padding.top = 0;
    padding.bottom = 12;
    input_shape.w = 16;
    padding.left = 12;
    padding.right = 0;
    CONV_ITEM(true, 6, 12, 0)
    // kh = 2, kw = 1
    input_shape.h = 16;
    padding.top = 0;
    padding.bottom = 12;
    input_shape.w = 28;
    padding.left = 0;
    padding.right = 0;
    CONV_ITEM(true, 7, 12, 0)
    // kh = 2, kw = 2
    input_shape.h = 16;
    padding.top = 0;
    padding.bottom = 12;
    input_shape.w = 16;
    padding.left = 0;
    padding.right = 12;
    CONV_ITEM(true, 8, 12, 12)
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_14(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    int IC_new = (param->IC + 1) / 2;
    int Tc = IC_new;
    IC_new = 1;
    int IC = param->IC / Tc;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->OC, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = IC, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 2};
    dim4 output_stride, input_stride, kernel_stride;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t input_addr2 = input_addr + input_shape.n * input_stride.n * sizeof(float);
    local_addr_t kernel_addr = input_addr2 + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    local_addr_t kernel_addr2 = kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float);
    // conv2d
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    // view the data type of kernel as fp32x2
    dim4 kernel_shape_2IC = {.n = IC_new, .c = param->OC, .h = param->kernel_h, .w = param->kernel_w};
    dim4 kernel_stride_2IC;
    okk_compact_stride(&kernel_stride_2IC, 0, &kernel_shape_2IC);
    dim4 i_stride = { .n = param->IC * param->W * param->H, .c = param->W * param->H, .h = param->W, .w = 1 };
    dim4 k_stride = { .n = param->OC * param->kernel_h * param->kernel_w * 2, .c = param->kernel_h * param->kernel_w * 2, .h = 2, .w = 1 };
    dim4 k_shape = {.n = IC_new, .c = param->OC, .h = 1, .w = 1};
    x32 zero = { .fp32 = 0 };
    okk_gdma_32bit_set_C_local(kernel_addr, zero, &kernel_shape, &kernel_stride);
    okk_gdma_32bit_set_C_local(kernel_addr2, zero, &kernel_shape, &kernel_stride);
    okk_gdma_32bit_cpy_S2L(
		    input_addr,
		    param->input_addr,
		    &input_shape,
		    NULL,
		    &i_stride);
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr,
		    param->kernel_addr,
		    &k_shape,
		    &kernel_stride,
		    &k_stride);
    for (int c = 0; c < Tc-1; c++) {
	    okk_parallel_start();
	    okk_bdc_conv2d(
			    output_addr,
			    input_addr,
			    kernel_addr,
			    NO_USE,
			    &input_shape,
			    672,
			    1,
			    1,
			    &input_stride,
			    &kernel_stride_2IC,
			    false,
			    c,
			    &padding,
			    &stride,
			    &dilation);
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addr2 + sizeof(float),
			    param->kernel_addr + (c * 1344 + 1) * sizeof(float),
			    &k_shape,
			    &kernel_stride,
			    &k_stride);
	    okk_parallel_end();
	    okk_parallel_start();
	    okk_bdc_conv2d(
			    output_addr,
			    input_addr,
			    kernel_addr2,
			    NO_USE,
			    &input_shape,
			    672,
			    1,
			    1,
			    &input_stride,
			    &kernel_stride_2IC,
			    false,
			    true,
			    &padding,
			    &stride,
			    &dilation);
	    okk_gdma_32bit_cpy_S2L(
			    kernel_addr,
			    param->kernel_addr + (c + 1) * 5376,
			    &k_shape,
			    &kernel_stride,
			    &k_stride);
	    okk_gdma_32bit_cpy_S2L(
			    input_addr2,
			    param->input_addr + (c + 1) * 968,
			    &input_shape,
			    NULL,
			    &i_stride);
	    okk_parallel_end();
	    local_addr_t tmp = input_addr2;
	    input_addr2 = input_addr;
	    input_addr = tmp;
    }
    okk_parallel_start();
    okk_bdc_conv2d(
		    output_addr,
		    input_addr,
		    kernel_addr,
		    NO_USE,
		    &input_shape,
		    672,
		    1,
		    1,
		    &input_stride,
		    &kernel_stride_2IC,
		    false,
		    true,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_S2L(
		    kernel_addr2 + sizeof(float),
		    param->kernel_addr + ((Tc-1) * 1344 + 1) * sizeof(float),
		    &k_shape,
		    &kernel_stride,
		    &k_stride);
    okk_parallel_end();
    okk_bdc_conv2d(
		    output_addr,
		    input_addr,
		    kernel_addr2,
		    NO_USE,
		    &input_shape,
		    672,
		    1,
		    1,
		    &input_stride,
		    &kernel_stride_2IC,
		    false,
		    true,
		    &padding,
		    &stride,
		    &dilation);
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
void conv2d_contest(const void *args) {
    param_t *param = (param_t *)args;
    switch (param->IC) {
        case 512:  conv2d_11(args); return;
        case 1024: conv2d_12(args); return;
        case 2048: conv2d_13(args); return;
        case 4032: conv2d_14(args); return;
    }
    switch (param->W) {
        case 640: conv2d_0(args); return;
        case 960: conv2d_1(args); return;
        case 1920: conv2d_2(args); return;
        case 384: conv2d_3(args); return;
        case 227: conv2d_4(args); return;
        case 127: conv2d_5(args); return;
        case 256: conv2d_6(args); return;
        case 160: conv2d_7(args); return;
        case 50: conv2d_8(args); return;
 	case 30:  conv2d_9(args); return;	
 	case 128: conv2d_10(args); return;	
    }
}
OKKERNEL_FUNC_REGISTER(conv2d_contest);
