#ifndef _CANNY_H_
#define _CANNY_H_
#include <stdio.h>
#include <string.h>
#include "ed_pixel.h"


__global__ void cu_apply_gaussian_filter(pixel_t *in_pixels, pixel_t *out_pixels, int rows, int cols);
__device__ void trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels,
                                      unsigned idx, pixel_channel_t t_low, unsigned img_width);
__global__ void cu_apply_hysteresis(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                        pixel_channel_t t_high, pixel_channel_t t_low, 
                        unsigned img_height, unsigned img_width);
void cu_detectEdges(pixel_t *orig_pixels, int rows, int cols);

//cu_compute_intensity_gradient();
//cu_suppress_non_max();
//cu_apply_double_threshold();
//cu_apply_hysteresis(pixel_t *out_pixels, pixel_t *in_pixels, pixel_t hi_thld, pixel_t lo_thld);

#endif //_CANNY_H_
