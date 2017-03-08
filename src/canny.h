
#ifndef _CANNY_H_
#define _CANNY_H_

#include <stdio.h>
#include <string.h>
#include "ed_pixel.h"
#define KERNEL_SIZE 7

void cu_detect_edges(pixel_t *orig_pixels, int rows, int cols, double kernel[KERNEL_SIZE][KERNEL_SIZE]);
void cu_test_hysteresis(pixel_channel_t *in, pixel_channel_t *out, unsigned row, unsigned cols); // TODO remove later

#endif //_CANNY_H_
