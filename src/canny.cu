
#include <stdio.h>
#include <string.h>
#include "ed_pixel.h"

#define EDGE 0xFFFF

__device__ void trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                                     unsigned idx, pixel_channel_t t_low, unsigned img_width);

__global__
void cu_apply_hysteresis(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                        pixel_channel_t t_high, pixel_channel_t t_low, 
                        unsigned img_height, unsigned img_width)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix; // TODO fix indexing
    if (iy <= img_height && ix <= img_width) {

        // apply high threshold
        if ((in_pixels[idx] > t_high) && (out_pixels[idx] != EDGE)) {
            out_pixels[idx] = EDGE;
        }
        // apply low threshold to neighbors
        trace_immed_neighbors(out_pixels, in_pixels, idx, t_low, img_width);
    }
}

__device__
void trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, 
                            unsigned idx, pixel_channel_t t_low, unsigned img_width)
{
    /* directions representing indices of neighbors */
    unsigned n, s, e, w;
    unsigned nw, ne, sw, se;

    /* get indices */
    n = idx - img_width;
    nw = n - 1;
    ne = n + 1;
    s = idx + img_width;
    sw = s - 1;
    se = s + 1;
    w = idx - 1;
    e = idx + 1;

    if ((in_pixels[nw] >= t_low) && (out_pixels[nw] != EDGE)) {
        out_pixels[nw] = EDGE;
    }
    if ((in_pixels[n] >= t_low) && (out_pixels[n] != EDGE)) {
        out_pixels[n] = EDGE;
    }
    if ((in_pixels[ne] >= t_low) && (out_pixels[ne] != EDGE)) {
        out_pixels[ne] = EDGE;
    }
    if ((in_pixels[w] >= t_low) && (out_pixels[w] != EDGE)) {
        out_pixels[w] = EDGE;
    }
    if ((in_pixels[e] >= t_low) && (out_pixels[e] != EDGE)) {
        out_pixels[e] = EDGE;
    }
    if ((in_pixels[sw] >= t_low) && (out_pixels[sw] != EDGE)) {
        out_pixels[sw] = EDGE;
    }
    if ((in_pixels[s] >= t_low) && (out_pixels[s] != EDGE)) {
        out_pixels[s] = EDGE;
    }
    if ((in_pixels[se] >= t_low) && (out_pixels[se] != EDGE)) {
        out_pixels[se] = EDGE;
    }
}

