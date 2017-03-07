
#ifndef _ED_PIXEL_H_
#define _ED_PIXEL_H_

#include <stdint.h>

#define pixel_channel_t uint16_t

struct pixel_t {
    pixel_channel_t red;
    pixel_channel_t green;
    pixel_channel_t blue;

    /* Overloaded operators for comparing pixels */
    // TODO re-evaluate only using red channel here
    bool operator==(const pixel_t &rhs) {
        return (red == rhs.red);
    }

    bool operator!=(const pixel_t &rhs) {
        return (red != rhs.red);
    }

    bool operator>(const pixel_t &rhs) {
        return (red > rhs.red);
    }

    bool operator>=(const pixel_t &rhs) {
        return (red >= rhs.red);
    }

    bool operator<(const pixel_t &rhs) {
        return (red < rhs.red);
    }

    bool operator<=(const pixel_t &rhs) {
        return (red <= rhs.red);
    }
};

struct pixel_t_signed {
    int16_t red;
    int16_t green;
    int16_t blue;
};

typedef int16_t pixel_channel_t_signed;

#endif // _ED_PIXEL_H_
