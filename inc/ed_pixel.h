
#ifndef _ED_PIXEL_H_
#define _ED_PIXEL_H_

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

#endif // _ED_PIXEL_H_
