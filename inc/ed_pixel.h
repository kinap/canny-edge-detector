
#ifndef _ED_PIXEL_H_
#define _ED_PIXEL_H_

struct pixel_t {
    uint16_t red;
    uint16_t green;
    uint16_t blue;
    uint16_t alpha;

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
