
#ifndef _ED_PIXEL_H_
#define _ED_PIXEL_H_

struct pixel_t {
    uint8_t red;
    uint8_t green;
    uint8_t blue;

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
