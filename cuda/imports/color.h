#ifndef COLOR_H
#define COLOR_H

#include "rtweekend.h"
#include "vec3.h"
#include "interval.h"

using color = vec3;

__device__ inline float linear_to_gamma(float d){
    //gamma 2, so -2 exponent
    if(d > 0){
        return sqrt(d);
    }
    return 0;
}

__device__ inline int float_to_color(float d, interval i) {
    return (int) (i.clamp(d) * (float) 255.999);
}

//writes individual pixel to the stream out
__device__ void write_color(int *fb, int pixel_index, const color& c){
    //using x, y, z to respect interface
    interval color_int = interval(0, 0.999);
    auto r = c.x();
    auto g = c.y();
    auto b = c.z();

    //adjust to gamma 
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    //convert to int vals between 0, 255
    int r_int = float_to_color(r, color_int);
    int g_int = float_to_color(g, color_int);
    int b_int = float_to_color(b, color_int);

    fb[pixel_index + 0] = r_int;
    fb[pixel_index + 1] = g_int;
    fb[pixel_index + 2] = b_int;
};

#endif