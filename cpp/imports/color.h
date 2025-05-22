#ifndef COLOR_H
#define COLOR_H

#include "rtweekend.h"
#include "vec3.h"
#include "interval.h"

using color = vec3;

inline double linear_to_gamma(double d){
    //gamma 2, so -2 exponent
    if(d > 0){
        return sqrt(d);
    }
    return 0;
}

inline int double_to_color(double d, interval i) {
    return (int) (i.clamp(d) * 255.999);
}

//writes individual pixel to the stream out
void write_color(std::ostream& out, const color& c){
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
    int r_int = double_to_color(r, color_int);
    int g_int = double_to_color(g, color_int);
    int b_int = double_to_color(b, color_int);

    out<<r_int<<" "<<g_int<<" "<<b_int<<"\n";
};

#endif