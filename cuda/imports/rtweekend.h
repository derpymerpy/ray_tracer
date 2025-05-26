#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

using std::make_shared;
using std::shared_ptr;

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;


inline float degrees_to_radians(float degrees) {
    return degrees*pi/180;
}

inline float radians_to_degrees(float rad) {
    return rad*180/pi;
}

inline float random_float() {
    return ((float) std::rand())/(RAND_MAX+1.0);
}

inline float random_float(float min, float max) {
    return min + random_float() * (max-min);
}

__device__ inline float float_min(float x, float y){
    if (x < y) return x;
    return y;
}

__device__ inline float float_max(float x, float y){
    if (x < y) return y;
    return x;
}



#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif