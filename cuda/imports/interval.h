#ifndef INTERVAL_H
#define INTERVAL_H 

#include "rtweekend.h"

class interval{
    public: 
        __device__ float min;
        __device__ float max;

        __device__ interval(): min(+infinity), max(-infinity) {}

        __device__ interval(float min, float max): min(min), max(max) {}

        __device__ float size() const {
            return max - min;
        }

        __device__ bool contains(float d){
            return min <= d && d <= max;
        }
        
        __device__ bool surrounds(float d){
            return min < d && d < max;
        }

        __device__ float clamp(float d){
            if (d < min) return min;
            if (d > max) return max;
            return d;
        }

        static const interval empty;
        static const interval universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif