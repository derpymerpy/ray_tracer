#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <fstream>
#include <curand_kernel.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )



const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " 
                  <<file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

inline float degrees_to_radians(float degrees) {
    return degrees*pi/180.0f;
}

inline float radians_to_degrees(float rad) {
    return rad*180.0f/pi;
}

__device__ inline float random_float(curandState *thread_state) {
    return (curand_uniform(thread_state));
}

__device__ inline float random_float(float min, float max, curandState *local_state) {
    return min + random_float(local_state) * (max-min);
}

__host__ __device__ inline float float_min(float x, float y){
    if (x < y) return x;
    return y;
}

__host__ __device__ inline float float_max(float x, float y){
    if (x < y) return y;
    return x;
}



#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif