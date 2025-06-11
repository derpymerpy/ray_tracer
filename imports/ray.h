#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
    public:
        __device__ ray() {}
        
        __device__ ray(const point3& origin, const vec3& direction) 
            : orig(origin), dir(direction), tm(0.0) {}

        __device__ ray(const point3& origin, const vec3& direction, float time) 
            : orig(origin), dir(direction), tm(time) {}

        __device__ const point3& origin() const {
            return orig;
        }
        
        __device__ const vec3& direction() const{
            return dir;
        }

        //returns position of ray at time t
        __device__ point3 at(float t) const{
            return orig + t*dir;
        }

        __device__ float time() const{
            return tm;
        }

    

    private: 
        point3 orig;
        vec3 dir;
        float tm;
};

#endif