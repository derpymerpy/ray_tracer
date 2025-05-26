#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"

class material;

//class to store hits
class hit_record{
    public:
        float t;
        point3 p;
        vec3 norm; //should be a unit vector
        bool front_face;
        material *mat;

        __device__ void set_face_normal(const ray& r, const vec3& outward_normal){
            //negative if ray and outward norm and opposit -> ray on outside
            front_face = dot(r.direction(), outward_normal) < 0;
            if(front_face) norm = outward_normal;
            else norm = -outward_normal;
        }
};

class hittable{
    public: 
        __device__ virtual ~hittable() = default;

        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0; 
};

#endif