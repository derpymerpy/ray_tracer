#ifndef SPHERE_H
#define SPHERE_H

#include "rtweekend.h"
#include "hittable.h"
#include "material.h"

class sphere : public hittable{
    public: 

        __device__ sphere(const point3& center, float radius, material *mat)
            : center(center), radius(float_max(0.0001f, radius)), mat(mat) {}

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            vec3 cq = center - r.origin();
            float a = r.direction().length_squared();
            float c = cq.length_squared() - radius*radius;
            float h = dot(r.direction(), cq);

            float disc = h*h - a*c;
            //no hits
            if(disc < 0 || a < 1e-6f){
                return false;
            }

            float sqrt_disc = sqrtf(disc);
            float root = (h - sqrt_disc)/a;
            if(!ray_t.surrounds(root)){
                root = (h+sqrt_disc)/a;
                if(!ray_t.surrounds(root)) return false;
            }
            //root is the valid root
            rec.t = root;
            rec.p = r.at(root);
            vec3 outward_norm = (rec.p - center)/radius;
            //set norm to be opposite of the ray
            rec.set_face_normal(r, outward_norm);
            rec.mat = mat;

            return true;
        }

    private: 
        point3 center;
        float radius;
        material* mat;
};

#endif