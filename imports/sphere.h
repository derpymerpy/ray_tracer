#ifndef SPHERE_H
#define SPHERE_H

#include "rtweekend.h"
#include "hittable.h"
#include "material.h"

class sphere : public hittable{
    public: 

        __device__ sphere(const point3& center, float radius, material *mat)
            : center(ray(center, vec3(0, 0, 0))), radius(float_max(0.0001f, radius)), mat(mat) {}

        __device__ sphere(const point3& center_start, const point3& center_end, float radius, material *mat)
            : center(ray(center_start, center_end - center_start)), radius(float_max(0.0001f, radius)), mat(mat) {}

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            
            point3 current_center = center.at(r.time());

            vec3 cq = current_center - r.origin();
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
            vec3 outward_norm = (rec.p - current_center)/radius;
            //set norm to be opposite of the ray
            rec.set_face_normal(r, outward_norm);
            rec.mat = mat;

            return true;
        }

    private: 
        ray center;
        float radius;
        material* mat;
};

#endif