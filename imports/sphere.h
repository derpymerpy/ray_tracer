#ifndef SPHERE_H
#define SPHERE_H


#include "rtweekend.h"
#include "hittable.h"
#include "material.h"

class sphere : public hittable{
    public: 
        sphere(const point3& center, double radius, shared_ptr<material> mat)
            : center(center), radius(std::fmax(0, radius)), mat(mat) {}

        bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            vec3 cq = center - r.origin();
            double a = r.direction().length_squared();
            double c = cq.length_squared() - radius*radius;
            double h = dot(r.direction(), cq);

            double disc = h*h - a*c;
            //no hits
            //std::cout<<a<<" "<<c<<" "<<h<<" "<<disc<<std::endl;
            if(disc < 0){
                return false;
            }

            double sqrt_disc = std::sqrt(disc);
            double root = (h - sqrt_disc)/a;
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
        double radius;
        shared_ptr<material> mat;
};

#endif