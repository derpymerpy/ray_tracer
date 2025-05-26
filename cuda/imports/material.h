#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"
#include "ray.h"
#include "hittable.h"

class material {
    public: 
        __device__ virtual ~material() = default;

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, 
                             color& attentuation, ray& scattered) const {
            return false;
        }
    protected: 
        color albedo;
};

class lambertian : public material {
    public: 
        __device__ lambertian(const color& albedo){
            this->albedo = albedo;
        }
        
        __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, 
                     ray& scattered) const override {
            vec3 direction = rec.norm + random_unit_vec();
            if(direction.near_zero()) direction = rec.norm;
            
            scattered = ray(rec.p, direction);
            attenuation = albedo;
            return true;
        }

};

class metal : public material {
    public: 
        __device__ metal(const color& albedo, float fuzz): fuzz(fuzz < 1 ? fuzz : 1) {
            this -> albedo = albedo;
        }

        __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, 
                     ray& scattered) const override {
            vec3 reflection = reflect(r_in.direction(), rec.norm) + fuzz*random_unit_vec(); 
            scattered = ray(rec.p, reflection);
            attenuation = albedo;
            return (dot(scattered.direction(), rec.norm) > 0);
        }

    private: 
        float fuzz;
};

class dielectric : public material {
    public: 
        __device__ dielectric(const color& albedo, float refractive_index): refractive_index(refractive_index) {
            this->albedo = albedo;
        }

        __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, 
                     ray& scattered) const override {
            attenuation = albedo;

            float ri = refractive_index;
            //reciprocal if the ray is going the other way
            if(rec.front_face) ri = 1.0/refractive_index;
        
            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = float_min(dot(-unit_direction, rec.norm), 1.0f);
            float sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

            vec3 result;
            if(ri * sin_theta > 1.0 || reflectance(cos_theta, refractive_index) > random_float()){
                result = reflect(r_in.direction(), rec.norm);
            }
            else{
                result = refract(unit_direction, rec.norm, ri);
            }

            scattered = ray(rec.p, result);
            return true;
        }

    private: 
        float refractive_index;

        __device__ static float reflectance(float cos, float reflective_index) {
            auto r0 = (1 - reflective_index)/(1 + reflective_index);
            r0 = r0 * r0;
            return r0 + (1-r0) * std::pow(1-cos, 5);
        }
};


#endif