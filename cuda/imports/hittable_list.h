#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rtweekend.h"
#include "hittable.h"
#include <vector>

//extends hittable so that the same hit function can be used for indivdual
//hittable objects and hittable_lists
class hittable_list : public hittable {
    public: 
        __device__ hittable_list() {}

        __device__ hittable_list(hittable** obj, int n): objects{obj}, cnt{n} {}

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override{
            //track the nearest hit
            hit_record rec_temp;
            bool has_hit = false;
            double t_temp = ray_t.max;

            for (int i = 0; i<cnt; i++){
                hittable *obj = *(objects+i);
                if(obj->hit(r, interval(ray_t.min, t_temp), rec_temp)){
                    has_hit = true;
                    t_temp = rec_temp.t;
                    //store the hit record
                    rec = rec_temp;
                }
            }

            return has_hit;
        }
    private:
        int cnt; //number of elements
        hittable **objects; //list of pointers to hittables 

};

#endif