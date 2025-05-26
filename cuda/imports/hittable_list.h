#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rtweekend.h"
#include "hittable.h"
#include <vector>

//extends hittable so that the same hit function can be used for indivdual
//hittable objects and hittable_lists
class hittable_list : public hittable {
    public: 
        std::vector<hittable*> objects;

        __device__ hittable_list() {}
        __device__ hittable_list(hittable* obj){
            add(obj);
        }

        __device__ hittable_list(hittable** obj, int n){
            for(int i = 0; i < n; i++){
                add(*(obj+i));
            }
        }

        __device__ void clear(){
            objects.clear();
        }

        __device__ void add(hittable* obj){
            objects.push_back(obj);
        }

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override{
            //track the nearest hit
            hit_record rec_temp;
            bool has_hit = false;
            double t_temp = ray_t.max;

            for(hittable *obj : objects){
                //has a closer hit 
                if(obj->hit(r, interval(ray_t.min, t_temp), rec_temp)){
                    has_hit = true;
                    t_temp = rec_temp.t;
                    //store the hit record
                    //rec was passed as reference, this step works
                    rec = rec_temp;
                }
            }

            return has_hit;
        }
};

#endif