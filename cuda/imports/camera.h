#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h" //temporary
#include "material.h"

#include <set>
using namespace std;



class camera{ 
    public: 
        float aspect_ratio = 16.0/9.0;
        int image_width = 800;

        int samples_per_pixel = 10;
        int max_depth = 10;

        float defocus_angle = 0.0; //angle at apex of defocus cone. controls radius of defocus disk
        float focus_distance = 10.0; //distance from camera center to focus plane

        //vertical field of view
        float vfov = 90;

        // cuda thread dimensions 
        int tx = 8;
        int ty = 8;


        __device__ vec3 temp_color(const ray& r, hittable_list **world) {
            hit_record rec;
            if ((*world)->hit(r, interval(0.0, __FLT_MAX__), rec)) { 
                return 0.5f * (rec.norm + vec3(1.0, 1.0, 1.0));
            }
            else{
                vec3 unit_direction = unit_vector(r.direction());
                float t = 0.5f * (unit_direction.y() + 1.0f);
                return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            }
        }

        __host__ void initialize_camera() {
            image_height = (int) image_width/aspect_ratio;
            //set image_height to 1 if less than 1 
            image_height = image_height > 1 ? image_height : 1;

            num_pixels = image_height * image_width;
            fb_size = 3 * num_pixels;

            float vfov_rad = degrees_to_radians(vfov);

            //viewport placed at focus plane 
            float viewport_height = focus_distance * 2 * std::tan(vfov_rad/2);
            float viewport_width = viewport_height * ((float)(image_width)/image_height);

            //setting unit vectors. v and w are set by set_camera_center
            unit_u = cross(unit_w, unit_v);

            vec3 viewport_u = viewport_width * unit_u; //right 
            vec3 viewport_v = viewport_height * -1 * unit_v; //down 

            //vectors to move between pixels
            d_u = viewport_u / image_width;
            d_v = viewport_v / image_height;

            //upper left corner of viewport 
            point3 viewport_ul = camera_center + focus_distance * unit_w - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_ul + (d_u + d_v)/2;
            
            //basis vectors for defocus offset
            float defocus_radius = focus_distance * std::tan(degrees_to_radians(defocus_angle/2));
            defocus_disk_u = unit_u * defocus_radius;
            defocus_disk_v = unit_v * defocus_radius;
        }

        //move to renderer

        __host__ bool set_camera_center(point3 camera_center_loc, point3 image_center_loc, vec3 up_dir){
            vec3 w_temp = unit_vector(image_center_loc - camera_center_loc);
            if (cross(w_temp, up_dir).length_squared() == 0) {
                //invalid up 
                return false;
            }
            camera_center = camera_center_loc;
            unit_w = w_temp;

            vec3 up_proj = unit_w * dot(up_dir, unit_w);
            unit_v = unit_vector(up_dir - up_proj);

            return true;
        }

        __host__ __device__ int image_h() const{
            return image_height;
        }

        __host__ __device__ int image_w() const{
            return image_width;
        }

        __host__ __device__ int n_pixels() const{
            return num_pixels;
        }

        __host__ __device__ int fb_n() const{
            return fb_size;
        }

        __host__ __device__ const point3& get_camera_center() const{
            return camera_center;
        }

        __host__ __device__ const vec3& get_du() const{
            return d_u;
        }

        __host__ __device__ const vec3& get_dv() const{
            return d_v;
        }

        __host__ __device__ const vec3& get_u() const{
            return unit_u;
        }

        __host__ __device__ const vec3& get_v() const{
            return unit_v;
        }

        __host__ __device__ const vec3& get_w() const{
            return unit_w;
        }

        __host__ __device__ const point3& get_pixel00_loc() const{
            return pixel00_loc;
        }

        __device__ vec3 pixel_sample_offset(float s, curandState *local_state) const{
            return vec3(random_float(-s/2, s/2, local_state), random_float(-s/2, s/2, local_state), 0);
        }

        __device__ color ray_color(const ray& r, const hittable& world, curandState *local_state){

            hit_record rec;

            ray cur_ray = r;
            color total_attentuation = color(1.0, 1.0, 1.0);
            for(int i = 0; i < samples_per_pixel; i++){
                bool hit_any = world.hit(cur_ray, interval(0.001, infinity), rec);
                //check for hits
                if(hit_any){
                    color attentuation = color(1, 1, 1);
                    if(rec.mat -> scatter(cur_ray, rec, attentuation, cur_ray, local_state)){
                        total_attentuation = total_attentuation * attentuation;
                    }
                    else{
                        return color(0, 0, 0);
                    }
                }
                //no hits -> background
                else{
                    vec3 unit_direction = unit_vector(r.direction());
                    //ranges between 0.0-1.0
                    auto a = 0.5 * (unit_direction.y() + 1.0);
                    return total_attentuation * ((1.0-a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0));
                }
            }
            return color(1, 0, 0);
        }

        __device__ ray get_sample_ray(point3 pixel_center, float s, curandState *local_state) const {
            vec3 offset = pixel_sample_offset(s, local_state);
            point3 ray_origin = defocus_disk_offset(local_state);
            // point3 target_pixel = pixel_center;
            //offset for antialiasing
            vec3 ray_direction = pixel_center 
                                 + offset.x() * d_u 
                                 + offset.y() * d_v
                                 - ray_origin;
            ray r(ray_origin, ray_direction);
            return r;
        }

        __host__ void host_to_shared(camera *dest) const{
            dest->aspect_ratio = aspect_ratio;
            dest->image_width = image_width;

            dest->samples_per_pixel = samples_per_pixel;
            dest->max_depth = max_depth;

            dest->defocus_angle = defocus_angle; //angle at apex of defocus cone. controls radius of defocus disk
            dest->focus_distance = focus_distance; //distance from camera center to focus plane

            //vertical field of view
            dest->vfov = vfov;

            // cuda thread dimensions 
            dest->tx = tx;
            dest->ty = ty;

            dest->image_height = image_height;
            dest->num_pixels = num_pixels;
            dest->fb_size = fb_size;

            dest->camera_center = camera_center;

            //vectors to move between pixels
            dest->d_u = d_u;
            dest->d_v = d_v;

            //unit direction vectors 
            dest->unit_u = unit_u;
            dest->unit_v = unit_v;
            dest->unit_w = unit_w;

            //first pixel
            dest->pixel00_loc = pixel00_loc;

            dest->defocus_disk_u = defocus_disk_u;
            dest->defocus_disk_v = defocus_disk_v;
        }

    private: 
        int image_height;
        int num_pixels;
        int fb_size;

        point3 camera_center = point3(0, 0, 0);

        //vectors to move between pixels
        vec3 d_u;
        vec3 d_v;

        //unit direction vectors 
        vec3 unit_u;
        vec3 unit_v = vec3(0, 1, 0);
        vec3 unit_w = vec3(0, 0, -1);

        //first pixel
        point3 pixel00_loc;

        vec3 defocus_disk_u;
        vec3 defocus_disk_v;


        __device__ vec3 defocus_disk_offset(curandState *local_state) const{
            //temporarily not used
            return camera_center;
            if (defocus_angle > 0){
                vec3 offset = random_in_unit_disk(local_state);    
                //use u, v as unit direction vectors
                return camera_center + (offset.x() * defocus_disk_u) + (offset.y() * defocus_disk_v);
            }
            return camera_center;
        }

};

#endif