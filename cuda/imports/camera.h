#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h" //temporary
#include "material.h"

#include <set>
using namespace std;


__global__ void initiate_randstate(int image_width, int image_height, curandState *rand_state) {
    int i = threadIdx.x+ blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i >= image_width || j >= image_height) return;
    int pixel_index = j * image_width + i;

    curand_init(0, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void initiate_world(hittable_list **d_world, hittable **d_list){
    //only one block
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i != 0 || j != 0) return; //ensure only run once
    *(d_list+0) = new sphere(vec3(0,0,-1), 0.5);
    *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *d_world = new hittable_list(d_list, 2);  
}

__global__ void free_world(hittable_list **d_world, hittable **d_list){
    //only one block
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i != 0 || j != 0) return; //ensure only run once
    delete *(d_list + 0);
    delete *(d_list + 1);
    delete *(d_world);
}

__global__ void render_world(int *fb, int image_width, int image_height, point3 pixel00_loc, vec3 du, vec3 dv, point3 camera_center, hittable_list **d_world, curandState *d_rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //ensure not out of bounds
    if (i >= image_width || j >= image_height) return;
    int pixel_index = j * image_width + i;

    point3 pixel_center = pixel00_loc + i*du + j*dv;
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);

    color pixel_color = temp_color(r, d_world);

    write_color(fb, pixel_index, pixel_color);
}

__device__ vec3 temp_color(const ray& r, hittable_list **world) {
    hit_record rec;
    if ((*world)->hit(r, interval(0.0, __FLT_MAX__), rec)) { 
        return 0.5f * (rec.norm + vec3(0, 0, 0));
    }
    else{
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}


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

        __device__ int image_h() const{
            return image_height;
        }

        __device__ int image_w() const{
            return image_width;
        }

        __device__ int n_pixels() const{
            return num_pixels;
        }

        __device__ int fb_n() const{
            return fb_size;
        }

        __device__ const point3& get_camera_center() const{
            return camera_center;
        }

        __device__ const vec3& get_du() const{
            return d_u;
        }

        __device__ const vec3& get_dv() const{
            return d_v;
        }

        __device__ const vec3& get_u() const{
            return unit_u;
        }

        __device__ const vec3& get_v() const{
            return unit_v;
        }

        __device__ const vec3& get_w() const{
            return unit_w;
        }

        __device__ const point3& get_pixel00_loc() const{
            return pixel00_loc;
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

        __device__ color ray_color(const ray& r, const hittable& world, int depth){
            if (depth < 0) return color(0, 0, 0);

            hit_record rec;
            //check for hits
            bool hit_any = world.hit(r, interval(0.001, infinity), rec);
            if(hit_any){
                ray scattered;
                color attentuation;
                if(rec.mat -> scatter(r, rec, attentuation, scattered)){
                    return attentuation * ray_color(scattered, world, depth-1);
                }
                return color(0, 0, 0);
            }
            
            //background
            vec3 unit_direction = unit_vector(r.direction());
            //ranges between 0.0-1.0
            auto a = 0.5 * (unit_direction.y() + 1.0);
            return (1.0-a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        }

        __device__ vec3 pixel_sample_offset(float s) const{
            return vec3(random_float(-s/2, s/2), random_float(-s/2, s/2), 0);
        }

        __device__ vec3 defocus_disk_offset() const{
            if (defocus_angle > 0){
                vec3 offset = random_in_unit_disk();    
                //use u, v as unit direction vectors
                return camera_center + (offset.x() * defocus_disk_u) + (offset.y() * defocus_disk_v);
            }
            return camera_center;
        }

        __device__ ray get_sample_ray(point3 pixel_center, float s) const {
            vec3 offset = pixel_sample_offset(s);
            point3 ray_origin = defocus_disk_offset();
            //offset for antialiasing
            vec3 ray_direction = pixel_center 
                                 + offset.x() * d_u 
                                 + offset.y() * d_v
                                 - ray_origin;
            ray r(ray_origin, ray_direction);
            return r;
        }
};

#endif