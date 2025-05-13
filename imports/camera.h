#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"

class camera{ 
    public: 
        double aspect_ratio = 16.0/9.0;
        int image_width = 800;

        int samples_per_pixel = 10;
        int max_depth = 10;

        double defocus_angle = 0.0; //angle at apex of defocus cone. controls radius of defocus disk
        double focus_distance = 10.0; //distance from camera center to focus plane

        //vertical field of view
        double vfov = 90;

        void render(const hittable& world){
            initialize();

            //header for P3 format
            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
            for(int j = 0; j < image_height; j++){
                //progress indicator 
                std::clog << "\rScanlines remaining: " << image_height - j << " " << std::flush;

                for(int i = 0; i < image_width; i++){
                    //center of pixel
                    point3 pixel_center = pixel00_loc + (i * d_u) + (j * d_v);
                    color pixel_color = color(0, 0, 0);

                    for(int sample = 0; sample < samples_per_pixel; sample++){
                        ray r = get_sample_ray(pixel_center, 1.0);
                        pixel_color += ray_color(r, world, max_depth);
                    }
                    
                    write_color(std::cout, pixel_color/samples_per_pixel);
                }
            }
            std::clog<< "\rDONE.                            \n"<<std::flush;
        }

        bool set_camera_center(point3 camera_center_loc, point3 image_center_loc, vec3 up_dir){
            vec3 view_angle_temp = unit_vector(image_center_loc - camera_center_loc);
            if (cross(view_angle_temp, up_dir).length_squared() == 0) {
                //invalid up 
                return false;
            }
            camera_center = camera_center_loc;
            view_angle = view_angle_temp;

            vec3 up_proj = view_angle * dot(up_dir, view_angle)/view_angle.length_squared();
            up = unit_vector(up_dir - up_proj);

            return true;
        }

    private: 
        int image_height;
        double viewport_h; //height of viewport
        double viewport_w; //width of viewport

        point3 camera_center = point3(0, 0, 0);
        vec3 view_angle = point3(0, 0, -1); // unit vector in the direction the camera is pointing
        vec3 up = vec3(0, 1, 0); // the "up" of the frame (orientation)
        
        //vectors along viewport 
        vec3 viewport_u;
        vec3 viewport_v;

        //vectors to move between pixels
        vec3 d_u;
        vec3 d_v;

        //unit direction vectors 
        vec3 unit_u, unit_v;

        //first pixel
        point3 pixel00_loc;

        vec3 defocus_disk_u;
        vec3 defocus_disk_v;


        void initialize() {
            image_height = (int) image_width/aspect_ratio;
            //set image_height to 1 if less than 1 
            image_height = image_height > 1 ? image_height : 1;

            double vfov_rad = degrees_to_radians(vfov);
            viewport_h = focus_distance * 2 * std::tan(vfov_rad/2);
            viewport_w = viewport_h * ((double)(image_width)/image_height);

            viewport_u = viewport_w * cross(view_angle, up); //right 
            viewport_v = viewport_h * -1 * up; //down 
            
            unit_u = unit_vector(viewport_u);
            unit_v = unit_vector(viewport_v);

            d_u = viewport_u / image_width;
            d_v = viewport_v / image_height;

            //upper left corner of viewport 
            point3 viewport_ul = camera_center + focus_distance * view_angle - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_ul + (d_u + d_v)/2;
            
            //basis vectors for defocus offset
            double defocus_radius = focus_distance * std::tan(degrees_to_radians(defocus_angle/2));
            defocus_disk_u = unit_u * defocus_radius;
            defocus_disk_v = unit_v * defocus_radius;
        }

        color ray_color(const ray& r, const hittable& world, int depth){
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

        vec3 pixel_sample_offset(double s) const{
            return vec3(random_double(-s/2, s/2), random_double(-s/2, s/2), 0);
        }

        vec3 defocus_disk_offset() const{
            if (defocus_angle > 0){
                vec3 offset = random_in_unit_disk();    
                //use u, v as unit direction vectors
                return camera_center + (offset.x() * defocus_disk_u) + (offset.y() * defocus_disk_v);
            }
            return camera_center;
        }

        ray get_sample_ray(point3 pixel_center, double s) const {
            vec3 offset = pixel_sample_offset(s);
            point3 ray_origin = defocus_disk_offset();
            //offset for antialiasing
            vec3 ray_direction = pixel_center 
                                 + offset.x() * d_u 
                                 + offset.y() * d_v
                                 - camera_center;
            ray r(ray_origin, ray_direction);
            return r;
        }
};

#endif