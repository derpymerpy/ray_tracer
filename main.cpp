#include "imports/rtweekend.h"
#include "imports/hittable.h"
#include "imports/hittable_list.h"
#include "imports/sphere.h"
#include "imports/camera.h"
#include "imports/material.h"
#include "scene.cpp"


using namespace std;



int main() { 
    
    hittable_list world = get_scene();

    //CAMERA 
    camera cam;
    cam.aspect_ratio = 16.0/9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 50;
    cam.max_depth = 50;
    cam.vfov = 20;

    if(!cam.set_camera_center(point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0))){
         std::clog << "INVALID UP\n" << std::flush;
         return -1;
    }

    cam.render(world);
}