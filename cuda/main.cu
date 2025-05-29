#include <fstream>
#include "imports/rtweekend.h"
#include "imports/hittable.h"
#include "imports/hittable_list.h"
#include "imports/sphere.h"
#include "imports/material.h"
#include "imports/camera.h"
#include "imports/renderer.h"

using namespace std;


int main() { 
    //CAMERA 
    camera cam;
    cam.aspect_ratio = 16.0/9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    
    cam.vfov = 20;
    // cam.defocus_angle = 10.0;
    // cam.focus_distance = 3.4;

    if(!cam.set_camera_center(point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0))){
         std::clog << "INVALID UP\n" << std::flush;
         return -1;
    }

    hittable_list world;
    cam.initialize_camera();

    render(&world, &cam);
}