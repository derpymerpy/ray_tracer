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
    //get output destination 
    cout<<"output file: ";
    string output_file;
    cin>>output_file;
    ofstream fout("renders/" + output_file + ".ppm");
    
    //CAMERA 
    //h_cam
    camera cam;
    cam.aspect_ratio = 16.0f/9.0f;
    cam.image_width = 800;
    cam.samples_per_pixel = 50;
    cam.max_depth = 50;
    
    cam.vfov = 20;
    cam.defocus_angle = 0.6f;
    cam.focus_distance = 10.0f;

    if(!cam.set_camera_center(point3(13, 2, 3), point3(0, 0, 0), vec3(0, 1, 0))){
         std::clog << "INVALID UP\n" << std::flush;
         return -1;
    }

    render(&cam, fout);
}