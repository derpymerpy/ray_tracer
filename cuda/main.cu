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
    cam.aspect_ratio = 16.0/9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    
    cam.vfov = 90;
    // cam.defocus_angle = 10.0;
    // cam.focus_distance = 3.4;

    if(!cam.set_camera_center(point3(0, 0, 0), point3(0, 0, -1), vec3(0, 1, 0))){
         std::clog << "INVALID UP\n" << std::flush;
         return -1;
    }

    cam.initialize_camera();
    //null is TEMPORARY. currently not being used
    render(nullptr, &cam, fout);
}