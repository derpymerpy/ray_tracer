#include "imports/rtweekend.h"
#include "imports/hittable.h"
#include "imports/hittable_list.h"
#include "imports/sphere.h"
#include "imports/camera.h"
#include "imports/material.h"

using namespace std;



int main() { 
    //WORLD
    hittable_list world;
    shared_ptr<material> material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    //shared_ptr<material> material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    //shared_ptr<material> material_left = make_shared<metal>(color(0.8, 0.8, 0.8));
    //shared_ptr<material> material_right = make_shared<metal>(color(0.8, 0.6, 0.2));

    world.add(make_shared<sphere>(point3( 0.0, 0, -1.0), 100.0, material_ground));
    //world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.2),   0.5, material_center));
    //world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    //world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));

    //CAMERA 
    camera cam;
    cam.aspect_ratio = 16.0/9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;


    cam.render(world);
}