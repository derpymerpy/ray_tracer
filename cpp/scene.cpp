#include "imports/rtweekend.h"
#include "imports/hittable.h"
#include "imports/hittable_list.h"
#include "imports/sphere.h"
#include "imports/camera.h"
#include "imports/material.h"

using namespace std;

hittable_list get_scene(){
    //WORLD
    hittable_list world;
    shared_ptr<material> material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    shared_ptr<material> material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    shared_ptr<material> material_left = make_shared<dielectric>(color(1.0, 1.0, 1.0), 1.5);
    shared_ptr<material> material_bubble = make_shared<dielectric>(color(1.0, 1.0, 1.0), 1.0/1.5);
    shared_ptr<material> material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.2);

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.2),   0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.4, material_bubble));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));

    return world;
}