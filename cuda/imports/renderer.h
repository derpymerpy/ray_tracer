#ifndef RENDER_H
#define RENDER_H


#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h" //temporary
#include "material.h"
#include "camera.h"


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

//assume camera is already initialized
__host__ void render(const hittable *world, const camera* cam){
    cout<<"output file: ";
    string output_file;
    cin>>output_file;
    ofstream fout("renders/" + output_file + ".ppm");


    //block dimensions
    dim3 blocks((cam->image_w())/(cam->tx) + 1, (cam->image_h())/(cam->ty) + 1); 
    dim3 threads(cam->tx, cam->ty);

    //frame buffer to store pixel values
    int *fb;
    checkCudaErrors(cudaMallocManaged((void**) &fb, cam->fb_n() * sizeof(int)));

    //random state for cuda
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, cam->n_pixels() * sizeof(curandState)));

    hittable **d_hittable_list;
    int world_size = 2; //hardcoded ground + sphere
    checkCudaErrors(cudaMalloc((void**) &d_hittable_list, world_size * sizeof(hittable*))); 

    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable_list*)));

    initiate_world<<<1, 1>>> (d_world, d_hittable_list);
    initiate_randstate<<<1, 1>>> (cam->image_w(), cam->image_h(), d_rand_state);
    render_world<<<blocks, threads>>>(fb, cam->image_w(), cam->image_h(), cam->get_pixel00_loc(),
                                     cam->get_du(), cam->get_dv(), cam->get_camera_center(), d_world, d_rand_state);
    
    checkCudaErrors(cudaDeviceSynchronize());

    set<int> printed;
    for(int i = 0; i < cam->image_w() * cam->image_h(); i++){
        printed.insert(i);
    }

    //print to file
    fout << "P3\n" << cam->image_w() << " " << cam->image_h() << "\n255\n";
    cout<<"total pixels: "<< printed.size()<<endl;
    for (int j = 0; j < cam->image_h(); j++){
        for(int i = 0; i < cam->image_w(); i++){
            int pixel_index = 3 * (j * cam->image_w() + i);
            printed.erase(pixel_index/3);
            fout<<fb[pixel_index + 0] << " "
                <<fb[pixel_index + 1] << " "
                <<fb[pixel_index + 2] <<"\n"; 
        }
    }
    cout<<"missing pixels: "<< printed.size()<<endl;
    free_world<<<1,1>>>(d_world, d_hittable_list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_hittable_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
}
#endif