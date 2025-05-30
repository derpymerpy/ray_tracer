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

__global__ void initiate_camera(camera** h_cam, camera** d_cam){

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

__global__ void render_world(int *fb, camera *cam, hittable_list **d_world, curandState *d_rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //ensure not out of bounds
    if (i >= cam->image_w() || j >= cam->image_h()) return;
    int pixel_index = j * cam->image_w() + i;

    point3 pixel_center = cam->get_pixel00_loc() + i*cam->get_du() + j*cam->get_dv();
    vec3 ray_direction = pixel_center - cam->get_camera_center();
    ray r(cam->get_camera_center(), ray_direction);

    color pixel_color = cam->temp_color(r, d_world);

    write_color(fb, pixel_index, pixel_color);
}

//assume camera is already initialized
__host__ void render(const hittable *world, const camera* h_cam, ostream& destination){
    


    //block dimensions
    dim3 blocks((h_cam->image_w())/(h_cam->tx) + 1, (h_cam->image_h())/(h_cam->ty) + 1); 
    dim3 threads(h_cam->tx, h_cam->ty);

    //frame buffer to store pixel values
    int *fb;
    checkCudaErrors(cudaMallocManaged((void**) &fb, h_cam->fb_n() * sizeof(int)));

    //random state for cuda
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, h_cam->n_pixels() * sizeof(curandState)));

    //move camera to gpu 
    camera *d_cam;
    checkCudaErrors(cudaMalloc((void**) &d_cam, sizeof(camera)));
    checkCudaErrors(cudaMemcpy(d_cam, h_cam, sizeof(camera), cudaMemcpyHostToDevice)); //shallow copy, should be fine 

    hittable **d_hittable_list;
    int world_size = 2; //hardcoded ground + sphere
    checkCudaErrors(cudaMalloc((void**) &d_hittable_list, world_size * sizeof(hittable*))); 

    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable_list*)));

    initiate_world<<<1, 1>>> (d_world, d_hittable_list);
    initiate_randstate<<<1, 1>>> (h_cam->image_w(), h_cam->image_h(), d_rand_state);
    render_world<<<blocks, threads>>>(fb, d_cam, d_world, d_rand_state);
    
    checkCudaErrors(cudaDeviceSynchronize());

    set<int> printed;
    for(int i = 0; i < h_cam->image_w() * h_cam->image_h(); i++){
        printed.insert(i);
    }

    //print to file
    destination << "P3\n" << h_cam->image_w() << " " << h_cam->image_h() << "\n255\n";
    cout<<"total pixels: "<< printed.size()<<endl;
    for (int j = 0; j < h_cam->image_h(); j++){
        for(int i = 0; i < h_cam->image_w(); i++){
            int pixel_index = 3 * (j * h_cam->image_w() + i);
            printed.erase(pixel_index/3);
            destination<<fb[pixel_index + 0] << " "
                <<fb[pixel_index + 1] << " "
                <<fb[pixel_index + 2] <<"\n"; 
        }
    }
    cout<<"missing pixels: "<< printed.size()<<endl;
    free_world<<<1,1>>>(d_world, d_hittable_list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_hittable_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));
}
#endif