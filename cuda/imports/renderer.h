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


__global__ void initiate_world(hittable_list **d_world, hittable **d_list, material **mat){
    //only one block
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i != 0 || j != 0) return; //ensure only run once

    //figure out a better way to configure this later
    *(mat+0) = new lambertian (color(0.8, 0.8, 0.0)); //ground material
    *(mat+1) = new lambertian (color(0.1, 0.2, 0.5)); //center
    *(mat+2) = new metal (color(0.8, 0.8, 0.8), 0.3); //left
    *(mat+3) = new metal (color(0.8, 0.6, 0.2), 1); //right
    
    *(d_list+0) = new sphere(point3(0, -100.5, -1), 100, *(mat+0));
    *(d_list+1) = new sphere(point3(0, 0, -1.2), 0.5, *(mat+1));
    *(d_list+2) = new sphere(point3(-1.0, 0, -1), 0.5, *(mat+2));
    *(d_list+3) = new sphere(point3(1.0, 0, -1), 0.5, *(mat+3));

    *d_world = new hittable_list(d_list, 4);  
}

__global__ void free_world(hittable_list **d_world, hittable **d_list, material **mat){
    //only one block
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i != 0 || j != 0) return; //ensure only run once

    delete *(d_list + 0);
    delete *(d_list + 1);
    delete *(d_list + 2);
    delete *(d_list + 3);

    delete *(d_world);
    
    delete *(mat+0);
    delete *(mat+1);
    delete *(mat+2);
    delete *(mat+3);
}



__global__ void render_world(int *fb, camera *cam, hittable_list **d_world, curandState *d_rand_state, int num_samples){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //ensure not out of bounds
    if (i >= cam->image_w() || j >= cam->image_h()) return;
    int pixel_index = j * cam->image_w() + i;

    curandState local_rand_state = d_rand_state[pixel_index];

    color pixel_color = color(0, 0, 0);
    
    for(int sample = 0; sample < num_samples; sample++){
        point3 pixel_center = cam->get_pixel00_loc() + i*cam->get_du() + j*cam->get_dv();
        ray r = cam->get_sample_ray(pixel_center, 1.0, &local_rand_state);
        pixel_color += cam->ray_color(r, **d_world, &local_rand_state);
        // pixel_color += cam->temp_color(r, d_world);
    }
    pixel_color = pixel_color/num_samples;

    write_color(fb, pixel_index, pixel_color);
}


__host__ void render(camera* h_cam, ostream& destination){
    
    h_cam -> initialize_camera();

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
    checkCudaErrors(cudaMallocManaged((void**) &d_cam, sizeof(camera)));
    h_cam->host_to_shared(d_cam);

    hittable **d_hittable_list;
    int world_size = 4; //hardcoded ground + spheres
    checkCudaErrors(cudaMalloc((void**) &d_hittable_list, world_size * sizeof(hittable*))); 

    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable_list*)));

    material **mat;
    checkCudaErrors(cudaMalloc((void**) &mat, 4*sizeof(material*)));
    initiate_world<<<1, 1>>> (d_world, d_hittable_list, mat);
    clog<<"world initiated\n"<<flush;
    initiate_randstate<<<blocks, threads>>> (h_cam->image_w(), h_cam->image_h(), d_rand_state);
    clog<<"randstate initiated\n"<<flush;
    render_world<<<blocks, threads>>>(fb, d_cam, d_world, d_rand_state, h_cam->samples_per_pixel);
    
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
    free_world<<<1,1>>>(d_world, d_hittable_list, mat);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_hittable_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));
}
#endif