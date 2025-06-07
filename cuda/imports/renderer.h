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


__global__ void initiate_world(hittable_list **d_world, hittable **d_list, material **mat_list, curandState* d_rand_state){
    //only one block
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i != 0 || j != 0) return; //ensure only run once

    //ground
    *(mat_list+0) = new lambertian(color(0.5f, 0.5f, 0.5f));
    *(d_list + 0) = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, *(mat_list+0));

    *(mat_list+1) = new dielectric(color(1.0f, 1.0f, 1.0f), 1.5f);
    *(d_list + 1) = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, *(mat_list+1));

    *(mat_list+2) = new lambertian(color(0.4f, 0.2f, 0.1f));
    *(d_list + 2) = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, *(mat_list+2));

    *(mat_list+3) = new metal(color(0.7f, 0.6f, 0.5f), 0.0);
    *(d_list + 3) = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, *(mat_list+3));

    curandState local_rand_state = d_rand_state[0];

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            
            int obj_index = 22*(a+11) + b + 11 + 4;

            float choose_mat = random_float(&local_rand_state);
            point3 center(a + 0.9f*random_float(&local_rand_state), 0.2f, b + 0.9f*random_float(&local_rand_state));

            while ((center - point3(4.0f, 0.2f, 0.0f)).length() <= 0.9f){
                center = point3(a + 0.9f*random_float(&local_rand_state), 0.2f, b + 0.9f*random_float(&local_rand_state));
            }

            if (choose_mat < 0.8f) {
                // diffuse
                auto albedo = random_vec(0, 1, &local_rand_state) * random_vec(0, 1, &local_rand_state);
                *(mat_list + obj_index) = new lambertian(albedo);
                *(d_list + obj_index) = new sphere(center, 0.2, *(mat_list + obj_index));
            } 
            else if (choose_mat < 0.95f) {
                // metal
                auto albedo = random_vec(0.5, 1, &local_rand_state);
                auto fuzz = random_float(0, 0.5, &local_rand_state);
                *(mat_list + obj_index) = new metal(albedo, fuzz);
                *(d_list + obj_index) = new sphere(center, 0.2, *(mat_list + obj_index));
            } 
            else {
                // glass
                *(mat_list + obj_index) = new dielectric(color(1, 1, 1), 1.5);
                *(d_list + obj_index) = new sphere(center, 0.2, *(mat_list + obj_index));
            }
        }
    }


    *d_world = new hittable_list(d_list, 22 * 22 + 4);  
}

__global__ void free_world(hittable_list **d_world, int world_size, hittable **d_list, material **mat_list, int mat_cnt){
    //only one block
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i != 0 || j != 0) return; //ensure only run once

    for (int i = 0; i < world_size; i++){
        delete(*(d_list+i));
    }

    delete *(d_world);
    
    for (int i =0; i < mat_cnt; i++){
        delete(*(mat_list+i));
    }
}



__global__ void render_world(int *fb, camera *cam, hittable_list **d_world, curandState *d_rand_state, int num_samples){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //ensure not out of bounds
    if (i >= cam->image_w() || j >= cam->image_h()) return;
    int pixel_index = j * cam->image_w() + i;

    curandState local_rand_state = d_rand_state[pixel_index];

    color pixel_color = color(0.0f, 0.0f, 0.0f);
    
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
    
    int world_size = 22 * 22 + 4;
    int mat_cnt = 22 * 22 + 4;

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
    checkCudaErrors(cudaMalloc((void**) &d_hittable_list, world_size * sizeof(hittable*))); 

    hittable_list **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable_list*)));

    material **mat;
    checkCudaErrors(cudaMalloc((void**) &mat, mat_cnt*sizeof(material*)));

    initiate_randstate<<<blocks, threads>>> (h_cam->image_w(), h_cam->image_h(), d_rand_state);
    clog<<"randstate initiated\n"<<flush;

    initiate_world<<<1, 1>>> (d_world, d_hittable_list, mat, d_rand_state);
    clog<<"world initiated\n"<<flush;
        
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
    free_world<<<1,1>>>(d_world, world_size, d_hittable_list, mat, mat_cnt);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_hittable_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(mat));
}
#endif