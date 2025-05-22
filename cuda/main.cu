#include <fstream>

#include "imports/rtweekend.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

using namespace std;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " 
                  <<file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(int *fb, int image_width, int image_height, point3 pixel00_loc, vec3 du, vec3 dv, point3 camera_center){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //ensure not out of bounds
    if (i >= image_width || j >= image_height) return;
    int pixel_index = (j * image_width + i)*3;

    point3 pixel_center = pixel00_loc + i*du + j*dv;
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);

    vec3 unit_direction = unit_vector(r.direction());
    float a = ((float) 0.5) * (unit_direction.y() + (float) 1.0);

    color pixel_color = ((float) 1.0 - a)*color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    write_color(fb, pixel_index, color(255,255,255));
}

int main() {
    cout<<"output file: "<<endl;
    string output_file;
    cin>>output_file;
    ofstream fout(output_file);

    int image_width = 256;
    int image_height = 256;

    int num_pixels = image_height * image_width;
    size_t fb_size = num_pixels * 3;

    //========camera stuff============
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    //===============================

    int* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size * sizeof(int)));
    
    int tx = 32;
    int ty = 32;

    dim3 blocks(image_width/tx + 1, image_height/ty + 1); //round up 
    dim3 threads(tx, ty);

    render<<<blocks, threads>>>(fb, image_width, image_height, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);
    //wait for gpu to finish
    checkCudaErrors(cudaDeviceSynchronize());

    fout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++){
        for(int i = 0; i < image_width; i++){
            int pixel_index = 3 * (j * image_width + i);
            fout<<fb[pixel_index + 0] << " "
                <<fb[pixel_index + 1] << " "
                <<fb[pixel_index + 2] <<"\n"; 
        }
    }
    checkCudaErrors(cudaFree(fb));
}