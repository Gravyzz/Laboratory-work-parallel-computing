
//компиляция - clang++ -std=c++17 -O3 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp -framework OpenCL main.cpp -o filters
//запуск -  ./filters
//смотрим результат  - open output.png


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <arm_neon.h>

#include <iostream> 

#include <cmath>
#include <algorithm>   

#include <chrono>     
#include <vector>     
#include <string>     



#include <OpenCL/cl.h>  // OpenCL на macOS
#include <fstream>      // для чтения файла filters.cl
#include <sstream>      // для удобного чтения файла в строку





//  Последовательная  реализация
void invert_sequential(unsigned char* img, int width, int height, int channels) {

    int total_pixels = width * height;
    for (int p = 0; p < total_pixels; p++) {
        int idx = p * channels;

        img[idx + 0] = 255 - img[idx + 0];  
        img[idx + 1] = 255 - img[idx + 1]; 
        img[idx + 2] = 255 - img[idx + 2]; 

    }
}


void sobel_sequential(unsigned char* img, int width, int height, int channels) {
    
    int total_bytes = width * height * channels;
    std::vector<unsigned char> output(total_bytes, 0);

    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int gx = 0;
            int gy = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
        
                    int ny = y + dy;
                    int nx = x + dx;

                    int neighbor_idx = (ny * width + nx) * channels;

                    int brightness = (img[neighbor_idx + 0]
                                    + img[neighbor_idx + 1]
                                    + img[neighbor_idx + 2]) / 3;

                    int kx = 0;
                    if (dx == -1) kx = (dy == 0) ? -2 : -1;
                    if (dx ==  1) kx = (dy == 0) ?  2 :  1;

                    int ky = 0;
                    if (dy == -1) ky = (dx == 0) ? -2 : -1;
                    if (dy ==  1) ky = (dx == 0) ?  2 :  1;

                    gx += brightness * kx;
                    gy += brightness * ky;
                }
            }

            int magnitude = (int)std::sqrt((double)(gx * gx + gy * gy));


            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0)   magnitude = 0;

            int out_idx = (y * width + x) * channels;
            output[out_idx + 0] = (unsigned char)magnitude;
            output[out_idx + 1] = (unsigned char)magnitude;
            output[out_idx + 2] = (unsigned char)magnitude;

            if (channels == 4) {
                output[out_idx + 3] = 255;
            }
        }
    }

    std::copy(output.begin(), output.end(), img);
}


void median_sequential(unsigned char* img, int width, int height, int channels) {
    int total_bytes = width * height * channels;

    std::vector<unsigned char> output(img, img + total_bytes);

    unsigned char window_r[9];
    unsigned char window_g[9];
    unsigned char window_b[9];

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int k = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    int idx = (ny * width + nx) * channels;

                    window_r[k] = img[idx + 0];
                    window_g[k] = img[idx + 1];
                    window_b[k] = img[idx + 2];
                    k++;
                }
            }

   
            std::sort(window_r, window_r + 9);
            std::sort(window_g, window_g + 9);
            std::sort(window_b, window_b + 9);


            int out_idx = (y * width + x) * channels;
            output[out_idx + 0] = window_r[4];
            output[out_idx + 1] = window_g[4];
            output[out_idx + 2] = window_b[4];

        }
    }

    std::copy(output.begin(), output.end(), img);
}









//  OpenMP-версии фильтров
void invert_openmp(unsigned char* img, int width, int height, int channels) {
    int total_pixels = width * height;
    
    #pragma omp parallel for
    for (int p = 0; p < total_pixels; p++) {
        int idx = p * channels;
        img[idx + 0] = 255 - img[idx + 0];
        img[idx + 1] = 255 - img[idx + 1];
        img[idx + 2] = 255 - img[idx + 2];
    }
}

void sobel_openmp(unsigned char* img, int width, int height, int channels) {
    int total_bytes = width * height * channels;
    std::vector<unsigned char> output(total_bytes, 0);

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0;
            int gy = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    int neighbor_idx = (ny * width + nx) * channels;

                    int brightness = (img[neighbor_idx + 0]
                                    + img[neighbor_idx + 1]
                                    + img[neighbor_idx + 2]) / 3;

                    int kx = 0;
                    if (dx == -1) kx = (dy == 0) ? -2 : -1;
                    if (dx ==  1) kx = (dy == 0) ?  2 :  1;

                    int ky = 0;
                    if (dy == -1) ky = (dx == 0) ? -2 : -1;
                    if (dy ==  1) ky = (dx == 0) ?  2 :  1;

                    gx += brightness * kx;
                    gy += brightness * ky;
                }
            }

            int magnitude = (int)std::sqrt((double)(gx * gx + gy * gy));
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0)   magnitude = 0;

            int out_idx = (y * width + x) * channels;
            output[out_idx + 0] = (unsigned char)magnitude;
            output[out_idx + 1] = (unsigned char)magnitude;
            output[out_idx + 2] = (unsigned char)magnitude;
            if (channels == 4) {
                output[out_idx + 3] = 255;
            }
        }
    }

    std::copy(output.begin(), output.end(), img);
}

void median_openmp(unsigned char* img, int width, int height, int channels) {
    int total_bytes = width * height * channels;
    std::vector<unsigned char> output(img, img + total_bytes);

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        unsigned char window_r[9];
        unsigned char window_g[9];
        unsigned char window_b[9];

        for (int x = 1; x < width - 1; x++) {
            int k = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    int idx = (ny * width + nx) * channels;

                    window_r[k] = img[idx + 0];
                    window_g[k] = img[idx + 1];
                    window_b[k] = img[idx + 2];
                    k++;
                }
            }

            std::sort(window_r, window_r + 9);
            std::sort(window_g, window_g + 9);
            std::sort(window_b, window_b + 9);

            int out_idx = (y * width + x) * channels;
            output[out_idx + 0] = window_r[4];
            output[out_idx + 1] = window_g[4];
            output[out_idx + 2] = window_b[4];
        }
    }

    std::copy(output.begin(), output.end(), img);
}











//  SIMD

void invert_simd(unsigned char* img, int width, int height, int channels) {
    int total_bytes = width * height * channels;

    uint8x16_t v_mask = vdupq_n_u8(0xFF);

    int i = 0;
    for (; i + 16 <= total_bytes; i += 16) {
        uint8x16_t v = vld1q_u8(img + i);   
        v = veorq_u8(v, v_mask);            
        vst1q_u8(img + i, v);             
    }


    for (; i < total_bytes; i++) {
        img[i] = 255 - img[i];
    }
}


static void rgb_to_gray_plane(const unsigned char* img, int width, int height, int channels,
                              std::vector<unsigned char>& gray) {
    int total_pixels = width * height;
    gray.resize(total_pixels);
    for (int p = 0; p < total_pixels; p++) {
        int idx = p * channels;
        gray[p] = (unsigned char)((img[idx + 0] + img[idx + 1] + img[idx + 2]) / 3);
    }
}

void sobel_simd(unsigned char* img, int width, int height, int channels) {
    int w = width, h = height;

    std::vector<unsigned char> gray;
    rgb_to_gray_plane(img, width, height, channels, gray);

    std::vector<unsigned char> result(width * height, 0);

    for (int y = 1; y < h - 1; y++) {
        int x = 1;

        for (; x + 8 <= w - 1; x += 8) {

            #define LOAD16(off) vreinterpretq_s16_u16( \
                vmovl_u8(vld1_u8(&gray[(off)])))

            int16x8_t a = LOAD16((y - 1) * w + (x - 1));
            int16x8_t b = LOAD16((y - 1) * w + (x));
            int16x8_t c = LOAD16((y - 1) * w + (x + 1));
            int16x8_t d = LOAD16((y) * w + (x - 1));
            int16x8_t f = LOAD16((y) * w + (x + 1));
            int16x8_t g = LOAD16((y + 1) * w + (x - 1));
            int16x8_t hh= LOAD16((y + 1) * w + (x));
            int16x8_t i = LOAD16((y + 1) * w + (x + 1));

            #undef LOAD16

            int16x8_t gx = vsubq_s16(
                vaddq_s16(vaddq_s16(c, i), vshlq_n_s16(f, 1)),
                vaddq_s16(vaddq_s16(a, g), vshlq_n_s16(d, 1)));

            int16x8_t gy = vsubq_s16(
                vaddq_s16(vaddq_s16(a, c), vshlq_n_s16(b, 1)),
                vaddq_s16(vaddq_s16(g, i), vshlq_n_s16(hh, 1)));

            int32x4_t gx_lo = vmovl_s16(vget_low_s16(gx));
            int32x4_t gx_hi = vmovl_s16(vget_high_s16(gx));
            int32x4_t gy_lo = vmovl_s16(vget_low_s16(gy));
            int32x4_t gy_hi = vmovl_s16(vget_high_s16(gy));

            float32x4_t mlo = vsqrtq_f32(vcvtq_f32_s32(
                vaddq_s32(vmulq_s32(gx_lo, gx_lo), vmulq_s32(gy_lo, gy_lo))));
            float32x4_t mhi = vsqrtq_f32(vcvtq_f32_s32(
                vaddq_s32(vmulq_s32(gx_hi, gx_hi), vmulq_s32(gy_hi, gy_hi))));


            int32x4_t ilo = vcvtq_s32_f32(mlo);
            int32x4_t ihi = vcvtq_s32_f32(mhi);
            int16x8_t packed16 = vcombine_s16(vqmovn_s32(ilo), vqmovn_s32(ihi));
            uint8x8_t packed8 = vqmovun_s16(packed16);  

            vst1_u8(&result[y * w + x], packed8);
        }


        for (; x < w - 1; x++) {
            int A = gray[(y-1)*w + (x-1)], B = gray[(y-1)*w + x], C = gray[(y-1)*w + (x+1)];
            int D = gray[ y   *w + (x-1)], F = gray[ y   *w + (x+1)];
            int Gg= gray[(y+1)*w + (x-1)], H = gray[(y+1)*w + x], I = gray[(y+1)*w + (x+1)];
            int gx = (C + 2*F + I) - (A + 2*D + Gg);
            int gy = (A + 2*B + C) - (Gg + 2*H + I);
            int mag = (int)std::sqrt((double)(gx*gx + gy*gy));
            if (mag > 255) mag = 255;
            result[y * w + x] = (unsigned char)mag;
        }
    }


    int total_pixels = width * height;
    for (int p = 0; p < total_pixels; p++) {
        int idx = p * channels;
        img[idx + 0] = result[p];
        img[idx + 1] = result[p];
        img[idx + 2] = result[p];
        if (channels == 4) img[idx + 3] = 255;
    }
}


static uint8x16_t median_vector9(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2,
                                 uint8x16_t v3, uint8x16_t v4, uint8x16_t v5,
                                 uint8x16_t v6, uint8x16_t v7, uint8x16_t v8) {
    #define SWAP_MINMAX(a, b) do {          \
        uint8x16_t mn = vminq_u8((a), (b)); \
        uint8x16_t mx = vmaxq_u8((a), (b)); \
        (a) = mn;                           \
        (b) = mx;                           \
    } while (0)

    SWAP_MINMAX(v0, v1); SWAP_MINMAX(v3, v4); SWAP_MINMAX(v6, v7);
    SWAP_MINMAX(v1, v2); SWAP_MINMAX(v4, v5); SWAP_MINMAX(v7, v8);
    SWAP_MINMAX(v0, v1); SWAP_MINMAX(v3, v4); SWAP_MINMAX(v6, v7);
    SWAP_MINMAX(v0, v3); SWAP_MINMAX(v3, v6); SWAP_MINMAX(v0, v3);
    SWAP_MINMAX(v1, v4); SWAP_MINMAX(v4, v7); SWAP_MINMAX(v1, v4);
    SWAP_MINMAX(v2, v5); SWAP_MINMAX(v5, v8); SWAP_MINMAX(v2, v5);
    SWAP_MINMAX(v1, v3); SWAP_MINMAX(v5, v7); SWAP_MINMAX(v2, v6);
    SWAP_MINMAX(v4, v6); SWAP_MINMAX(v2, v4); SWAP_MINMAX(v2, v3);
    SWAP_MINMAX(v5, v6);

    #undef SWAP_MINMAX
    return v4;
}
 

static unsigned char median9_scalar(unsigned char* w) {
    std::sort(w, w + 9);
    return w[4];
}

void median_simd(unsigned char* img, int width, int height, int channels) {
    int total_bytes = width * height * channels;

    std::vector<unsigned char> out(img, img + total_bytes);
    if (channels != 4) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                for (int c = 0; c < 3; c++) {
                    unsigned char window[9];
                    int k = 0;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                            window[k++] = img[((y + dy) * width + (x + dx)) * channels + c];
                    out[(y * width + x) * channels + c] = median9_scalar(window);
                }
            }
        }
        std::copy(out.begin(), out.end(), img);
        return;
    }

    for (int y = 1; y < height - 1; y++) {
        int x = 1;

        for (; x + 16 <= width - 1; x += 16) {
            uint8x16x4_t output = vld4q_u8(img + (y * width + x) * 4);

            for (int channel = 0; channel < 3; channel++) {
                uint8x16_t w[9];
                int k = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        const unsigned char* ptr = img + ((y + dy) * width + (x + dx)) * 4;
                        uint8x16x4_t rgba = vld4q_u8(ptr);  
                        w[k++] = rgba.val[channel];          
                    }
                }

                output.val[channel] = median_vector9(w[0], w[1], w[2], w[3], w[4],
                                                      w[5], w[6], w[7], w[8]);
            }

            vst4q_u8(out.data() + (y * width + x) * 4, output);
        }

        for (; x < width - 1; x++) {
            for (int c = 0; c < 3; c++) {
                unsigned char window[9];
                int k = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        window[k++] = img[((y + dy) * width + (x + dx)) * 4 + c];
                out[(y * width + x) * 4 + c] = median9_scalar(window);
            }
        }
    }

    std::copy(out.begin(), out.end(), img);
}














//  OpenCLContext — класс-обёртка над инициализацией
struct OpenCLContext {
    cl_platform_id platform = nullptr;
    cl_device_id device   = nullptr;
    cl_context context  = nullptr;
    cl_command_queue queue    = nullptr;
    cl_program program  = nullptr;

    cl_kernel kernel_invert = nullptr;
    cl_kernel kernel_sobel  = nullptr;
    cl_kernel kernel_median = nullptr;

    bool ready = false;

    void init(const char* cl_filename) {
        cl_int err;

        clGetPlatformIDs(1, &platform, nullptr);


        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "OpenCL: GPU не найдено\n";
            return;
        }

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

        queue = clCreateCommandQueue(context, device, 0, &err);

        std::ifstream file(cl_filename);
        if (!file) {
            std::cerr << "OpenCL: не нашёл файл " << cl_filename << "\n";
            return;
        }
        std::stringstream ss;
        ss << file.rdbuf();
        std::string source_str = ss.str();
        const char* source_ptr = source_str.c_str();
        size_t source_len = source_str.size();

        program = clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &err);
        err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);
            std::cerr << "Ошибка компиляции OpenCL kernel:\n" << log.data() << "\n";
            return;
        }

        kernel_invert = clCreateKernel(program, "invert", &err);
        kernel_sobel  = clCreateKernel(program, "sobel",  &err);
        kernel_median = clCreateKernel(program, "median", &err);

        ready = true;
        std::cout << "OpenCL инициализирован успешно.\n";
    }

    ~OpenCLContext() {
        if (kernel_invert) clReleaseKernel(kernel_invert);
        if (kernel_sobel)  clReleaseKernel(kernel_sobel);
        if (kernel_median) clReleaseKernel(kernel_median);
        if (program) clReleaseProgram(program);
        if (queue)   clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

OpenCLContext g_cl;


void invert_opencl(unsigned char* img, int width, int height, int channels) {
    if (!g_cl.ready) return;

    int total_pixels = width * height;
    int total_bytes  = total_pixels * channels;

    cl_int err;

    cl_mem buffer = clCreateBuffer(g_cl.context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   total_bytes, img, &err);

    clSetKernelArg(g_cl.kernel_invert, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(g_cl.kernel_invert, 1, sizeof(int), &total_pixels);
    clSetKernelArg(g_cl.kernel_invert, 2, sizeof(int), &channels);


    size_t global_size = total_pixels;
    clEnqueueNDRangeKernel(g_cl.queue, g_cl.kernel_invert,
                           1, nullptr, &global_size, nullptr,
                           0, nullptr, nullptr);

    clEnqueueReadBuffer(g_cl.queue, buffer, CL_TRUE,
                        0, total_bytes, img, 0, nullptr, nullptr);

    clReleaseMemObject(buffer);
}

void sobel_opencl(unsigned char* img, int width, int height, int channels) {
    if (!g_cl.ready) return;

    int total_bytes = width * height * channels;
    cl_int err;

    cl_mem buf_in  = clCreateBuffer(g_cl.context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    total_bytes, img, &err);
    cl_mem buf_out = clCreateBuffer(g_cl.context,
                                    CL_MEM_WRITE_ONLY,
                                    total_bytes, nullptr, &err);

    clSetKernelArg(g_cl.kernel_sobel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(g_cl.kernel_sobel, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(g_cl.kernel_sobel, 2, sizeof(int), &width);
    clSetKernelArg(g_cl.kernel_sobel, 3, sizeof(int), &height);
    clSetKernelArg(g_cl.kernel_sobel, 4, sizeof(int), &channels);

    size_t global_size[2] = { (size_t)width, (size_t)height };
    clEnqueueNDRangeKernel(g_cl.queue, g_cl.kernel_sobel,
                           2, nullptr, global_size, nullptr,
                           0, nullptr, nullptr);

    clEnqueueReadBuffer(g_cl.queue, buf_out, CL_TRUE,
                        0, total_bytes, img, 0, nullptr, nullptr);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
}

void median_opencl(unsigned char* img, int width, int height, int channels) {
    if (!g_cl.ready) return;

    int total_bytes = width * height * channels;
    cl_int err;

    cl_mem buf_in  = clCreateBuffer(g_cl.context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    total_bytes, img, &err);
    cl_mem buf_out = clCreateBuffer(g_cl.context,
                                    CL_MEM_WRITE_ONLY,
                                    total_bytes, nullptr, &err);

    clSetKernelArg(g_cl.kernel_median, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(g_cl.kernel_median, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(g_cl.kernel_median, 2, sizeof(int), &width);
    clSetKernelArg(g_cl.kernel_median, 3, sizeof(int), &height);
    clSetKernelArg(g_cl.kernel_median, 4, sizeof(int), &channels);

    size_t global_size[2] = { (size_t)width, (size_t)height };
    clEnqueueNDRangeKernel(g_cl.queue, g_cl.kernel_median,
                           2, nullptr, global_size, nullptr,
                           0, nullptr, nullptr);

    clEnqueueReadBuffer(g_cl.queue, buf_out, CL_TRUE,
                        0, total_bytes, img, 0, nullptr, nullptr);

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
}






// Время мерим 
double benchmark(
    const std::string& name,
    const unsigned char* original,
    int width, int height, int channels,
    int runs,
    void (*filter_fn)(unsigned char*, int, int, int)
) {
    int total_bytes = width * height * channels;

    std::vector<unsigned char> buffer(total_bytes);

    std::copy(original, original + total_bytes, buffer.begin());
    filter_fn(buffer.data(), width, height, channels);


    double total_ms = 0.0;
    for (int run = 0; run < runs; run++) {
        std::copy(original, original + total_bytes, buffer.begin());
        auto t_start = std::chrono::high_resolution_clock::now();
        filter_fn(buffer.data(), width, height, channels);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        total_ms += elapsed.count();
    }

    double avg_ms = total_ms / runs;

    std::cout << "  [" << name << "] среднее за " << runs << " запусков: "
              << avg_ms << " мс\n";

    return avg_ms;
}





int main(int argc, char* argv[]) {
     g_cl.init("filters.cl");

    const char* filename = (argc >= 2) ? argv[1] : "input.png";
    std::cout << "Файл: " << filename << "\n";

    setlocale(0, "RUS");

    int width;     
    int height;    
    int channels;  

   unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);

    if (img == nullptr) {
        std::cout << "Ошибка: не получилось открыть input.png\n";
        std::cout << "Убедись, что файл лежит рядом с программой.\n";
        return 1;  
    }

    std::cout << "Загрузил картинку: " << width << "x" << height
              << ", каналов на пиксель: " << channels << "\n";

    std::cout << " Инверсия цветов ";
    benchmark("Sequential", img, width, height, channels, 10, invert_sequential);
    benchmark("OpenMP", img, width, height, channels, 10, invert_openmp);
    benchmark("SIMD", img, width, height, channels, 10, invert_simd);
    if (g_cl.ready) benchmark("OpenCL", img, width, height, channels, 10, invert_opencl);

    std::cout << " Обнаружение границ ";
    benchmark("Sequential", img, width, height, channels, 10, sobel_sequential);
   benchmark("OpenMP", img, width, height, channels, 10, sobel_openmp);
    benchmark("SIMD", img, width, height, channels, 10, sobel_simd);
    if (g_cl.ready) benchmark("OpenCL", img, width, height, channels, 10, sobel_opencl);

    std::cout << " Медианный фильтр ";
    benchmark("Sequential", img, width, height, channels, 10, median_sequential);
    benchmark("OpenMP", img, width, height, channels, 10, median_openmp);
    benchmark("SIMD", img, width, height, channels, 10, median_simd);
    if (g_cl.ready) benchmark("OpenCL", img, width, height, channels, 10, median_opencl);

    sobel_sequential(img, width, height, channels);

    stbi_write_png("output.png", width, height, channels, img, width * channels);

    std::cout << "Сохранил результат в output.png\n";

    stbi_image_free(img);

    return 0; 
}