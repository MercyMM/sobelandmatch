
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include "cv.h"
#include "highgui.h"
#include "elas.h"
#include <vector>
#include "triangle.h"
#include "matrix.h"
#include <stdlib.h>

using namespace std;

#define WIDTH 320
#define HEIGH 240
//#define HEIGH 120
#define GRID_SIZE 20
enum setting { ROBOTICS, MIDDLEBURY };

// parameter settings
struct parameters {
    int32_t disp_min;               // min disparity
    int32_t disp_max;               // max disparity
    float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)  最大视差唯一性百分比
    int32_t support_texture;        // min texture for support points      最小纹理支持点
    int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
    int32_t incon_window_size;      // window size of inconsistent support point check
    int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
    int32_t incon_min_support;      // minimum number of consistent support points
    bool    add_corners;            // add support points at image corners with nearest neighbor disparities
    int32_t grid_size;              // size of neighborhood for additional support point extrapolation
    float   beta;                   // image likelihood parameter
    float   gamma;                  // prior constant
    float   sigma;                  // prior sigma
    float   sradius;                // prior sigma radius
    int32_t match_texture;          // min texture for dense matching
    int32_t lr_threshold;           // disparity threshold for left/right consistency check
    float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
    int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
    int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
    bool    filter_median;          // optional median filter (approximated)
    bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
    bool    postprocess_only_left;  // saves time by not postprocessing the right image
    bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
                                    // note: for this option D1 and D2 must be passed with size
                                    //       width/2 x height/2 (rounded towards zero)

                                    // constructor
    parameters(setting s = ROBOTICS) {

        // default settings in a robotics environment
        // (do not produce results in half-occluded areas
        //  and are a bit more robust towards lighting etc.)  //默认设置为实验环境，不能在half-occluded环境和大量光照条件下使用
        if (s == ROBOTICS) {
            disp_min = 0;
            disp_max = 63;
            support_threshold = 0.85;
            support_texture = 10;
            candidate_stepsize = 5;
            incon_window_size = 5;
            incon_threshold = 5;
            incon_min_support = 5;
            add_corners = 0;
            grid_size = 20;
            beta = 0.02;
            gamma = 3;
            sigma = 1;
            sradius = 2;
            match_texture = 1;                //dense matching的最小纹理
            lr_threshold = 2;                //一致性检测阈值
            speckle_sim_threshold = 2;                //删除细小片段相似性分割阈值
            speckle_size = 200;              //删除细小片段size
            ipol_gap_width = 7;                //间隙插值阈值
            filter_median = 0;
            filter_adaptive_mean = 0;
            postprocess_only_left = 1;
            subsampling = 0;



            // default settings for middlebury benchmark
            // (interpolate all missing disparities)   middlebury基准，插入所有失踪的视差
        }
        else {
            disp_min = 0;
            disp_max = 63;
            support_threshold = 0.85;
            support_texture = 10;
            candidate_stepsize = 5;
            incon_window_size = 5;
            incon_threshold = 5;
            incon_min_support = 5;
            add_corners = 1;
            grid_size = 20;
            beta = 0.02;
            gamma = 5;
            sigma = 1;
            sradius = 3;
            match_texture = 0;
            lr_threshold = 2;
            speckle_sim_threshold = 1;
            speckle_size = 200;
            ipol_gap_width = 5000;
            filter_median = 1;
            filter_adaptive_mean = 0;
            postprocess_only_left = 0;
            subsampling = 0;
        }
    }
};

// parameter set
parameters param(ROBOTICS);

//static cudaStream_t stream1, stream2, stream3, stream4;

//struct support_pt {
//    int32_t u;
//    int32_t v;
//    int32_t d;
//    support_pt(int32_t u, int32_t v, int32_t d) :u(u), v(v), d(d) {}
//};

//struct support_pt1 {
//    int32_t u;
//    int32_t v;
//    int32_t d;
//};

struct triangle {
    int32_t c1, c2, c3;
    float   t1a, t1b, t1c;
    float   t2a, t2b, t2c;
    triangle(int32_t c1, int32_t c2, int32_t c3) :c1(c1), c2(c2), c3(c3) {}
};


struct triangle1 {
    int32_t c1, c2, c3;
    float   t1a, t1b, t1c;
    float   t2a, t2b, t2c;
    int32_t pointNum;
};
struct plane {
    float   t1a, t1b, t1c;
    float   t2a;
};


void* HostMalWC(void **p, long size)
{
    void *p_g;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)p,size,cudaHostAllocWriteCombined|cudaHostAllocMapped);
    //将常规的主机指针转换成指向设备内存空间的指针
    cudaHostGetDevicePointer(&p_g, *p, 0);
    return p_g;
}

void cudaFreeHost_cpuaa(void *p)
{
    cudaFreeHost(p);
}

void* HostMal(void **p, long size)
{
    void *p_g;
    cudaSetDeviceFlags(cudaDeviceMapHost);
//    cudaHostAlloc((void**)p,size, cudaHostAllocDefault | cudaHostAllocMapped);
    cudaHostAlloc((void**)p, size, cudaHostAllocDefault );
    //将常规的主机指针转换成指向设备内存空间的指针
    cudaHostGetDevicePointer(&p_g, *p, 0);
    return p_g;
}



__global__ void createDesc_gpu_kernel(uint8_t* I_desc, uint8_t* I_du, uint8_t* I_dv)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < WIDTH - 3 && y < HEIGH - 3 && x >= 3 && y >= 3) {

    uint8_t *I_desc_curr;
    uint32_t addr_v0, addr_v1, addr_v2, addr_v3, addr_v4;


//    __shared__ uint8_t      I_dv_share[320 * 5];
//    __shared__ uint8_t      I2_du_share[320 * 3];

        addr_v2 = y * WIDTH;
        addr_v0 = addr_v2 - 2 * WIDTH;
        addr_v1 = addr_v2 - 1 * WIDTH;
        addr_v3 = addr_v2 + 1 * WIDTH;
        addr_v4 = addr_v2 + 2 * WIDTH;

        I_desc_curr = I_desc + (y* WIDTH + x) * 16;

        *(I_desc_curr++) = *(I_du + addr_v0 + x + 0);
        *(I_desc_curr++) = *(I_du + addr_v1 + x - 2);
        *(I_desc_curr++) = *(I_du + addr_v1 + x + 0);
        *(I_desc_curr++) = *(I_du + addr_v1 + x + 2);
        *(I_desc_curr++) = *(I_du + addr_v2 + x - 1);
        *(I_desc_curr++) = *(I_du + addr_v2 + x + 0);
        *(I_desc_curr++) = *(I_du + addr_v2 + x + 0);
        *(I_desc_curr++) = *(I_du + addr_v2 + x + 1);
        *(I_desc_curr++) = *(I_du + addr_v3 + x - 2);
        *(I_desc_curr++) = *(I_du + addr_v3 + x + 0);
        *(I_desc_curr++) = *(I_du + addr_v3 + x + 2);
        *(I_desc_curr++) = *(I_du + addr_v4 + x + 0);
        *(I_desc_curr++) = *(I_dv + addr_v1 + x + 0);
        *(I_desc_curr++) = *(I_dv + addr_v2 + x - 1);
        *(I_desc_curr++) = *(I_dv + addr_v2 + x + 1);
        *(I_desc_curr++) = *(I_dv + addr_v3 + x + 0);

    }
}



int createDesc_gpu(uint8_t* I_desc, uint8_t* I_du, uint8_t* I_dv )
{
    dim3 threads(320, 1);
    dim3 grid( WIDTH / (threads.x), HEIGH / threads.y );

    createDesc_gpu_kernel<<<grid, threads, 0 >>>(I_desc, I_du, I_dv );
//    cudaDeviceSynchronize();
}


int createDesc_gpu2(uint8_t **I_desc, uint8_t* I_du, uint8_t* I_dv )
{
    uint8_t *I_desc_g;
    cudaMalloc((void**)&I_desc_g, 16 * WIDTH * HEIGH * sizeof(uint8_t));

    dim3 threads(320, 1);
    dim3 grid( WIDTH / (threads.x), HEIGH / threads.y );
    createDesc_gpu_kernel<<<grid, threads, 0 >>>(I_desc_g, I_du, I_dv );
    cudaMemcpy(*I_desc, I_desc_g, 16 * WIDTH * HEIGH * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(I_desc_g);
}





__device__ uint32_t getAddressOffsetImage1(const int32_t& u, const int32_t& v, const int32_t& width) {
    return v*width + u;
}

inline uint32_t getAddressOffsetImage(const int32_t& u, const int32_t& v, const int32_t& width) {
    return v*width + u;
}



__device__ unsigned int computeMatchEnergy1(unsigned char* dst1, unsigned char* dst2, int offset) {
    unsigned int a, b, c, e, r0, r4;

    a = abs(*(dst1 + offset) - *(dst2 + offset)) + abs(*(dst1 + offset + 1) - *(dst2 + offset + 1));
    b = abs(*(dst1 + offset + 2) - *(dst2 + offset + 2)) + abs(*(dst1 + offset + 3) - *(dst2 + offset + 3));
    c = abs(*(dst1 + offset + 4) - *(dst2 + offset + 4)) + abs(*(dst1 + offset + 5) - *(dst2 + offset + 5));
    e = abs(*(dst1 + offset + 6) - *(dst2 + offset + 6)) + abs(*(dst1 + offset + 7) - *(dst2 + offset + 7));
    r0 = a + b + c + e;

    a = abs(*(dst1 + offset + 8) - *(dst2 + offset + 8)) + abs(*(dst1 + offset + 9) - *(dst2 + offset + 9));
    b = abs(*(dst1 + offset + 10) - *(dst2 + offset + 10)) + abs(*(dst1 + offset + 11) - *(dst2 + offset + 11));
    c = abs(*(dst1 + offset + 12) - *(dst2 + offset + 12)) + abs(*(dst1 + offset + 13) - *(dst2 + offset + 13));
    e = abs(*(dst1 + offset + 14) - *(dst2 + offset + 14)) + abs(*(dst1 + offset + 15) - *(dst2 + offset + 15));
    r4 = a + b + c + e;

    return r0 + r4;
}

inline uint32_t getAddressOffsetGrid(const int32_t& x, const int32_t& y, const int32_t& d, const int32_t& width, const int32_t& disp_num) {
    return (y*width + x)*disp_num + d;
}

__device__ uint32_t getAddressOffsetGrid1(const int32_t& x, const int32_t& y, const int32_t& d, const int32_t& width, const int32_t& disp_num) {
    return (y*width + x)*disp_num + d;
}


__device__ void updatePosteriorMinimumNew(unsigned char* dst1, unsigned char* dst2, const int32_t &d, int32_t &val, int32_t &min_val, int32_t &min_d) {
    val = computeMatchEnergy1(dst1, dst2, 0);
    if (val<min_val) {
        min_val = val;
        min_d = d;
    }
}

__device__ void updatePosteriorMinimumNew1(unsigned char* dst1, unsigned char* dst2, const int32_t &d, const int32_t &w, int32_t &val, int32_t &min_val, int32_t &min_d) {
    val = computeMatchEnergy1(dst1, dst2, 0) + w;
    if (val<min_val) {
        min_val = val;
        min_d = d;
    }
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}






#define D_candidate_stepsize 5
#define LR_THRESHOLD 2
#define INCON_THRESHOLD 5
#define INCON_MIN_SUPPORT 5
#define INCON_WINDOW_SIZE 5
#define SUPPORT_TEXTURE 10
#define DISP_MIN 0
#define DISP_MAX 63

#define SUPPORT_THRESHOLD 0.85


#define U_STEP 2
#define V_STEP 2
#define WINDOW_SIZE 2
#define MIN_1_E 32767
#define MIN_1_D  -1
#define MIN_2_E 32767
#define MIN_2_D  -1
#define DESC_OFFSET_1  (-16 * U_STEP - 16 * WIDTH * V_STEP)
#define DESC_OFFSET_2  (+16 * U_STEP - 16 * WIDTH * V_STEP)
#define DESC_OFFSET_3  (-16 * U_STEP + 16 * WIDTH * V_STEP)
#define DESC_OFFSET_4  (+16 * U_STEP + 16 * WIDTH * V_STEP)






__global__ void sptMathKernel(int32_t D_can_width, int32_t D_can_height, int16_t* D_can, uint8_t* desc1, uint8_t* desc2)
{
    int32_t u_warp;
    int disp_max_valid;
    int result1 = 0, result2 = 0, result3 = 0, result4 = 0;
    int32_t  line_offset;
    uint8_t *I1_line_addr, *I2_line_addr, *I1_block_addr, *I2_block_addr, *I_line_addr_tmp;
    int32_t sum = 0;
    int16_t min_1_E;
    int16_t min_1_d;
    int16_t min_2_E;
    int16_t min_2_d;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int u, v, d1, d2;

        u = x * D_candidate_stepsize;  //5
        v = y * D_candidate_stepsize;

        if (u >= WINDOW_SIZE + U_STEP && u <= WIDTH - WINDOW_SIZE - 1 - U_STEP \
                && v >= WINDOW_SIZE + V_STEP && v <= HEIGH - WINDOW_SIZE - 1 - V_STEP) {
//        if (u >= 5 && u <= 320 - 6 && v >= 5 && v <= 240 - 6)
            line_offset = 16 * WIDTH*v;
            I1_line_addr = desc1 + line_offset;
            I2_line_addr = desc2 + line_offset;
            I1_block_addr = I1_line_addr + 16 * u;


//            if (sum >= SUPPORT_TEXTURE) {
//                disp_max_valid = min(63, u - WINDOW_SIZE - U_STEP);
            disp_max_valid = min(63, u - 5);
                if (disp_max_valid >= 10) {
#pragma unroll
                        min_1_E = MIN_1_E;
                        min_1_d = MIN_1_D;
                        min_2_E = MIN_2_E;
                        min_2_d = MIN_2_D;
                    for (int16_t d = 0; d <= disp_max_valid; d++) {
                        u_warp = u - d;
                        I2_block_addr = I2_line_addr + 16 * u_warp;
                        result1 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_1);
                        result2 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_2);
                        result3 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_3);
                        result4 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_4);
                        result4 = computeMatchEnergy1(I1_block_addr, I2_block_addr, 0);

                        sum = result1 + result2 + result3 + result4;

                        if (sum<min_1_E) {
                            min_2_E = min_1_E;
                            min_2_d = min_1_d;
                            min_1_E = sum;
                            min_1_d = d;
                        }
                        else if (sum<min_2_E) {
                            min_2_E = sum;
                            min_2_d = d;
                        }
                    }
                    //sum_dev[x + y * D_can_width] = 10;

                    if (min_1_d >= 0 && min_2_d >= 0 && (float)min_1_E < SUPPORT_THRESHOLD*(float)min_2_E) {
                        if ((d1= min_1_d) >= 0) {
                            u = u - d1;
                            D_can[x + y * D_can_width] = d1;
                        /*
                            if (u >= WINDOW_SIZE + U_STEP && u <= WIDTH - WINDOW_SIZE - 1 - U_STEP && v >= WINDOW_SIZE + V_STEP && v <= HEIGH - WINDOW_SIZE - 1 - V_STEP) {
                                //I1_line_addr = desc2 + line_offset;
                                //I2_line_addr = desc1 + line_offset;
                                I_line_addr_tmp = I1_line_addr;
                                I1_line_addr = I2_line_addr;
                                I2_line_addr = I_line_addr_tmp;

                                I1_block_addr = I1_line_addr + 16 * u;

                                sum = 0;
                                min_1_E = MIN_1_E;
                                min_1_d = MIN_1_D;
                                min_2_E = MIN_2_E;
                                min_2_d = MIN_2_D;
#pragma unroll
                                for (int32_t i = 0; i < 16; i++)
                                    sum += abs((int32_t)(*(I1_block_addr + i)) - 128);
                                if (sum >= SUPPORT_TEXTURE) {
                                    disp_max_valid = min(63, WIDTH - u - WINDOW_SIZE - U_STEP);
                                    if (disp_max_valid >= 10) {
#pragma unroll
                                        for (int16_t d = 0; d <= disp_max_valid; d++) {
                                            u_warp = u + d;
                                            I2_block_addr = I2_line_addr + 16 * u_warp;
                                            result1 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_1);
                                            result2 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_2);
                                            result3 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_3);
                                            result4 = computeMatchEnergy1(I1_block_addr, I2_block_addr, DESC_OFFSET_4);
                                            sum = result1 + result2 + result3 + result4;
                                            if (sum<min_1_E) {
                                                min_2_E = min_1_E;
                                                min_2_d = min_1_d;
                                                min_1_E = sum;
                                                min_1_d = d;
                                            }
                                            else if (sum<min_2_E) {
                                                min_2_E = sum;
                                                min_2_d = d;
                                            }
                                        }
                                        if (min_1_d >= 0 && min_2_d >= 0 && (float)min_1_E < SUPPORT_THRESHOLD*(float)min_2_E) {
                                             d2 = min_1_d;
                                             if (d2 >= 0 && abs(d1 - d2) <= LR_THRESHOLD) {
                                                 D_can[x + y * D_can_width] = d1;

                                             }
                                        }
                                    }
                                }
                            }

                        */
                        }
                    }

                }
               /*
#pragma unroll
//            for (int32_t i = 0; i < 16; i++)
            int32_t i = 0;
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
            sum += abs((int32_t)(*(I1_block_addr + i++)) - 128);
                if (sum <= SUPPORT_TEXTURE) {
                    D_can[x + y * D_can_width] = -1;

                }
                */
//            }


        }
//    }
}


void addCornerSupportPoints(vector<Elas::support_pt> &p_support, int32_t width, int32_t height) {

    // list of border points
    vector<Elas::support_pt> p_border;
    p_border.push_back(Elas::support_pt(0, 0, 0));
    p_border.push_back(Elas::support_pt(0, height - 1, 0));
    p_border.push_back(Elas::support_pt(width - 1, 0, 0));
    p_border.push_back(Elas::support_pt(width - 1, height - 1, 0));

    // find closest d
    for (int32_t i = 0; i<p_border.size(); i++) {
        int32_t best_dist = 10000000;
        for (int32_t j = 0; j<p_support.size(); j++) {
            int32_t du = p_border[i].u - p_support[j].u;
            int32_t dv = p_border[i].v - p_support[j].v;
            int32_t curr_dist = du*du + dv*dv;
            if (curr_dist<best_dist) {
                best_dist = curr_dist;
                p_border[i].d = p_support[j].d;
            }
        }
    }

    // for right image
    p_border.push_back(Elas::support_pt(p_border[2].u + p_border[2].d, p_border[2].v, p_border[2].d));
    p_border.push_back(Elas::support_pt(p_border[3].u + p_border[3].d, p_border[3].v, p_border[3].d));

    // add border points to support points
    for (int32_t i = 0; i<p_border.size(); i++)
        p_support.push_back(p_border[i]);
}


__global__ void removeInconsistentSupportPoints1(int16_t* D_can, int32_t D_can_width, int32_t D_can_height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int u, v;

    if (x < D_can_width && y < D_can_height) {
        int16_t d_can = *(D_can + getAddressOffsetImage1(x, y, D_can_width));
        if (d_can >= 0) {
            int32_t support = 0;
            for (int32_t u_can_2 = x - INCON_WINDOW_SIZE; u_can_2 <= x + INCON_WINDOW_SIZE; u_can_2++) {
                for (int32_t v_can_2 = y - INCON_WINDOW_SIZE; v_can_2 <= y + INCON_WINDOW_SIZE; v_can_2++) {
                    if (u_can_2 >= 0 && v_can_2 >= 0 && u_can_2<D_can_width && v_can_2<D_can_height) {
                        int16_t d_can_2 = *(D_can + getAddressOffsetImage1(u_can_2, v_can_2, D_can_width));
                        if (d_can_2 >= 0 && abs(d_can - d_can_2) <= INCON_THRESHOLD)
                            support++;
                    }
                }
            }

            // invalidate support point if number of supporting points is too low
            if (support<INCON_MIN_SUPPORT)
                *(D_can + getAddressOffsetImage1(x, y, D_can_width)) = -1;
        }
    }
}

__global__ void removeRedundantSupportPoints1(int16_t* D_can, int32_t D_can_width, int32_t D_can_height,
    int32_t redun_max_dist, int32_t redun_threshold, bool vertical) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < D_can_width && y < D_can_height) {
        // parameters
        int32_t redun_dir_u[2] = { 0,0 };
        int32_t redun_dir_v[2] = { 0,0 };
        if (vertical) {
            redun_dir_v[0] = -1;
            redun_dir_v[1] = +1;
        }
        else {
            redun_dir_u[0] = -1;
            redun_dir_u[1] = +1;
        }
        int16_t d_can = *(D_can + getAddressOffsetImage1(x, y, D_can_width));
        if (d_can >= 0) {
            // check all directions for redundancy
            bool redundant = true;
            for (int32_t i = 0; i<2; i++) {

                // search for support
                int32_t u_can_2 = x;
                int32_t v_can_2 = y;
                int16_t d_can_2;
                bool support = false;
                for (int32_t j = 0; j<redun_max_dist; j++) {
                    u_can_2 += redun_dir_u[i];
                    v_can_2 += redun_dir_v[i];
                    if (u_can_2<0 || v_can_2<0 || u_can_2 >= D_can_width || v_can_2 >= D_can_height)
                        break;
                    d_can_2 = *(D_can + getAddressOffsetImage1(u_can_2, v_can_2, D_can_width));
                    if (d_can_2 >= 0 && abs(d_can - d_can_2) <= redun_threshold) {
                        support = true;
                        break;
                    }
                }

                // if we have no support => point is not redundant
                if (!support) {
                    redundant = false;
                    break;
                }
            }

            // invalidate support point if it is redundant
            if (redundant)
                *(D_can + getAddressOffsetImage1(x, y, D_can_width)) = -1;
        }
    }
}


vector<Elas::support_pt> computeSupportMatches_g(uint8_t* I_desc1, uint8_t* I_desc2)
{

    // create matrix for saving disparity candidates
    int32_t D_can_width = 64;
    int32_t D_can_height = 48;
//    for (int32_t u = 0; u<WIDTH; u += D_candidate_stepsize) D_can_width++;  //64
//    for (int32_t v = 0; v<HEIGH; v += D_candidate_stepsize) D_can_height++; //58



    int16_t* D_can_gpu;
    int16_t* sum_gpu;
    int16_t* D_can_cpu = (int16_t*)malloc(D_can_width*D_can_height * sizeof(int16_t)); //143 * 99

    cudaMalloc((void **)&D_can_gpu, D_can_width*D_can_height * sizeof(int16_t));

    dim3 threads(64, 1);
    dim3 grid(D_can_width / threads.x, D_can_height / threads.y);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemset(D_can_gpu, -1, D_can_width*D_can_height * sizeof(int16_t));

    sptMathKernel << <grid, threads, 0, stream1>> > (D_can_width, D_can_height, D_can_gpu, I_desc1, I_desc2);

    cudaMemcpy(D_can_cpu, D_can_gpu, D_can_width*D_can_height * sizeof(int16_t), cudaMemcpyDeviceToHost);


    // remove inconsistent support points
    removeInconsistentSupportPoints1<< <grid, threads >> > (D_can_gpu, D_can_width, D_can_height);

    removeRedundantSupportPoints1 << <grid, threads >> > (D_can_gpu, D_can_width, D_can_height, 5, 1, true);
    removeRedundantSupportPoints1 << <grid, threads >> > (D_can_gpu, D_can_width, D_can_height, 5, 1, false);

    cudaMemcpy(D_can_cpu, D_can_gpu, D_can_width*D_can_height * sizeof(int16_t), cudaMemcpyDeviceToHost);

    // move support points from image representation into a vector representation
    vector<Elas::support_pt> p_support;
    for (int32_t u_can = 1; u_can<D_can_width; u_can++)
        for (int32_t v_can = 1; v_can<D_can_height; v_can++)
            if (*(D_can_cpu + getAddressOffsetImage(u_can, v_can, D_can_width)) >= 0)
                p_support.push_back(Elas::support_pt(u_can*D_candidate_stepsize,
                    v_can*D_candidate_stepsize,
                    *(D_can_cpu + getAddressOffsetImage(u_can, v_can, D_can_width))));

    // if flag is set, add support points in image corners
    // with the same disparity as the nearest neighbor support point
//    if (param.add_corners)
//        addCornerSupportPoints(p_support, WIDTH, HEIGH);

    // free memory
    free(D_can_cpu);
    // return support point vector
    return p_support;
}












__constant__ int32_t grid_dims_g[3] = {65, WIDTH/GRID_SIZE, HEIGH/GRID_SIZE} ;

__global__ void Triangle_Match1(triangle1* tri, int32_t* disparity_grid,\
                                uint8_t* I1_desc, uint8_t* I2_desc, int32_t* P, \
                                int32_t plane_radius, bool right_image, float* D,  \
                                int32_t* tp)

{

    float plane_a = 0, plane_b = 0, plane_c = 0, plane_d = 0;

        int u = blockDim.x * blockIdx.x + threadIdx.x;
        int v = blockDim.y * blockIdx.y + threadIdx.y;
        int32_t id;
        __shared__ uint8_t      I1_desc_share[320 * 16];
        __shared__ uint8_t      I2_desc_share[320 * 16];



        for(int i = 0; i < 16; i += 1 )
        {
           I1_desc_share[u + i*320] = I1_desc[v * 320*16 + u + i*320];
           I2_desc_share[u + i*320 ] = I2_desc[v * 320*16 + u + i*320];
        }




    __syncthreads();


    id = tp[2 * u + v * 2 * WIDTH + 1];

    plane_a = tri[id].t1a;
    plane_b = tri[id].t1b;
    plane_c = tri[id].t1c;
    plane_d = tri[id].t2a;


            bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;

            // get image width and height
            const int32_t disp_num = grid_dims_g[0] - 1;
//            const int32_t disp_num = grid_dims[0] - 1;
            const int32_t window_size = 2;

            // address of disparity we want to compute
            uint32_t d_addr;
            d_addr = getAddressOffsetImage1(u, v, WIDTH);




            // compute line start address
            int32_t  line_offset = 16 * WIDTH*max(min(v, HEIGH - 3), 2);
            uint8_t *I1_line_addr, *I2_line_addr;

//                I1_line_addr = I1_desc + line_offset;
//                I2_line_addr = I2_desc + line_offset;
//            uint8_t* I1_block_addr = I1_line_addr + 16 * u;

            I2_line_addr = I2_desc_share ;
            uint8_t* I1_block_addr = I1_desc_share + 16 * u;

            // does this patch have enough texture?
            int32_t sum = 0;
//int32_t match_texture = 1;
//        //#pragma unroll
//            for (int32_t i = 0; i<16; i++)
//                sum += abs((int32_t)(*(I1_block_addr + i)) - 127);
//            if (sum<match_texture)
//                return;

            // compute disparity, min disparity and max disparity of plane prior
//            int32_t d_plane = (int32_t)(plane_a*(float)u + plane_b*(float)v + plane_c);
            int32_t d_plane = (int32_t)(0);
            int32_t d_plane_min = max(d_plane - plane_radius, 0);
            int32_t d_plane_max = min(d_plane + plane_radius, disp_num - 1);

            // get grid pointer
            int32_t  grid_x = (int32_t)floor((float)u / (float)GRID_SIZE);
            int32_t  grid_y = (int32_t)floor((float)v / (float)GRID_SIZE);

            uint32_t grid_addr = getAddressOffsetGrid1(grid_x, grid_y, 0, grid_dims_g[1], grid_dims_g[0]);
            int32_t  num_grid = *(disparity_grid + grid_addr);
            int32_t* d_grid = disparity_grid + grid_addr + 1;

//            uint32_t grid_addr = grid_x * grid_dims_g[0];
//            int32_t  num_grid = *(disparity_grid_g + grid_addr);
//            int32_t* d_grid = disparity_grid_g + grid_addr + 1;

            // loop variables
            int32_t d_curr, u_warp, val;
            int32_t min_val = 10000;
            int32_t min_d = -1;


            // left image

            if (!right_image) {
        //#pragma unroll
                for (int32_t i = 0; i<num_grid; i++) {
                    d_curr = d_grid[i];
                    if (d_curr<d_plane_min || d_curr>d_plane_max) {
                        u_warp = u - d_curr;
//                        if (u_warp<window_size || u_warp >= WIDTH - window_size)
//                            continue;
//                        updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
                        updatePosteriorMinimumNew(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, val, min_val, min_d);
//                          updatePosteriorMinimumNew(I1_block_addr, I2_desc_share + 16 * u_warp, d_curr, val, min_val, min_d);

                    }
                }
        //#pragma unroll
                for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++) {
                    u_warp = u - d_curr;
//                    if (u_warp<window_size || u_warp >= WIDTH - window_size)
//                        continue;
//                    updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
                    updatePosteriorMinimumNew1(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, valid ? *(P + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
//                    updatePosteriorMinimumNew1(I1_block_addr, I2_desc_share + 16 * u_warp, d_curr, valid ? *(P + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
                }

                // right image
            }
            else {
        //#pragma unroll
                for (int32_t i = 0; i<num_grid; i++) {
                    d_curr = d_grid[i];
                    if (d_curr<d_plane_min || d_curr>d_plane_max) {
                        u_warp = u + d_curr;
                        if (u_warp<window_size || u_warp >= WIDTH - window_size)
                            continue;
//                        updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
                        updatePosteriorMinimumNew(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, val, min_val, min_d);
//                        updatePosteriorMinimumNew(I1_block_addr, I2_desc_share + 16 * u_warp, d_curr, val, min_val, min_d);
                    }
                }
        //#pragma unroll
                for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++) {
                    u_warp = u + d_curr;
                    if (u_warp<window_size || u_warp >= WIDTH - window_size)
                        continue;
//                    updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
                    updatePosteriorMinimumNew1(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, valid ? *(P + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
//                    updatePosteriorMinimumNew1(I1_block_addr, I2_desc_share + 16 * u_warp, d_curr, valid ? *(P + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
                }
            }



            // set disparity value
            if (min_d >= 0) *(D + d_addr) = min_d; // MAP value (min neg-Log probability)
            else          *(D + d_addr) = -1;    // invalid disparity



}





//void computeTrianglePoints(support_pt1* p_support, triangle1* tri, bool right_image, int32_t width, int32_t TRI_SIZE, int32_t* tp) {
void computeTrianglePoints(const vector<Elas::support_pt> &p_support, const vector<Elas::triangle> &tri, \
                           bool right_image, int32_t width, int32_t TRI_SIZE, int32_t* tp) {

    // loop variables
    int32_t c1, c2, c3;
//    float plane_a, plane_b, plane_c, plane_d;

    // for all triangles do
    for (uint32_t i = 0; i<TRI_SIZE; i++) {
        int num = 0;
        // get plane parameters
        uint32_t p_i = i * 3;

        // triangle corners
        c1 = tri[i].c1;
        c2 = tri[i].c2;
        c3 = tri[i].c3;

        // sort triangle corners wrt. u (ascending)
        float tri_u[3];
        if (!right_image) {     //左图像
            tri_u[0] = p_support[c1].u;
            tri_u[1] = p_support[c2].u;
            tri_u[2] = p_support[c3].u;
        }
        else {                //右图像
            tri_u[0] = p_support[c1].u - p_support[c1].d;
            tri_u[1] = p_support[c2].u - p_support[c2].d;
            tri_u[2] = p_support[c3].u - p_support[c3].d;
        }
        float tri_v[3] = { p_support[c1].v,p_support[c2].v,p_support[c3].v };

        for (uint32_t j = 0; j<3; j++) {
            for (uint32_t k = 0; k<j; k++) {
                if (tri_u[k]>tri_u[j]) {
                    float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
                    float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
                }
            }
        }

        // rename corners
        float A_u = tri_u[0]; float A_v = tri_v[0];
        float B_u = tri_u[1]; float B_v = tri_v[1];
        float C_u = tri_u[2]; float C_v = tri_v[2];

        // compute straight lines connecting triangle corners
        float AB_a = 0; float AC_a = 0; float BC_a = 0;
        if ((int32_t)(A_u) != (int32_t)(B_u)) AB_a = (A_v - B_v) / (A_u - B_u);
        if ((int32_t)(A_u) != (int32_t)(C_u)) AC_a = (A_v - C_v) / (A_u - C_u);
        if ((int32_t)(B_u) != (int32_t)(C_u)) BC_a = (B_v - C_v) / (B_u - C_u);
        float AB_b = A_v - AB_a*A_u;
        float AC_b = A_v - AC_a*A_u;
        float BC_b = B_v - BC_a*B_u;


        // first part (triangle corner A->B)
        if ((int32_t)(A_u) != (int32_t)(B_u)) {
            for (int32_t u = max((int32_t)A_u, 0); u < min((int32_t)B_u, width); u++) {
                if (!param.subsampling || u % 2 == 0) {
                    int32_t v_1 = (uint32_t)(AC_a*(float)u + AC_b);
                    int32_t v_2 = (uint32_t)(AB_a*(float)u + AB_b);
                    for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
                        if (!param.subsampling || v % 2 == 0)
                        {
                            *((int16_t*)(tp + 2 * u + v * 2 * width)) = u;
                            *((int16_t*)(tp + 2 * u + v * 2 * width) + 1) = v;
                            *(tp + 2 * u + v * 2 * width + 1) = i;
//                            num++;
                        }
                }
            }

        }

        // second part (triangle corner B->C)
        if ((int32_t)(B_u) != (int32_t)(C_u)) {
            for (int32_t u = max((int32_t)B_u, 0); u < min((int32_t)C_u, width); u++) {
                if (!param.subsampling || u % 2 == 0) {
                    int32_t v_1 = (uint32_t)(AC_a*(float)u + AC_b);
                    int32_t v_2 = (uint32_t)(BC_a*(float)u + BC_b);
                    for (int32_t v = min(v_1, v_2); v < max(v_1, v_2); v++)
                        if (!param.subsampling || v % 2 == 0)
                        {
                            *((int16_t*)(tp + 2 * u + v * 2 * width)) = u;
                            *((int16_t*)(tp + 2 * u + v * 2 * width) + 1) = v;
                            *(tp + 2 * u + v * 2 * width + 1) = i;
//                            num++;
                        }
                }
            }
        }
//        tri[i].pointNum = num;
    }

}



int32_t width, height, bpl;

uint8_t* I_desc1 = NULL;
uint8_t* I_desc2 = NULL;
int32_t* grid_dims_gpu = NULL;
int32_t* disparity_grid_gpu_1 = NULL;
int32_t* disparity_grid_gpu_2 = NULL;
float* D1_gpu = NULL;
float* D2_gpu = NULL;
int32_t* P_gpu = NULL;
triangle1* tri_gpu_1, *tri_gpu_2;
plane *plane_1, *plane_2;
plane *plane_g1, *plane_g2;


cudaError_t err;

int32_t dims[3] = {WIDTH,HEIGH,WIDTH};

static int flag = 1;


void cuda_computeD(int32_t* disparity_grid_1, int32_t* disparity_grid_2,  vector<Elas::support_pt> &p_support, \
              vector<Elas::triangle> &tri_1, vector<Elas::triangle> &tri_2, \
              float* D1, float* D2, uint8_t* I1, uint8_t* I2)
{



    clock_t t1, t2;

    // get width, height and bytes per line
    width = dims[0];    //715*492
    height = dims[1];
    bpl = width + 15 - (width - 1) % 16;  //720

    // allocate memory for disparity grid
    int32_t grid_width = (int32_t)ceil((float)width / (float)20);
    int32_t grid_height = (int32_t)ceil((float)height / (float)20);
    int32_t grid_dims[3] = { 63 + 2,grid_width,grid_height };
//    grid_dims[3] = { 63 + 2,grid_width,grid_height };


        cudaSetDeviceFlags(cudaDeviceMapHost);



int32_t P_SUPPORT_SIZE = p_support.size();
int32_t TRI_SIZE1 = tri_1.size();
int32_t TRI_SIZE2 = tri_2.size();

int32_t* tp1_cpu, *tp2_cpu;
int32_t *tp1_gpu, *tp2_gpu;


//cout<<"P_SUPPORT_SIZE: "<<P_SUPPORT_SIZE<<endl;
//cout<< "TRI_SIZE1: " << TRI_SIZE1 <<endl;
//cout<< "TRI_SIZE2: " << TRI_SIZE2 <<endl;


cudaHostAlloc((void**)&tp2_cpu,sizeof(int32_t) * width * height * 2,cudaHostAllocWriteCombined|cudaHostAllocMapped);
cudaHostAlloc((void**)&tp1_cpu,sizeof(int32_t) * width * height * 2,cudaHostAllocWriteCombined|cudaHostAllocMapped);

//tp2_cpu = (int32_t*)malloc(sizeof(int32_t) * width * height * 2);
//tp1_cpu = (int32_t*)malloc(sizeof(int32_t) * width * height * 2);

for (int j = 0; j < height; j++) {
    for (int i = 0; i < width * 2; i++) {
        tp1_cpu[i + j * width * 2] = -1;
        tp2_cpu[i + j * width * 2] = -1;
    }
}

t1 = clock();
computeTrianglePoints(p_support, tri_1, 0, width, TRI_SIZE1, tp1_cpu);
computeTrianglePoints(p_support, tri_2, 1, width, TRI_SIZE2, tp2_cpu);
t2 = clock();
//printf("computeTripoints : %ldms\n", (t2 - t1)/1000);

cudaHostGetDevicePointer(&tp1_gpu, tp1_cpu, 0);
cudaHostGetDevicePointer(&tp2_gpu, tp2_cpu, 0);

//if(1 == flag){
//    cudaMalloc((void **)&tp1_gpu, sizeof(int32_t) * width * height * 2);
//    cudaMalloc((void **)&tp2_gpu, sizeof(int32_t) * width * height * 2);
//}

//t1 = clock();
//cudaMemcpy(tp1_gpu, tp1_cpu, sizeof(int32_t) * width * height * 2, cudaMemcpyHostToDevice);
//cudaMemcpy(tp2_gpu, tp2_cpu, sizeof(int32_t) * width * height * 2, cudaMemcpyHostToDevice);
//t2 = clock();
//printf("cudaMemcpy tp1_gpu: %ldms\n", (t2 - t1)/1000);

//if(1 == flag){
//    cudaMalloc((void **)&grid_dims_gpu, sizeof(int32_t) * 3);
    cudaMalloc((void **)&disparity_grid_gpu_1, sizeof(int32_t) * (param.disp_max + 2) * grid_height * grid_width);
    cudaMalloc((void **)&disparity_grid_gpu_2, sizeof(int32_t) * (param.disp_max + 2) * grid_height * grid_width);
//}
t1 = clock();
//cudaMemcpy(grid_dims_gpu, grid_dims, sizeof(int32_t) * 3, cudaMemcpyHostToDevice);
cudaMemcpy(disparity_grid_gpu_1, disparity_grid_1, sizeof(int32_t) * (param.disp_max + 2) * grid_height * grid_width, cudaMemcpyHostToDevice);
cudaMemcpy(disparity_grid_gpu_2, disparity_grid_2, sizeof(int32_t) * (param.disp_max + 2) * grid_height * grid_width, cudaMemcpyHostToDevice);
t2 = clock();
//printf("cudaMemcpy disparity_grid_gpu_2: %ldms\n", (t2 - t1)/1000);



t1 = clock();
    cudaMalloc((void **)&tri_gpu_1, sizeof(triangle1) * TRI_SIZE1);
    cudaMalloc((void **)&tri_gpu_2, sizeof(triangle1) * TRI_SIZE2);

//    cudaMalloc((void **)&D1_gpu, sizeof(float) * width * height);
//    cudaMalloc((void **)&D2_gpu, sizeof(float) * width * height);
    cudaMalloc((void **)&P_gpu, sizeof(int32_t) * width * height);
    cudaMalloc((void **)&I_desc1, 16 * width*height * sizeof(uint8_t));
    cudaMalloc((void **)&I_desc2, 16 * width*height * sizeof(uint8_t));

t2 = clock();
//printf("cudaMalloc I_desc: %ldms\n", (t2 - t1)/1000);

flag = 0;
t1 = clock();
cudaMemcpy(tri_gpu_1, &tri_1[0], sizeof(Elas::triangle) * TRI_SIZE1, cudaMemcpyHostToDevice);
cudaMemcpy(tri_gpu_2, &tri_2[0], sizeof(Elas::triangle) * TRI_SIZE2, cudaMemcpyHostToDevice);
//cudaMemcpy(D1_gpu, D1, sizeof(float) * width * height, cudaMemcpyHostToDevice);
//cudaMemcpy(D2_gpu, D2, sizeof(float) * width * height, cudaMemcpyHostToDevice);
cudaMemcpy(I_desc1, I1, 16 * width*height * sizeof(uint8_t), cudaMemcpyHostToDevice);
cudaMemcpy(I_desc2, I2, 16 * width*height * sizeof(uint8_t), cudaMemcpyHostToDevice);
t2 = clock();
//printf("cudaMemcpy I_desc: %ldms\n", (t2 - t1)/1000);


// number of disparities
const int32_t disp_num = grid_dims[0] - 1;
// descriptor window_size
int32_t window_size = 2;
// pre-compute prior
float two_sigma_squared = 2 * param.sigma*param.sigma;
int32_t* P = new int32_t[disp_num];
for (int32_t delta_d = 0; delta_d<disp_num; delta_d++)
    P[delta_d] = (int32_t)((-log(param.gamma + exp(-delta_d*delta_d / two_sigma_squared)) + log(param.gamma)) / param.beta);
int32_t plane_radius = (int32_t)max((float)ceil(param.sigma*param.sradius), (float)2.0);
//plane_radius = 2;
t1 = clock();
cudaMemcpy(P_gpu, P, sizeof(int32_t) * disp_num, cudaMemcpyHostToDevice);
t2 = clock();
//printf("cudaMemcpy P_gpu: %ldms\n", (t2 - t1)/1000);


//bool subsampling = param.subsampling;
//int32_t match_texture = param.match_texture;
//int32_t grid_size = param.grid_size;



dim3 threads(320, 1);
dim3 grid(iDivUp(width, (threads.x)), iDivUp(height,threads.y));

//printf("goin Triangle_match kernel\n");
t1 = clock();
Triangle_Match1 << <grid, threads, 0>> > (tri_gpu_1, disparity_grid_gpu_1, \
              I_desc1, I_desc2, P_gpu, plane_radius, 0, D1, tp1_gpu);

Triangle_Match1 << <grid, threads, 0>> > (tri_gpu_2, disparity_grid_gpu_2, \
     I_desc2, I_desc1, P_gpu, plane_radius, 1, D2, \
                      tp2_gpu);

//cudaMemcpy(D1, D1_gpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
//cudaMemcpy(D2, D2_gpu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
t2 = clock();
//printf("Triangle_Match1 : %ldms\n", (t2 - t1)/1000);

//cudaThreadExit();

//cudaFree(D1_gpu);
//cudaFree(D2_gpu);
//cudaFree(P_gpu);
//cudaFree(I_desc1);
//cudaFree(I_desc2);
//cudaFree(tp1_gpu);
//cudaFree(tp2_gpu);
//cudaFree(grid_dims_gpu);
//cudaFree(disparity_grid_gpu_1);
//cudaFree(disparity_grid_gpu_2);
//cudaFree(plane_g1);
//cudaFree(plane_g2);
  cudaDeviceSynchronize();



}














//err = cudaFuncSetCacheConfig(Triangle_Match1,cudaFuncCachePreferL1);
//if(cudaSuccess != err)
//{
//    printf("cudaFuncSetCacheConfig error %s\n", cudaGetErrorString(err));
//}


//Triangle_Match1 << <1, 715*492>> > (tri_gpu_1, disparity_grid_gpu_1, \
//    grid_dims_gpu, I_desc1, I_desc2, P_gpu, plane_radius, 0, D1_gpu, width, height, TRI_SIZE1, subsampling, \
//                                                   match_texture, grid_size, tp1_gpu);




//for(int i = 0; i< 10000 ; i++)
//{
//    //cout <<I1+i<<" ";
//    printf("%d ", *(D1+i));
//    if(i%20 == 0)
//        cout<<endl;
//}


//    for(int i = 10000; i< 11000 ; i++)
//    {
//        //cout <<I1+i<<" ";
//        printf("%d ", *(D1+i));
//        if(i%20 == 0)
//            cout<<endl;
//    }





//printf("over memcpy\n");
// err = cudaGetLastError();
//if(cudaSuccess != err)
//{
//    printf("error %s\n", cudaGetErrorString(err));
//}


//    for(int i = 7000; i< 8000 ; i++)
//    {
//        //cout <<I1+i<<" ";
//        printf("%d ", *(D1+i));
//        if(i%20 == 0)
//            cout<<endl;
//    }

//int main()
//{
//    cuda_computeD(int32_t* disparity_grid_1, int32_t* disparity_grid_2,  vector<Elas::support_pt> &p_support, \
//                  vector<Elas::triangle> &tri_1, vector<Elas::triangle> &tri_2, \
//                  float* D1, float* D2,uint8_t* I1, uint8_t* I2, int dim);
//    return 0;
//}
