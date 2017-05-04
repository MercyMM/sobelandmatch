
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

/***
 * 1. cudaHostAlloc and cudaFreeHost wrap
 * 2. create Descriptor of two img
 * 3. compute support point
 * 4. convert dis to cloud
 * 5. cuda_computeD
 * */



#define WIDTH 320
#define HEIGH 240
#define GRID_SIZE 20
enum setting { ROBOTICS, MIDDLEBURY };

// parameter set
Elas::parameters param;


struct plane {
    float   t1a, t1b, t1c;
    float   t2a;
};

/**
 * 1. cudaHostAlloc and cudaFreeHost wrap
 *
 * */
static int count_alloc = 0;
static int count_free = 0;
void cudaFreeHost_cpuaa(void *p)
{
    count_free++;
    cudaFreeHost(p);
    cudaError_t err = cudaGetLastError();
    if(0 != err) printf("cuda error: %s\n", cudaGetErrorString(err));
}

void SetDeviceMap()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

void* HostMal(void **p, long size)
{
    count_alloc++;
    void *p_g;
//    cudaHostAlloc((void**)p,size, cudaHostAllocDefault | cudaHostAllocMapped);
    cudaHostAlloc((void**)p, size, cudaHostAllocDefault );
    cudaError_t err = cudaGetLastError();
    if(0 != err) printf("cuda error: %s\n", cudaGetErrorString(err));
    //将常规的主机指针转换成指向设备内存空间的指针
    cudaHostGetDevicePointer(&p_g, *p, 0);
    err = cudaGetLastError();
    if(0 != err) printf("cuda error: %s\n", cudaGetErrorString(err));
    return p_g;
}

void allocFreeCount()
{
    printf("count_alloc = %d\n", count_alloc);
    printf("count_free= %d\n", count_free);
}

/**
 * 1. over: cudaHostAlloc and cudaFreeHost wrap
 * */

//dim3 threads(320 - 6 , 1);
//dim3 grid( 1, 240 -6 ); => (0, 233)



/**
 * 2. create Descriptor of two img
 * */
__global__ \
void createDesc_gpu_kernel(uint8_t* I_desc, uint8_t* I_du, uint8_t* I_dv)
{

    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;

    int x = u + 3;
    int y = v + 3;

    __shared__ uint8_t      I_du_share[320 * 5];
    __shared__ uint8_t      I_dv_share[320 * 3];

    uint8_t *I_desc_curr;

    for(int i = 0; i < 5; i++){
        *(I_du_share + x + i * 320) = *(I_du + x + (y-2 + i) * 320);
    }

    for(int i = 0; i < 3; i++){
        *(I_dv_share + x + i * 320) = *(I_dv + x + (y-1 + i) * 320);
    }

    __syncthreads();

    I_desc_curr = I_desc + (y* WIDTH + x) * 16;

    *(I_desc_curr++) = *(I_du_share + (0 * WIDTH + x + 0) );
    *(I_desc_curr++) = *(I_du_share + (1 * WIDTH + x - 2) );
    *(I_desc_curr++) = *(I_du_share + (1 * WIDTH + x + 0) );
    *(I_desc_curr++) = *(I_du_share + (1 * WIDTH + x + 2) );

    *(I_desc_curr++) = *(I_du_share + (2 * WIDTH + x - 1) );
    *(I_desc_curr++) = *(I_du_share + (2 * WIDTH + x + 0) );
    *(I_desc_curr++) = *(I_du_share + (2 * WIDTH + x + 0) );
    *(I_desc_curr++) = *(I_du_share + (2 * WIDTH + x + 2) );

    *(I_desc_curr++) = *(I_du_share + (3 * WIDTH + x - 2) );
    *(I_desc_curr++) = *(I_du_share + (3 * WIDTH + x + 0) );
    *(I_desc_curr++) = *(I_du_share + (3 * WIDTH + x + 2) );
    *(I_desc_curr++) = *(I_du_share + (4 * WIDTH + x + 0) );

    *(I_desc_curr++) = *(I_dv_share + (0 * WIDTH + x + 0) );
    *(I_desc_curr++) = *(I_dv_share + (1 * WIDTH + x + 1) );
    *(I_desc_curr++) = *(I_dv_share + (1 * WIDTH + x - 1) );
    *(I_desc_curr++) = *(I_dv_share + (2 * WIDTH + x + 0) );

}

int __createDesc_gpu(uint8_t* I_desc, uint8_t* I_du_g, uint8_t* I_dv_g )
{
    dim3 threads(WIDTH - 6 , 1);
    dim3 grid( 1, HEIGH - 6 );
//    cudaDeviceSynchronize();
    createDesc_gpu_kernel<<<grid, threads, 0 >>>(I_desc, I_du_g, I_dv_g );
//    cudaError_t err = cudaGetLastError();
//    printf("cuda error: %s\n", cudaGetErrorString(err));
//    cudaDeviceSynchronize(); //2.88 - 0.19

}

/**
 * 2. over: create Descriptor of two img
 * */



/**
 * 3. compute support point
 * */
__device__ \
uint32_t getAddressOffsetImage1(const int32_t& u, const int32_t& v, const int32_t& width)
{
    return v*width + u;
}

__device__ \
unsigned int computeMatchEnergy1(unsigned char* dst1, unsigned char* dst2, int offset)
{
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


__device__ \
uint32_t getAddressOffsetGrid1(const int32_t& x, const int32_t& y, \
                                          const int32_t& d, const int32_t& width, const int32_t& disp_num)
{
    return (y*width + x)*disp_num + d;
}


__device__ \
void updatePosteriorMinimumNew(unsigned char* dst1, unsigned char* dst2, const int32_t &d, int32_t &val, int32_t &min_val, int32_t &min_d)
{
    val = computeMatchEnergy1(dst1, dst2, 0);
    if (val<min_val) {
        min_val = val;
        min_d = d;
    }
}

__device__ \
void updatePosteriorMinimumNew1(unsigned char* dst1, unsigned char* dst2, const int32_t &d, const int8_t w, int32_t &val, int32_t &min_val, int32_t &min_d)
{
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

__device__ \
unsigned int computeMatchEnergy1_new(unsigned char* dst1_1, unsigned char* dst1_2, unsigned char* dst2_1, unsigned char* dst2_2, int32_t u, int32_t u_wrap) {
    unsigned int r0, r1, r2, r3;
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;

#pragma unroll
    for (int i = 0; i < 16; i++) {
        r0 += abs(dst2_1[(u_wrap << 4) - 32 + i] - dst1_1[(u << 4) - 32 + i]);
        r1 += abs(dst2_1[(u_wrap << 4) + 32 + i] - dst1_1[(u << 4) + 32 + i]);
        r2 += abs(dst2_2[(u_wrap << 4) - 32 + i] - dst1_2[(u << 4) - 32 + i]);
        r3 += abs(dst2_2[(u_wrap << 4) + 32 + i] - dst1_2[(u << 4) + 32 + i]);
    }

    return r0 + r1 + r2 + r3;
}


#define D_candidate_stepsize 5
#define INCON_THRESHOLD 5
#define INCON_MIN_SUPPORT 5
#define INCON_WINDOW_SIZE 5
#define SUPPORT_TEXTURE 10
#define DISP_MIN 0
#define DISP_MAX 63

#define SUPPORT_THRESHOLD 0.85


#define U_STEP 2
#define V_STEP 2
#define WINDOW_SIZE 3
#define MIN_1_E 32767
#define MIN_1_D  -1
#define MIN_2_E 32767
#define MIN_2_D  -1
#define DESC_OFFSET_1  (-16 * U_STEP)
#define DESC_OFFSET_2  (+16 * U_STEP)
#define DESC_OFFSET_3  (-16 * U_STEP)
#define DESC_OFFSET_4  (+16 * U_STEP)


#define BLOCKX  60
#define BLOCKY  1
#define GRIDX   1
#define GRIDY   46
//#define GRIDY   2
//dim3 threads(60, 1);
//dim3 grid(1, 46);

__constant__ uint32_t oneLine = WIDTH * 16;

__global__ void compEner_gpu(uint8_t* I1_desc_shared, uint8_t* I2_desc_shared, int u, int u_wrap,  uint32_t* sumResult)
{

    int x = threadIdx.x; // x = (0,15)
    int32_t sum = 0;
    sum  = abs(I1_desc_shared[(u - 2) << 4 + x ] - I2_desc_shared[(u_wrap - 2) << 4 + x]);
    sum += abs(I1_desc_shared[(u + 2) << 4 + x ] - I2_desc_shared[(u_wrap + 2) << 4 + x]);
    sum += abs(I1_desc_shared[(u + 2) << 4 + x  + oneLine] - I2_desc_shared[(u_wrap + 2) << 4 + x +oneLine]);
    sum += abs(I1_desc_shared[(u - 2) << 4 + x  + oneLine] - I2_desc_shared[(u_wrap - 2) << 4 + x +oneLine]);
    sumResult[x] = sum;
}

__global__ void sptMathKernel(int32_t D_can_width, int32_t D_can_height, int8_t* D_can, uint8_t* desc1, uint8_t* desc2)
{
    int32_t u_wrap;
    int disp_max_valid;
    int result1 = 0, result2 = 0, result3 = 0, result4 = 0;
    int32_t  line_offset;
    uint8_t *I1_line_addr, *I2_line_addr, *I1_block_addr, *I2_block_addr, *I_line_addr_tmp;
    uint8_t *I1_block_addr_1, *I1_block_addr_2, *I2_block_addr_1, *I2_block_addr_2;
    int32_t sum = 0;
    int16_t min_1_E;
    int16_t min_1_d;
    int16_t min_2_E;
    int16_t min_2_d;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int u, v, d1 = -1 , d2 = -1;

    u = (x + 3) * D_candidate_stepsize;  //5
    v = (y + 1) * D_candidate_stepsize;
    line_offset = 16 * WIDTH*v;
    I1_line_addr = desc1 + line_offset;
    I2_line_addr = desc2 + line_offset;

    __shared__ uint8_t I1_desc_shared[320 * 16 * 2];
    __shared__ uint8_t I2_desc_shared[320 * 16 * 2];

    for(int i = 0; i < 85; i++){
        I1_desc_shared[x + i * BLOCKX ] = *(I1_line_addr + x + i * BLOCKX - 2 * oneLine);
        I1_desc_shared[x + i * BLOCKX + oneLine] = *(I1_line_addr + x + i * BLOCKX + 2 * oneLine);

        I2_desc_shared[x + i * BLOCKX ] = *(I2_line_addr + x + i * BLOCKX - 2 * oneLine);
        I2_desc_shared[x + i * BLOCKX + oneLine] = *(I2_line_addr + x + i * BLOCKX + 2 * oneLine);
    }

    __syncthreads();

    for (int32_t i=0; i<16; i++)
        sum += abs((int32_t)(*(I1_line_addr + 16 * u +i))-128);
    if (sum<10){
      d1 = -1;
      return;
    }

    I1_block_addr_1 = I1_desc_shared + 16 * u ;
    I1_block_addr_2 = I1_desc_shared + 16 * u + oneLine ;

    disp_max_valid = min(63, u - 5);
    min_1_E = MIN_1_E;
    min_1_d = MIN_1_D;
    min_2_E = MIN_2_E;
    min_2_d = MIN_2_D;


    for (int16_t d = 0; d <= disp_max_valid; d++) {
        u_wrap = u - d;
        I2_block_addr_1 = I2_desc_shared + 16 * u_wrap;
        I2_block_addr_2 = I2_desc_shared + 16 * u_wrap + oneLine;
    //    result1 = computeMatchEnergy1(I1_block_addr_1, I2_block_addr_1, DESC_OFFSET_1);
    //    result2 = computeMatchEnergy1(I1_block_addr_1, I2_block_addr_1, DESC_OFFSET_2);
    //    result3 = computeMatchEnergy1(I1_block_addr_2, I2_block_addr_2, DESC_OFFSET_3);
        result4 = computeMatchEnergy1(I1_block_addr_2, I2_block_addr_2, DESC_OFFSET_4);
    //    sum = result1 + result2 + result3 + result4;
	sum = result4;
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

    if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E < 0.85*(float)min_2_E)
      d1 =  min_1_d;

    sum = 0;
    if (d1 >= 0){
        min_1_E = MIN_1_E;
        min_1_d = MIN_1_D;
        min_2_E = MIN_2_E;
        min_2_d = MIN_2_D;
        u = u - d1;
        disp_max_valid = min(63, WIDTH - u - 5);
        I2_block_addr_1 = I2_desc_shared + 16 * u;
        I2_block_addr_2 = I2_desc_shared + 16 * u + 320 * 16;

        sum = 0;
    #pragma unroll
        for (int32_t i=0; i<16; i++)
          sum += abs((int32_t)(*(I2_line_addr+i))-128);
        if (sum<10){
            return;
        }
        sum = 0;

        for(int16_t d = 0; d <= disp_max_valid; d++){
            u_wrap = u + d;
            I1_block_addr_1 = I1_desc_shared + 16 * u_wrap;
            I1_block_addr_2 = I1_desc_shared + 16 * u_wrap + 320 * 16;
        //    result1 = computeMatchEnergy1(I1_block_addr_1, I2_block_addr_1, DESC_OFFSET_1);
        //    result2 = computeMatchEnergy1(I1_block_addr_1, I2_block_addr_1, DESC_OFFSET_2);
        //    result3 = computeMatchEnergy1(I1_block_addr_2, I2_block_addr_2, DESC_OFFSET_3);
            result4 = computeMatchEnergy1(I1_block_addr_2, I2_block_addr_2, DESC_OFFSET_4);

          //  sum = result1 + result2 + result3 + result4;
	sum = result4;
//            sum = computeMatchEnergy1_new(I2_desc_shared, I2_desc_shared + oneLine, I1_desc_shared, I1_desc_shared + oneLine, u, u_wrap);

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

        if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E < 0.85*(float)min_2_E)
          d2 =  min_1_d;
        if( d2 >= 0 && abs(d2 - d1) <= 2 )
            D_can[x + y * D_can_width] = d1;
    }



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

vector<Elas::support_pt> computeSupportMatches_g(uint8_t* I_desc1, uint8_t* I_desc2, \
                                                 int8_t* D_sup_c, int8_t* D_sup_g)
{

    // create matrix for saving disparity candidates
    int32_t D_can_width = 60;  //[15,310] => 60
    int32_t D_can_height = 48; //[5, 230] => 46

    cudaMemset(D_sup_g, -1, D_can_width*D_can_height * sizeof(int8_t));

    dim3 threads(BLOCKX, BLOCKY);
    dim3 grid(GRIDX, GRIDY);
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaFuncSetCacheConfig(sptMathKernel,cudaFuncCachePreferShared);
    //compute support
    cudaDeviceSynchronize();    //
    sptMathKernel << <grid, threads, 0, stream1>> > (D_can_width, D_can_height, D_sup_g, I_desc1, I_desc2);
    cudaDeviceSynchronize();    //13ms

    //put D_sup to vector of support
    vector<Elas::support_pt> p_support;
    for (int32_t u_can = 0; u_can<D_can_width; u_can++)
        for (int32_t v_can = 0; v_can<D_can_height; v_can++)
            if (*(D_sup_c + u_can + v_can * D_can_width) >= 0)
                p_support.push_back(Elas::support_pt((u_can + 3)*D_candidate_stepsize,
                    (v_can + 1) * D_candidate_stepsize,
                    *(D_sup_c+ u_can + v_can * D_can_width)));


    return p_support;
}

/**
 * 3. over: compute support point
 * */


/**
 * 4. convert dis to cloud
 ***/

__global__ void Convert(float *D_g, float *cloud_g)
{
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    v += 20;
    float w = 0, x = 0, y = 0, z = 0;
    float dis = D_g[u + v * WIDTH];


        w = 0.006669723997311648 * dis;
        x = (float)((u - 161.2100334167481) / w);
        y = (float)((219 - v - 119.9240913391113) / w); //has bug
        z = (float)(241.57918 / w);

    *(cloud_g + (u + v * WIDTH) * 3) = x;
    *(cloud_g + (u + v * WIDTH) * 3 + 1) = y;
    *(cloud_g + (u + v * WIDTH) * 3 + 2) = z;

//        *(cloud_g + (u + v * WIDTH) * 3)     = z;
//        *(cloud_g + (u + v * WIDTH) * 3 + 1) = y;
//        *(cloud_g + (u + v * WIDTH) * 3 + 2) = x;

    //            A = [u, v, d,1];
    //            Q = [1, 0, 0, -161.2100334167481; 0, 1, 0, -119.9240913391113;
    //                 0, 0, 0, 241.57918; 0, 0, 0.006669723997311648, 0]


}

int ConvertD2Z(float* D1_g,  float* cloud_g)
{
    dim3 threads(320, 1);
    dim3 grid(1, 200);
    Convert<<<grid, threads>>>(D1_g, cloud_g);
    cudaDeviceSynchronize();
}


/**
 * 4. over: convert dis to cloud
 ***/



/****
 * 5. cuda_computeD
 ***/
int tri_size = 0;
__constant__ int32_t grid_dims_g[3] = {65, WIDTH/GRID_SIZE, HEIGH/GRID_SIZE} ;
__constant__ int8_t temp[] = {-14,-9,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

__global__ void Triangle_Match1(Elas::triangle* tri, int32_t* disparity_grid,\
                                uint8_t* I1_desc, uint8_t* I2_desc, int8_t* P, \
                                int32_t plane_radius, bool right_image, float* D,  \
                                int8_t* tp, int tri_size)
{

    float plane_a = 0, plane_b = 0, plane_c = 0, plane_d = 0;

    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int32_t id;
    __shared__ uint8_t      __I1_desc_share[320 * 16];
    __shared__ uint8_t      __I2_desc_share[320 * 16];

    for(int i = 0; i < 16; i += 1 )
    {
        __I1_desc_share[u + i*320] = I1_desc[v * 320*16 + u + i*320];
        __I2_desc_share[u + i*320] = I2_desc[v * 320*16 + u + i*320];
    }
    __syncthreads();

    id = tp[u + v * WIDTH];

        if(id > tri_size)
            printf("\nid out %d\n", id);

    plane_a = tri[id].t1a;
    plane_b = tri[id].t1b;
    plane_c = tri[id].t1c;
    plane_d = tri[id].t2a;


    bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;
    const int32_t window_size = 2;
    // address of disparity we want to compute
    uint32_t d_addr;
    d_addr = getAddressOffsetImage1(u, v, WIDTH);

            uint8_t *I1_line_addr, *I2_line_addr;
            I2_line_addr = __I2_desc_share ;
            uint8_t* I1_block_addr = __I1_desc_share + 16 * u;

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
            int32_t d_plane_max = min(d_plane + plane_radius, grid_dims_g[0] - 2);

            // get grid pointer
            int32_t  grid_x = (int32_t)floor((float)u / (float)GRID_SIZE);
            int32_t  grid_y = (int32_t)floor((float)v / (float)GRID_SIZE);

            uint32_t grid_addr = getAddressOffsetGrid1(grid_x, grid_y, 0, grid_dims_g[1], grid_dims_g[0]);
            int32_t  num_grid = *(disparity_grid + grid_addr);
            int32_t* d_grid = disparity_grid + grid_addr + 1;

            // loop variables
            int32_t d_curr, u_warp, val;
            int32_t min_val = 10000;
            int32_t min_d = -1;


            // left image

            if (!right_image) {
        #pragma unroll
                for (int32_t i = 0; i<num_grid; i++) {
                    d_curr = d_grid[i];
                    if (d_curr<d_plane_min || d_curr>d_plane_max) {
                        u_warp = u - d_curr;
                       updatePosteriorMinimumNew(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, val, min_val, min_d);
                    }
                }
        #pragma unroll
                for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++) {
                    u_warp = u - d_curr;
//                    updatePosteriorMinimumNew1(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, valid ? *(P + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
                    updatePosteriorMinimumNew1(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, valid ? *(temp + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
                }
            }
            else {
        #pragma unroll
                for (int32_t i = 0; i<num_grid; i++) {
                    d_curr = d_grid[i];
                    if (d_curr<d_plane_min || d_curr>d_plane_max) {
                        u_warp = u + d_curr;
                        if (u_warp<window_size || u_warp >= WIDTH - window_size)
                            continue;
                        updatePosteriorMinimumNew(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, val, min_val, min_d);
                    }
                }
        #pragma unroll
                for (d_curr = d_plane_min; d_curr <= d_plane_max; d_curr++) {
                    u_warp = u + d_curr;
                    if (u_warp<window_size || u_warp >= WIDTH - window_size)
                        continue;
//                    updatePosteriorMinimumNew1(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, valid ? *(P + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
                    updatePosteriorMinimumNew1(I1_block_addr, I2_line_addr + 16 * u_warp, d_curr, valid ? *(temp + abs(d_curr - d_plane)) : 0, val, min_val, min_d);
                }
            }



            // set disparity value
//            if (min_d >= 0){
            if (min_d > 0){
                *(D + d_addr) = min_d; // MAP value (min neg-Log probability)
            }else          *(D + d_addr) = -1;    // invalid disparity
}

void computeTrianglePoints(const vector<Elas::support_pt> &p_support, const vector<Elas::triangle> &tri, \
                           bool right_image, int32_t width, int32_t TRI_SIZE, int8_t* tp) {

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
//                            *((int16_t*)(tp + 2 * u + v * 2 * width)) = u;
//                            *((int16_t*)(tp + 2 * u + v * 2 * width) + 1) = v;
//                            *(tp + 2 * u + v * 2 * width + 1) = i;
                            *(tp + u + v * width ) = i;
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
//                            *((int16_t*)(tp + 2 * u + v * 2 * width)) = u;
//                            *((int16_t*)(tp + 2 * u + v * 2 * width) + 1) = v;
//                            *(tp + 2 * u + v * 2 * width + 1) = i;
                              *(tp + u + v * width) = i;
//                            num++;
                        }
                }
            }
        }
//        tri[i].pointNum = num;
    }

}

cudaError_t err;

int32_t dims[3] = {WIDTH, HEIGH, WIDTH};

void cuda_computeD(int32_t* disparity_grid_1, int32_t* disparity_grid_2,  vector<Elas::support_pt> &p_support, \
              vector<Elas::triangle> &tri_1, vector<Elas::triangle> &tri_2, \
              float* D1, float* D2, uint8_t* I1, uint8_t* I2, int8_t* P_g,\
             int8_t *tp1_g, int8_t* tp2_g, int8_t* tp1_c, int8_t* tp2_c)
{    
    int32_t width, height, bpl;
    clock_t t1, t2;

    // get width, height and bytes per line
    width = dims[0];    //
    height = dims[1];
    bpl = dims[2];  //

    // allocate memory for disparity grid
    int32_t grid_width = 16; //(int32_t)ceil((float)width / (float)20);
    int32_t grid_height = 12; //(int32_t)ceil((float)height / (float)20);
    int32_t grid_dims[3] = { 63 + 2,grid_width,grid_height };



    int32_t P_SUPPORT_SIZE = p_support.size();
    int32_t TRI_SIZE1 = tri_1.size();
    int32_t TRI_SIZE2 = tri_2.size();
    tri_size = TRI_SIZE1;

    int8_t* tp1_cpu = tp1_c;
    int8_t* tp2_cpu = tp2_c;
    int8_t *tp1_gpu = tp1_g;
    int8_t *tp2_gpu = tp2_g;


    computeTrianglePoints(p_support, tri_1, 0, width, TRI_SIZE1, tp1_cpu);
    computeTrianglePoints(p_support, tri_2, 1, width, TRI_SIZE2, tp2_cpu);

    Elas::triangle* tri_gpu_1, *tri_gpu_2;
    int32_t *P_gpu = NULL;  //action::::  cannot delete;

    cudaMalloc((void **)&tri_gpu_1, sizeof(Elas::triangle) * TRI_SIZE1);
    cudaMalloc((void **)&tri_gpu_2, sizeof(Elas::triangle) * TRI_SIZE2);
    cudaMalloc((void **)&P_gpu, sizeof(int8_t) * 64);
    err = cudaGetLastError();
    if(0 != err) printf("cuda error: %s\n", cudaGetErrorString(err));


    cudaMemcpy(tri_gpu_1, &tri_1[0], sizeof(Elas::triangle) * TRI_SIZE1, cudaMemcpyHostToDevice);
    cudaMemcpy(tri_gpu_2, &tri_2[0], sizeof(Elas::triangle) * TRI_SIZE2, cudaMemcpyHostToDevice);
    cudaMemcpy(P_gpu, P_g, sizeof(int8_t) * 64, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if(0 != err) printf("cuda error: %s\n", cudaGetErrorString(err));

    int32_t plane_radius = 2; //(int32_t)max((float)ceil(param.sigma*param.sradius), (float)2.0);

    dim3 threads(320, 1);
    dim3 grid(1, 240);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if(0 != err) printf("cuda error: %s\n", cudaGetErrorString(err));

    printf("goin Triangle_match kernel\n");
    Triangle_Match1 << <grid, threads, 0>> > (tri_gpu_1, disparity_grid_1, \
                  I1, I2, P_g, plane_radius, 0, D1, tp1_gpu, tri_size);

    tri_size = TRI_SIZE2;
    Triangle_Match1 << <grid, threads, 0>> > (tri_gpu_2, disparity_grid_2, \
                  I2, I1, P_g, plane_radius, 1, D2, tp2_gpu, tri_size);
    err = cudaGetLastError();
    if(0 != err) printf("Triangle_Match1 cuda error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(0 != err) printf("Triangle_Match1 cuda error: %s\n", cudaGetErrorString(err));

    cudaFree((void*)tri_gpu_1);
    cudaFree((void*)tri_gpu_2);
    cudaFree((void*)P_gpu);
    //  cudaDeviceSynchronize();



}


/****
 * 5. over: cuda_computeD
 ***/
