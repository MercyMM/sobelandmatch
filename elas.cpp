#include "elas.h"

#include <algorithm>
#include <math.h>
#include "triangle.h"
#include "matrix.h"
#include <arm_neon.h>
#include <sys/time.h>


const double CLK_TCK = 1000.0;

//#include <CL/cl.h>

using namespace std;

struct support_pt {
  int32_t u;
  int32_t v;
  int32_t d;
  support_pt(int32_t u,int32_t v,int32_t d):u(u),v(v),d(d){}
};

struct triangle {
  int32_t c1,c2,c3;
  float   t1a,t1b,t1c;
  float   t2a,t2b,t2c;
  triangle(int32_t c1,int32_t c2,int32_t c3):c1(c1),c2(c2),c3(c3){}
};


extern void* HostMal(void **p, long size);
extern void cuda_computeD(int32_t* disparity_grid_1, int32_t* disparity_grid_2,  vector<Elas::support_pt> &p_support, \
              vector<Elas::triangle> &tri_1, vector<Elas::triangle> &tri_2, \
              float* D1, float* D2,uint8_t* I1, uint8_t* I2, int8_t* P_g,\
                          int8_t *tp1_g, int8_t* tp2_g, int8_t* tp1_c, int8_t* tp2_c);

//extern vector<Elas::support_pt> computeSupportMatches_g(uint8_t* I_desc1, uint8_t* I_desc2) ;
extern vector<Elas::support_pt> computeSupportMatches_g(uint8_t* I1_desc, uint8_t* I2_desc,\
                                                 int8_t* D_sup_c, int8_t* D_sup_g);

#define GRID_WIDTH      16
#define GRID_HEIGH      12
#define WIDTH           320
#define HEIGH           240
#define D_CAN_WIDTH     60
#define D_CAN_HEIGH     48

Elas::Elas(parameters param, int32_t width, int32_t height, int32_t D_can_width, int32_t D_can_height)
:param(param),desc_1(width, height, width),\
desc_2(width,height,width)
{

    D_sup_g         = (int8_t*)HostMal((void**)&D_sup_c, D_can_width*D_can_height * sizeof(int8_t) );

    memset(D_sup_c, -1, D_can_width*D_can_height * sizeof(int8_t));
//    D1_data_g       = (float*)HostMal((void**)&D1_data_c, width * height * sizeof(float) * 3);
//    D2_data_g       = (float*)HostMal((void**)&D2_data_c, width * height * sizeof(float) * 3);

    D1_data_g       = (float*)HostMal((void**)&D1_data_c, width * height * sizeof(float));
    D2_data_g       = (float*)HostMal((void**)&D2_data_c, width * height * sizeof(float));

    disp_grid_1_g   = (int32_t*)HostMal((void**)&disp_grid_1_c,  65 * GRID_WIDTH * GRID_HEIGH * sizeof(int32_t));
    disp_grid_2_g   = (int32_t*)HostMal((void**)&disp_grid_2_c,  65 * GRID_WIDTH * GRID_HEIGH * sizeof(int32_t));
    tp1_g           = (int8_t*)HostMal((void**)&tp1_c, width * height * sizeof(int8_t) );
    tp2_g           = (int8_t*)HostMal((void**)&tp2_c, width * height * sizeof(int8_t) );

    P_g             = (int8_t*)HostMal((void**)&P_c, 64 * sizeof(int8_t));

    int8_t temp[] = {-14,-9,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64};
    for(int i = 0 ; i < 64; i++){
        P_c[i] = temp[i];
    }
}





void Elas::process (uint8_t* I1_,uint8_t* I2_)
{

      struct timeval start, end;
      clock_t t1,t2;
      width  = WIDTH;// dims[0];
      height = HEIGH; // dims[1];
      bpl    = width + 15-(width-1)%16;
      double timeuse;
      I1 = I1_;
      I2 = I2_;

      gettimeofday(&start, NULL);
      desc_1.compute(I1_,width,height,bpl,param.subsampling);
      desc_2.compute(I2_,width,height,bpl,param.subsampling);
      gettimeofday(&end, NULL);
      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
      cout << "two desc: " << timeuse/1000 << "ms" <<endl;

      gettimeofday(&start, NULL);
      vector<support_pt> p_support = computeSupportMatches_g(desc_1.I_desc_g, desc_2.I_desc_g,  D_sup_c, D_sup_g);
      memset(D_sup_c, -1, D_CAN_WIDTH*D_CAN_HEIGH * sizeof(int8_t));
     //      vector<support_pt> p_support = computeSupportMatches(desc_1.I_desc, desc_2.I_desc);
      gettimeofday(&end, NULL);
      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
      cout <<"computesuppportmatch: "<< timeuse/1000 << "ms. " ; cout << "support size: "<<p_support.size() <<endl;

      // if not enough support points for triangulation
      if (p_support.size()<3)
      {
        cout << "ERROR: Need at least 3 support points!" << endl;
        return;
      }

      gettimeofday(&start, NULL);
      vector<triangle> tri_1 = computeDelaunayTriangulation(p_support,0);
      vector<triangle> tri_2 = computeDelaunayTriangulation(p_support,1);
      gettimeofday(&end, NULL);
      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
      cout <<"Delaunay : "<< timeuse/1000 << "ms" <<endl;

      gettimeofday(&start, NULL);
      computeDisparityPlanes(p_support,tri_1,0);
      computeDisparityPlanes(p_support,tri_2,1);
      gettimeofday(&end, NULL);
      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
      cout <<"computediparityplanes: "<< timeuse/1000 << "ms" <<endl;


      // allocate memory for disparity grid
      int32_t grid_width   = 16; //int32_t)ceil((float)width/(float)param.grid_size);
      int32_t grid_height  = 12; //(int32_t)ceil((float)height/(float)param.grid_size);
      int32_t grid_dims[3] = {param.disp_max+2,grid_width,grid_height};

      gettimeofday(&start, NULL);
      createGrid(p_support, disp_grid_1_c, grid_dims,0);
      createGrid(p_support, disp_grid_2_c, grid_dims,1);
      gettimeofday(&end, NULL);
      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
      cout <<"creategrid: "<< timeuse/1000 << "ms" <<endl;


      gettimeofday(&start, NULL);
      cuda_computeD(disp_grid_1_g, disp_grid_1_g, p_support, tri_1, tri_2, D1_data_g, D2_data_g,\
                    desc_1.I_desc_g, desc_2.I_desc_g, P_g, tp1_g, tp2_g, tp1_c, tp2_c );
      gettimeofday(&end, NULL);
      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
      cout <<"cuda_computeD: "<< timeuse/1000 << "ms" <<endl;


//      gettimeofday(&start, NULL);
//      leftRightConsistencyCheck(D1_data_c, D2_data_c);
//      gettimeofday(&end, NULL);
//      timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
//      cout <<"leftRightConsistencyCheck: "<< timeuse/1000 << "ms" <<endl;


}

void Elas::removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height) {
  
  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      if (d_can>=0) {
        
        // compute number of other points supporting the current point
        int32_t support = 0;
        for (int32_t u_can_2=u_can-param.incon_window_size; u_can_2<=u_can+param.incon_window_size; u_can_2++) {
          for (int32_t v_can_2=v_can-param.incon_window_size; v_can_2<=v_can+param.incon_window_size; v_can_2++) {
            if (u_can_2>=0 && v_can_2>=0 && u_can_2<D_can_width && v_can_2<D_can_height) {
              int16_t d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
              if (d_can_2>=0 && abs(d_can-d_can_2)<=param.incon_threshold)
                support++;
            }
          }
        }
        
        // invalidate support point if number of supporting points is too low
        if (support<param.incon_min_support)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}

void Elas::removeRedundantSupportPoints(int16_t* D_can,int32_t D_can_width,int32_t D_can_height,
                                        int32_t redun_max_dist, int32_t redun_threshold, bool vertical) {
  
  // parameters
  int32_t redun_dir_u[2] = {0,0};
  int32_t redun_dir_v[2] = {0,0};
  if (vertical) {
    redun_dir_v[0] = -1;
    redun_dir_v[1] = +1;
  } else {
    redun_dir_u[0] = -1;
    redun_dir_u[1] = +1;
  }
    
  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      if (d_can>=0) {
        
        // check all directions for redundancy
        bool redundant = true;
        for (int32_t i=0; i<2; i++) {
          
          // search for support
          int32_t u_can_2 = u_can;
          int32_t v_can_2 = v_can;
          int16_t d_can_2;
          bool support = false;
          for (int32_t j=0; j<redun_max_dist; j++) {
            u_can_2 += redun_dir_u[i];
            v_can_2 += redun_dir_v[i];
            if (u_can_2<0 || v_can_2<0 || u_can_2>=D_can_width || v_can_2>=D_can_height)
              break;
            d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
            if (d_can_2>=0 && abs(d_can-d_can_2)<=redun_threshold) {
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
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}

void Elas::addCornerSupportPoints(vector<support_pt> &p_support) {
  
  // list of border points
  vector<support_pt> p_border;
  p_border.push_back(support_pt(0,0,0));
  p_border.push_back(support_pt(0,height-1,0));
  p_border.push_back(support_pt(width-1,0,0));
  p_border.push_back(support_pt(width-1,height-1,0));
  
  // find closest d
  for (int32_t i=0; i<p_border.size(); i++) {
    int32_t best_dist = 10000000;
    for (int32_t j=0; j<p_support.size(); j++) {
      int32_t du = p_border[i].u-p_support[j].u;
      int32_t dv = p_border[i].v-p_support[j].v;
      int32_t curr_dist = du*du+dv*dv;
      if (curr_dist<best_dist) {
        best_dist = curr_dist;
        p_border[i].d = p_support[j].d;
      }
    }
  }
  
  // for right image
  p_border.push_back(support_pt(p_border[2].u+p_border[2].d,p_border[2].v,p_border[2].d));
  p_border.push_back(support_pt(p_border[3].u+p_border[3].d,p_border[3].v,p_border[3].d));
  
  // add border points to support points
  for (int32_t i=0; i<p_border.size(); i++)
    p_support.push_back(p_border[i]);
}

inline int16_t Elas::computeMatchingDisparity (const int32_t &u,const int32_t &v,uint8_t* I1_desc,uint8_t* I2_desc,const bool &right_image) {

    const int32_t u_step      = 2;
    const int32_t v_step      = 2;
    const int32_t window_size = 3;

    int32_t desc_offset_1 = -16*u_step-16*width*v_step;
    int32_t desc_offset_2 = +16*u_step-16*width*v_step;
    int32_t desc_offset_3 = -16*u_step+16*width*v_step;
    int32_t desc_offset_4 = +16*u_step+16*width*v_step;

    // check if we are inside the image region
    if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {
        // get valid disparity range
        int32_t disp_min_valid = max(param.disp_min,0);
        int32_t disp_max_valid = param.disp_max;
        if (!right_image) disp_max_valid = min(param.disp_max,u-window_size-u_step);  // min(63, u-5)
        else              disp_max_valid = min(param.disp_max,width-u-window_size-u_step);
      if( 310 == u && 15 == v){
          printf("disp_max = %d\n", disp_max_valid);
      }
        // assume, that we can compute at least 10 disparities for this pixel
        if (disp_max_valid-disp_min_valid<10)
          return -1;

//        if (!right_image) printf("(%d, %d) ", u, v);
      // compute desc and start addresses
      int32_t  line_offset = 16*width*v;
      uint8_t *I1_line_addr,*I2_line_addr;
      if (!right_image) {
        I1_line_addr = I1_desc+line_offset;
        I2_line_addr = I2_desc+line_offset;
      } else {
        I1_line_addr = I2_desc+line_offset;
        I2_line_addr = I1_desc+line_offset;
      }

      // compute I1 block start addresses
      uint8_t* I1_block_addr = I1_line_addr+16*u;
      uint8_t* I2_block_addr;




//      if(20 == u && 20 == v){
//                for( int i = 0; i < 16; i++){
//                    printf("%d ",*( I1_block_addr - desc_offset_1 + i) );
//                }
//                for( int i = 0; i < 16; i++){
//                    printf("%d ",*( I2_block_addr - desc_offset_1 + i) );
//                }
//                printf("\n");
//                printf("\n");
//      }




      // we require at least some texture
      int32_t sum = 0;
      for (int32_t i=0; i<16; i++)
        sum += abs((int32_t)(*(I1_block_addr+i))-128);

      if (sum< 100)
        return -1;

      // load first blocks to xmm registers
//      xmm1 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_1));
//      xmm2 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_2));
//      xmm3 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_3));
//      xmm4 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_4));

//      printf(" %d ", sum);
      // declare match energy for each disparity
      int32_t u_warp;

      // best match
      int16_t min_1_E = 32767;
      int16_t min_1_d = -1;
      int16_t min_2_E = 32767;
      int16_t min_2_d = -1;

      // for all disparities do
      for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {

        // warp u coordinate
        if (!right_image) u_warp = u-d;
        else              u_warp = u+d;

        // compute I2 block start addresses
        I2_block_addr = I2_line_addr+16*u_warp;



        // compute match energy at this disparity

        uint32_t sum2=0;
        uint32_t sumA, sumB, sumC, sumD;
  uint8x16_t l_1 = vld1q_u8(I1_block_addr+desc_offset_1);
  uint8x16_t l_2 = vld1q_u8(I1_block_addr+desc_offset_2);
  uint8x16_t l_3 = vld1q_u8(I1_block_addr+desc_offset_3);
  uint8x16_t l_4 = vld1q_u8(I1_block_addr+desc_offset_4);


  uint8x16_t r_1 = vld1q_u8(I2_block_addr+desc_offset_1);
  uint8x16_t r_2 = vld1q_u8(I2_block_addr+desc_offset_2);
  uint8x16_t r_3 = vld1q_u8(I2_block_addr+desc_offset_3);
  uint8x16_t r_4 = vld1q_u8(I2_block_addr+desc_offset_4);

  uint8x16_t out1 = vdupq_n_u8(0);
  uint8x16_t out2 = vdupq_n_u8(0);
  uint8x16_t out3 = vdupq_n_u8(0);
  uint8x16_t out4 = vdupq_n_u8(0);

  out1 = vabaq_u8(out1, l_1, r_1);
  out2 = vabaq_u8(out2, l_2, r_2);
  out3 = vabaq_u8(out3, l_3, r_3);
  out4 = vabaq_u8(out4, l_4, r_4);

uint16x8_t sum_1 = vaddl_u8(vget_low_u8(out1), vget_high_u8(out1));
uint16x8_t sum_2 = vaddl_u8(vget_low_u8(out2), vget_high_u8(out2));
uint16x8_t sum_3 = vaddl_u8(vget_low_u8(out3), vget_high_u8(out3));
uint16x8_t sum_4 = vaddl_u8(vget_low_u8(out4), vget_high_u8(out4));

uint32x4_t sum_11 = vaddl_u16(vget_low_u16(sum_1), vget_high_u16(sum_1));
uint32x4_t sum_21 = vaddl_u16(vget_low_u16(sum_2), vget_high_u16(sum_2));
uint32x4_t sum_31 = vaddl_u16(vget_low_u16(sum_3), vget_high_u16(sum_3));
uint32x4_t sum_41 = vaddl_u16(vget_low_u16(sum_4), vget_high_u16(sum_4));

sum_11 = vaddq_u32(sum_11, sum_21);
sum_31 = vaddq_u32(sum_31, sum_41);
sum_11 = vaddq_u32(sum_11, sum_31);

//sum2 += vgetq_lane_u32(sum_11, 0);
//sum2 += vgetq_lane_u32(sum_11, 1);
//sum2 += vgetq_lane_u32(sum_11, 2);
//sum2 += vgetq_lane_u32(sum_11, 3);

sumA = vgetq_lane_u32(sum_11, 0);
sumB = vgetq_lane_u32(sum_11, 1);
sumC = vgetq_lane_u32(sum_11, 2);
sumD = vgetq_lane_u32(sum_11, 3);


//printf("%d ",sum2);
sum = sumA + sumB + sumC + sumD;
//if( 25 == u && 5 == v){
//    for(int i = 0; i < 16; i++)
//        printf("%d ", *(I1_block_addr + desc_offset_1 + i));
//    for(int i = 0; i < 16; i++)
//        printf("%d ", *(I2_block_addr + desc_offset_1 + i));
//    printf("\n");

//    printf("%d + %d + %d + %d = %d\n", sumA, sumB, sumC, sumD, sum );
//}

//int16_t min_1_E = 32767;
//int16_t min_1_d = -1;
//int16_t min_2_E = 32767;
//int16_t min_2_d = -1;

        // best + second best match
        if (sum<min_1_E) {
          min_2_E = min_1_E;
          min_2_d = min_1_d;
          min_1_E = sum;
          min_1_d = d;
        } else if (sum<min_2_E) {
          min_2_E = sum;
          min_2_d = d;
        }
//        if( 25 == u && 5 == v)
//        printf("aaaaaaa: %d, %d, %d, %d\n", min_1_d, min_2_d, min_1_E,  min_2_E);

      }
//      if( 25 == u && 5 == v)
//printf("end aa: %d, %d, %f, %f\n", min_1_d, min_2_d, min_1_E,  (float)min_2_E);
      // check if best and second best match are available and if matching ratio is sufficient
      if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<param.support_threshold*(float)min_2_E)
        return min_1_d;
      else
        return -1;

    } else
      return -1;

    //return -1;
}

/*

inline int16_t Elas::computeMatchingDisparity (const int32_t &u,const int32_t &v,uint8_t* I1_desc,uint8_t* I2_desc,const bool &right_image) {
  
  const int32_t u_step      = 2;
  const int32_t v_step      = 2;
  const int32_t window_size = 3;
  
  int32_t desc_offset_1 = -16*u_step-16*width*v_step;
  int32_t desc_offset_2 = +16*u_step-16*width*v_step;
  int32_t desc_offset_3 = -16*u_step+16*width*v_step;
  int32_t desc_offset_4 = +16*u_step+16*width*v_step;
  
  __m128i xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;

  // check if we are inside the image region
  if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {
    
    // compute desc and start addresses
    int32_t  line_offset = 16*width*v;
    uint8_t *I1_line_addr,*I2_line_addr;
    if (!right_image) {
      I1_line_addr = I1_desc+line_offset;
      I2_line_addr = I2_desc+line_offset;
    } else {
      I1_line_addr = I2_desc+line_offset;
      I2_line_addr = I1_desc+line_offset;
    }

    // compute I1 block start addresses
    uint8_t* I1_block_addr = I1_line_addr+16*u;
    uint8_t* I2_block_addr;
    
    // we require at least some texture
    int32_t sum = 0;
    for (int32_t i=0; i<16; i++)
      sum += abs((int32_t)(*(I1_block_addr+i))-128);
    if (sum<param.support_texture)
      return -1;
    
    // load first blocks to xmm registers
    xmm1 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_1));
    xmm2 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_2));
    xmm3 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_3));
    xmm4 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_4));
    
    // declare match energy for each disparity
    int32_t u_warp;
    
    // best match
    int16_t min_1_E = 32767;
    int16_t min_1_d = -1;
    int16_t min_2_E = 32767;
    int16_t min_2_d = -1;

    // get valid disparity range
    int32_t disp_min_valid = max(param.disp_min,0);
    int32_t disp_max_valid = param.disp_max;
    if (!right_image) disp_max_valid = min(param.disp_max,u-window_size-u_step);
    else              disp_max_valid = min(param.disp_max,width-u-window_size-u_step);
    
    // assume, that we can compute at least 10 disparities for this pixel
    if (disp_max_valid-disp_min_valid<10)
      return -1;

    // for all disparities do
    for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {

      // warp u coordinate
      if (!right_image) u_warp = u-d;
      else              u_warp = u+d;

      // compute I2 block start addresses
      I2_block_addr = I2_line_addr+16*u_warp;

      // compute match energy at this disparity
      xmm6 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_1));
      xmm6 = _mm_sad_epu8(xmm1,xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_2));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm2,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_3));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm3,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_4));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm4,xmm5),xmm6);
      sum  = _mm_extract_epi16(xmm6,0)+_mm_extract_epi16(xmm6,4);

      // best + second best match
      if (sum<min_1_E) {
        min_2_E = min_1_E;   
        min_2_d = min_1_d;
        min_1_E = sum;
        min_1_d = d;
      } else if (sum<min_2_E) {
        min_2_E = sum;
        min_2_d = d;
      }
    }

    // check if best and second best match are available and if matching ratio is sufficient
    if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<param.support_threshold*(float)min_2_E)
      return min_1_d;
    else
      return -1;
    
  } else
    return -1;
}
*/

/*****************************************************************************/
/*******************opencl����������������������������************************/
/*****************************************************************************/
/*
vector<Elas::support_pt> Elas::cl_sobelandcomputeSupportMatches (uint8_t* I1_desc,uint8_t* I2_desc)
{
	cl_uint numPlatforms = 0;           //the NO. of platforms
        cl_platform_id platform = nullptr;  //the chosen platform
        cl_context context = nullptr;       // OpenCL context
        cl_command_queue commandQueue = nullptr;
        cl_program program = nullptr;       // OpenCL kernel program object that'll be running on the compute device
        cl_mem input1MemObj = nullptr;      // input1 memory object for input argument 1
        cl_mem input2MemObj = nullptr;      // input2 memory object for input argument 2
        cl_mem outputMemObj = nullptr;      // output memory object for output
        cl_kernel kernel = nullptr;         // kernel object

        cl_int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
        if (status != CL_SUCCESS)
        {
            cout<<"Error: Getting platforms!"<<endl;
			exit(0);
		//	return NULL;
        }

//        /For clarity, choose the first available platform.
        if(numPlatforms > 0)
        {
            cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
            status = clGetPlatformIDs(numPlatforms, platforms, NULL);
            platform = platforms[0];
            free(platforms);
        }
        else
        {
            puts("Your system does not have any OpenCL platform!");
            exit(0);
			//return 0;
        }

//        Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.
        cl_uint                numDevices = 0;
        cl_device_id        *devices;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);    
        if (numDevices == 0) //no GPU available.
        {
            cout << "No GPU device available."<<endl;
            cout << "Choose CPU as default device."<<endl;
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);    
            devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
        }
        else
        {
            devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
            status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
            cout << "The number of devices: " << numDevices << endl;
        }

//        Step 3: Create context.
        context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);

//        Step 4: Creating command queue associate with the context.
        commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

//        Step 5: Create program object
        // Read the kernel code to the buffer
        FILE *fp = fopen("cl_kernel_sobelandmatch.cl", "rb");
        if(fp == nullptr)
        {
            puts("The kernel file not found!");
            goto RELEASE_RESOURCES;
        }
        fseek(fp, 0, SEEK_END);
        size_t kernelLength = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        char *kernelCodeBuffer = (char*)malloc(kernelLength + 1);
        fread(kernelCodeBuffer, 1, kernelLength, fp);
        kernelCodeBuffer[kernelLength] = '\0';
        fclose(fp);
        
        const char *aSource = kernelCodeBuffer;
        program = clCreateProgramWithSource(context, 1, &aSource, &kernelLength, NULL);

//        Step 6: Build program.
        status = clBuildProgram(program, 1,devices,NULL,NULL,NULL);

//        Step 7: Initial inputs and output for the host and create memory objects for the kernel
        int __declspec(align(32)) input1Buffer[128];    // 32 bytes alignment to improve data copy
        int __declspec(align(32)) input2Buffer[128];
        int __declspec(align(32)) outputBuffer[128];

        // Do initialization
        int i;
        for(i = 0; i < 128; i++)
            input1Buffer[i] = input2Buffer[i] = i + 1;
        memset(outputBuffer, 0, sizeof(outputBuffer));

        // Create mmory object
        input1MemObj = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 128 * sizeof(int), input1Buffer, nullptr);
        input2MemObj = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 128 * sizeof(int), input2Buffer, nullptr);
        outputMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 128 * sizeof(int), NULL, NULL);

//        Step 8: Create kernel object
        kernel = clCreateKernel(program,"sobelandmatch", NULL);

//        Step 9: Sets Kernel arguments.
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputMemObj);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&input1MemObj);
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&input2MemObj);

//        Step 10: Running the kernel.
        size_t global_work_size[1] = { 128 };
        status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
        clFinish(commandQueue);     // Force wait until the OpenCL kernel is completed

//        Step 11: Read the cout put back to host memory.
        status = clEnqueueReadBuffer(commandQueue, outputMemObj, CL_TRUE, 0, global_work_size[0] * sizeof(int), outputBuffer, 0, NULL, NULL);

RELEASE_RESOURCES:
//        Step 12: Clean the resources.
        status = clReleaseKernel(kernel);//*Release kernel.
        status = clReleaseProgram(program);    //Release the program object.
        status = clReleaseMemObject(input1MemObj);//Release mem object.
        status = clReleaseMemObject(input2MemObj);
        status = clReleaseMemObject(outputMemObj);
        status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
        status = clReleaseContext(context);//Release context.

        free(devices);
}

*/

vector<Elas::support_pt> Elas::computeSupportMatches (uint8_t* I1_desc,uint8_t* I2_desc) {
  
  // be sure that at half resolution we only need data
  // from every second line!
  int32_t D_candidate_stepsize = param.candidate_stepsize;
  if (param.subsampling)
    D_candidate_stepsize += D_candidate_stepsize%2;

  // create matrix for saving disparity candidates
  int32_t D_can_width  = 0;
  int32_t D_can_height = 0;
  for (int32_t u=0; u<width;  u+=D_candidate_stepsize) D_can_width++; //64,
  for (int32_t v=0; v<height; v+=D_candidate_stepsize) D_can_height++; //48
//  printf("D_can_width %d %d\n", D_can_width, D_can_height);

  int16_t* D_can = (int16_t*)calloc(D_can_width*D_can_height,sizeof(int16_t));

  // loop variables
  int32_t u,v;
  int16_t d,d2;

  int count_dGT0 = 0;
  int count_d2GT0 = 0;
//  clock_t t1, t2;
//  t1 = clock();
  // for all point candidates in image 1 do
  for (int32_t u_can=1; u_can<D_can_width; u_can++) {
    u = u_can*D_candidate_stepsize;

    for (int32_t v_can=1; v_can<D_can_height; v_can++) {
      v = v_can*D_candidate_stepsize;
      
      // initialize disparity candidate to invalid
      *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;

      // find forwards
      d = computeMatchingDisparity(u,v,I1_desc,I2_desc,false);
      if( 310 == u && 15 == v){
          printf("dddd: %d\n", d);
      }
      if (d>=0) {

//                    count_dGT0++;
        // find backwards
        d2 = computeMatchingDisparity(u-d,v,I1_desc,I2_desc,true);
        if( 310 == u && 15 == v){
            printf("dddd2: %d\n", d2);
        }
        if (d2>=0 && abs(d-d2)<=param.lr_threshold)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = d;
//        else
//            count_d2GT0++;
      }
    }
//        printf("\n");
  }
//int count_D = 0;
//  for(int i = 0; i < D_can_height; i++){
//      printf("%d: ", i);
//      for(int j = 0; j < D_can_width; j++){
//          if(D_can[i * D_can_width + j] >= 0){
//              count_D++;
//              printf("%d ", j);
//          }
//      }
//      printf("\n");
//  }

//  printf("count : %d, %d, \n", count_dGT0, count_d2GT0);


//  t2=clock();
//  cout <<"for: "<< (t2-t1)/CLK_TCK << "ms" <<endl;

  // remove inconsistent support points
//  removeInconsistentSupportPoints(D_can,D_can_width,D_can_height);
  
  // remove support points on straight lines, since they are redundant
  // this reduces the number of triangles a little bit and hence speeds up
  // the triangulation process
//  removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,true);
//  removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,false);
  
  // move support points from image representation into a vector representation
  vector<support_pt> p_support;
  for (int32_t u_can=1; u_can<D_can_width; u_can++)
    for (int32_t v_can=1; v_can<D_can_height; v_can++)
      if (*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))>=0)
        p_support.push_back(support_pt(u_can*D_candidate_stepsize,
                                       v_can*D_candidate_stepsize,
                                       *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))));
  
  // if flag is set, add support points in image corners
  // with the same disparity as the nearest neighbor support point
  if (param.add_corners)
    addCornerSupportPoints(p_support);

  // free memory
  free(D_can);
  
  // return support point vector
  return p_support; 
}

vector<Elas::triangle> Elas::computeDelaunayTriangulation (vector<support_pt> p_support,int32_t right_image) {

  // input/output structure for triangulation
  struct triangulateio in, out;
  int32_t k;

  // inputs
  in.numberofpoints = p_support.size();
  in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float));
  k=0;
  if (!right_image) {
    for (int32_t i=0; i<p_support.size(); i++) {
      in.pointlist[k++] = p_support[i].u;
      in.pointlist[k++] = p_support[i].v;
    }
  } else {
    for (int32_t i=0; i<p_support.size(); i++) {
      in.pointlist[k++] = p_support[i].u-p_support[i].d;
      in.pointlist[k++] = p_support[i].v;
    }
  }
  in.numberofpointattributes = 0;
  in.pointattributelist      = NULL;
  in.pointmarkerlist         = NULL;
  in.numberofsegments        = 0;
  in.numberofholes           = 0;
  in.numberofregions         = 0;
  in.regionlist              = NULL;
  
  // outputs
  out.pointlist              = NULL;
  out.pointattributelist     = NULL;
  out.pointmarkerlist        = NULL;
  out.trianglelist           = NULL;
  out.triangleattributelist  = NULL;
  out.neighborlist           = NULL;
  out.segmentlist            = NULL;
  out.segmentmarkerlist      = NULL;
  out.edgelist               = NULL;
  out.edgemarkerlist         = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
  char parameters[] = "zQB";
  triangulate(parameters, &in, &out, NULL);
  
  // put resulting triangles into vector tri
  vector<triangle> tri;
  k=0;
  for (int32_t i=0; i<out.numberoftriangles; i++) {
    tri.push_back(triangle(out.trianglelist[k],out.trianglelist[k+1],out.trianglelist[k+2]));
    k+=3;
  }
  
  // free memory used for triangulation
  free(in.pointlist);
  free(out.pointlist);
  free(out.trianglelist);
  
  // return triangles
  return tri;
}

void Elas::computeDisparityPlanes (vector<support_pt> p_support, vector<triangle> &tri, int32_t right_image) {

  // init matrices
  Matrix A(3,3);
  Matrix b(3,1);
  printf("tri.size: %d\n", tri.size());
  // for all triangles do
  for (int32_t i=0; i<tri.size(); i++) {
    
    // get triangle corner indices
    int32_t c1 = tri[i].c1;
    int32_t c2 = tri[i].c2;
    int32_t c3 = tri[i].c3;
    
    // compute matrix A for linear system of left triangle
    A.val[0][0] = p_support[c1].u;
    A.val[1][0] = p_support[c2].u;
    A.val[2][0] = p_support[c3].u;
    A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
    A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
    A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;
    
    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = p_support[c1].d;
    b.val[1][0] = p_support[c2].d;
    b.val[2][0] = p_support[c3].d;
    
    // on success of gauss jordan elimination
    if (b.solve(A)) {
      
      // grab results from b
      tri[i].t1a = b.val[0][0];
      tri[i].t1b = b.val[1][0];
      tri[i].t1c = b.val[2][0];
      
    // otherwise: invalid
    } else {
      tri[i].t1a = 0;
      tri[i].t1b = 0;
      tri[i].t1c = 0;
    }
//	cout<<"left:"<<tri[i].t1a;
    // compute matrix A for linear system of right triangle
    A.val[0][0] = p_support[c1].u-p_support[c1].d;
    A.val[1][0] = p_support[c2].u-p_support[c2].d;
    A.val[2][0] = p_support[c3].u-p_support[c3].d;
    A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
    A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
    A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;
    
    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = p_support[c1].d;
    b.val[1][0] = p_support[c2].d;
    b.val[2][0] = p_support[c3].d;
    
    // on success of gauss jordan elimination
    if (b.solve(A)) {
      
      // grab results from b
      tri[i].t2a = b.val[0][0];
      tri[i].t2b = b.val[1][0];
      tri[i].t2c = b.val[2][0];
      
    // otherwise: invalid
    } else {
      tri[i].t2a = 0;
      tri[i].t2b = 0;
      tri[i].t2c = 0;
    }
//	cout<<"right:"<<tri[i].t2a<<endl;
  }  
}

void Elas::createGrid(vector<support_pt> p_support,int32_t* disparity_grid,int32_t* grid_dims,bool right_image) {
  
  // get grid dimensions
  int32_t grid_width  = grid_dims[1];
  int32_t grid_height = grid_dims[2];
  
  // allocate temporary memory
  int32_t* temp1 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
  int32_t* temp2 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
  
  // for all support points do
  for (int32_t i=0; i<p_support.size(); i++) {
    
    // compute disparity range to fill for this support point
    int32_t x_curr = p_support[i].u;
    int32_t y_curr = p_support[i].v;
    int32_t d_curr = p_support[i].d;
    int32_t d_min  = max(d_curr-1,0);
    int32_t d_max  = min(d_curr+1,param.disp_max);
    
    // fill disparity grid helper
    for (int32_t d=d_min; d<=d_max; d++) {
      int32_t x;
      if (!right_image)
        x = floor((float)(x_curr/param.grid_size));
      else
        x = floor((float)(x_curr-d_curr)/(float)param.grid_size);
      int32_t y = floor((float)y_curr/(float)param.grid_size);
      
      // point may potentially lay outside (corner points)
      if (x>=0 && x<grid_width &&y>=0 && y<grid_height) {
        int32_t addr = getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1);
        *(temp1+addr) = 1;
      }
    }
  }
  
  // diffusion pointers
  const int32_t* tl = temp1 + (0*grid_width+0)*(param.disp_max+1);
  const int32_t* tc = temp1 + (0*grid_width+1)*(param.disp_max+1);
  const int32_t* tr = temp1 + (0*grid_width+2)*(param.disp_max+1);
  const int32_t* cl = temp1 + (1*grid_width+0)*(param.disp_max+1);
  const int32_t* cc = temp1 + (1*grid_width+1)*(param.disp_max+1);
  const int32_t* cr = temp1 + (1*grid_width+2)*(param.disp_max+1);
  const int32_t* bl = temp1 + (2*grid_width+0)*(param.disp_max+1);
  const int32_t* bc = temp1 + (2*grid_width+1)*(param.disp_max+1);
  const int32_t* br = temp1 + (2*grid_width+2)*(param.disp_max+1);
  
  int32_t* result    = temp2 + (1*grid_width+1)*(param.disp_max+1); 
  int32_t* end_input = temp1 + grid_width*grid_height*(param.disp_max+1);
  
  // diffuse temporary grid
  for( ; br != end_input; tl++, tc++, tr++, cl++, cc++, cr++, bl++, bc++, br++, result++ )
    *result = *tl | *tc | *tr | *cl | *cc | *cr | *bl | *bc | *br;
  
  // for all grid positions create disparity grid
  for (int32_t x=0; x<grid_width; x++) {
    for (int32_t y=0; y<grid_height; y++) {
        
      // start with second value (first is reserved for count)
      int32_t curr_ind = 1;
      
      // for all disparities do
      for (int32_t d=0; d<=param.disp_max; d++) {

        // if yes => add this disparity to current cell
        if (*(temp2+getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1))>0) {
          *(disparity_grid+getAddressOffsetGrid(x,y,curr_ind,grid_width,param.disp_max+2))=d;
          curr_ind++;
        }
      }
      
      // finally set number of indices
      *(disparity_grid+getAddressOffsetGrid(x,y,0,grid_width,param.disp_max+2))=curr_ind-1;
    }
  }
  
  // release temporary memory
  free(temp1);
  free(temp2);
}
//inline void Elas::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
//                                         const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
//}
//inline void Elas::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,
//                                         const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
//}
/*
inline void Elas::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,const int32_t &w,
                                         const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
  xmm2 = _mm_load_si128(I2_block_addr);
  xmm2 = _mm_sad_epu8(xmm1,xmm2);
  val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4)+w;
  if (val<min_val) {
    min_val = val;
    min_d   = d;
  }
}

inline void Elas::updatePosteriorMinimum(__m128i* I2_block_addr,const int32_t &d,
                                         const __m128i &xmm1,__m128i &xmm2,int32_t &val,int32_t &min_val,int32_t &min_d) {
  xmm2 = _mm_load_si128(I2_block_addr);
  xmm2 = _mm_sad_epu8(xmm1,xmm2);
  val  = _mm_extract_epi16(xmm2,0)+_mm_extract_epi16(xmm2,4);
  if (val<min_val) {
    min_val = val;
    min_d   = d;
  }
}
*/
inline void Elas::findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
                            int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                            int32_t *P,int32_t &plane_radius,bool &valid,bool &right_image,float* D){
 return ;
}


/*
inline void Elas::findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
                            int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                            int32_t *P,int32_t &plane_radius,bool &valid,bool &right_image,float* D){
  
  // get image width and height
  const int32_t disp_num    = grid_dims[0]-1;
  const int32_t window_size = 2;

  // address of disparity we want to compute
  uint32_t d_addr;
  if (param.subsampling) d_addr = getAddressOffsetImage(u/2,v/2,width/2);
  else                   d_addr = getAddressOffsetImage(u,v,width);
  
  // check if u is ok
  if (u<window_size || u>=width-window_size)
    return;

  // compute line start address
  int32_t  line_offset = 16*width*max(min(v,height-3),2);
  uint8_t *I1_line_addr,*I2_line_addr;
  if (!right_image) {
    I1_line_addr = I1_desc+line_offset;
    I2_line_addr = I2_desc+line_offset;
  } else {
    I1_line_addr = I2_desc+line_offset;
    I2_line_addr = I1_desc+line_offset;
  }

  // compute I1 block start address
  uint8_t* I1_block_addr = I1_line_addr+16*u;
  
  // does this patch have enough texture?
  int32_t sum = 0;
  for (int32_t i=0; i<16; i++)
    sum += abs((int32_t)(*(I1_block_addr+i))-128);
  if (sum<param.match_texture)
    return;

  // compute disparity, min disparity and max disparity of plane prior
  int32_t d_plane     = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
  int32_t d_plane_min = max(d_plane-plane_radius,0);
  int32_t d_plane_max = min(d_plane+plane_radius,disp_num-1);

  // get grid pointer
  int32_t  grid_x    = (int32_t)floor((float)u/(float)param.grid_size);
  int32_t  grid_y    = (int32_t)floor((float)v/(float)param.grid_size);
  uint32_t grid_addr = getAddressOffsetGrid(grid_x,grid_y,0,grid_dims[1],grid_dims[0]);  
  int32_t  num_grid  = *(disparity_grid+grid_addr);
  int32_t* d_grid    = disparity_grid+grid_addr+1;
  
  // loop variables
  int32_t d_curr, u_warp, val;
  int32_t min_val = 10000;
  int32_t min_d   = -1;
  __m128i xmm1    = _mm_load_si128((__m128i*)I1_block_addr);
  __m128i xmm2;

  // left image
  if (!right_image) { 
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u-d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u-d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
    }
    
  // right image
  } else {
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u+d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,xmm1,xmm2,val,min_val,min_d);
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u+d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      updatePosteriorMinimum((__m128i*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,xmm1,xmm2,val,min_val,min_d);
    }
  }

  // set disparity value
  if (min_d>=0) *(D+d_addr) = min_d; // MAP value (min neg-Log probability)
  else          *(D+d_addr) = -1;    // invalid disparity
}

*/

// TODO: %2 => more elegantly
void Elas::computeDisparity(vector<support_pt> p_support,vector<triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
                            uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D) {

  // number of disparities
  const int32_t disp_num  = grid_dims[0]-1;
  
  // descriptor window_size
  int32_t window_size = 2;
  
  // init disparity image to -10
  if (param.subsampling) {
    for (int32_t i=0; i<(width/2)*(height/2); i++)
      *(D+i) = -10;
  } else {
    for (int32_t i=0; i<width*height; i++)
      *(D+i) = -10;
  }
  
  // pre-compute prior 
  float two_sigma_squared = 2*param.sigma*param.sigma;
  int32_t* P = new int32_t[disp_num];
  for (int32_t delta_d=0; delta_d<disp_num; delta_d++)
    P[delta_d] = (int32_t)((-log(param.gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(param.gamma))/param.beta);
  int32_t plane_radius = (int32_t)max((float)ceil(param.sigma*param.sradius),(float)2.0);

  // loop variables
  int32_t c1, c2, c3;
  float plane_a,plane_b,plane_c,plane_d;
  
  // for all triangles do
  for (uint32_t i=0; i<tri.size(); i++) {
    
    // get plane parameters
    uint32_t p_i = i*3;
    if (!right_image) {    //��ͼ��
      plane_a = tri[i].t1a;
      plane_b = tri[i].t1b;
      plane_c = tri[i].t1c;
      plane_d = tri[i].t2a;
    } else {               //��ͼ��
      plane_a = tri[i].t2a;
      plane_b = tri[i].t2b;
      plane_c = tri[i].t2c;
      plane_d = tri[i].t1a;
    }
    
    // triangle corners
    c1 = tri[i].c1;
    c2 = tri[i].c2;
    c3 = tri[i].c3;

    // sort triangle corners wrt. u (ascending)    
    float tri_u[3];
    if (!right_image) {     //��ͼ��
      tri_u[0] = p_support[c1].u;
      tri_u[1] = p_support[c2].u;
      tri_u[2] = p_support[c3].u;
    } else {                //��ͼ��
      tri_u[0] = p_support[c1].u-p_support[c1].d;
      tri_u[1] = p_support[c2].u-p_support[c2].d;
      tri_u[2] = p_support[c3].u-p_support[c3].d;
    }
    float tri_v[3] = {p_support[c1].v,p_support[c2].v,p_support[c3].v};
    
    for (uint32_t j=0; j<3; j++) {
      for (uint32_t k=0; k<j; k++) {
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
    if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
    if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
    if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
    float AB_b = A_v-AB_a*A_u;
    float AC_b = A_v-AC_a*A_u;
    float BC_b = B_v-BC_a*B_u;
    
    // a plane is only valid if itself and its projection
    // into the other image is not too much slanted
    bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;
        
    // first part (triangle corner A->B)
    if ((int32_t)(A_u)!=(int32_t)(B_u)) {
      for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++){
        if (!param.subsampling || u%2==0) {
          int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
          int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
          for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
            if (!param.subsampling || v%2==0) 
			{   
				//if((!right_image && panLL[v*width+u]==1) || (right_image && panRR[v*width+u]==1))
				{
                    findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
					//if(!right_image) depthDataL[v*width+u]=D[v*width+u];
					//if(right_image)  depthDataR[v*width+u]=D[v*width+u];
				}
				/*else
				{    if(!right_image)  D[v*width+u]=depthDataL[v*width+u];
				     if(right_image)   D[v*width+u]=depthDataR[v*width+u];
				}*/
			
				
            }
        }
      }
    }

    // second part (triangle corner B->C)
    if ((int32_t)(B_u)!=(int32_t)(C_u)) {
      for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++){
        if (!param.subsampling || u%2==0) {
          int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
          int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
          for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
            if (!param.subsampling || v%2==0) 
			{
				//if((!right_image && panLL[v*width+u]==1) || (right_image && panRR[v*width+u]==1))
				{
                   findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,I1_desc,I2_desc,P,plane_radius,valid,right_image,D);
				   //if(!right_image) depthDataL[v*width+u]=D[v*width+u];
				   //if(right_image)  depthDataR[v*width+u]=D[v*width+u];
				}
				/*else
				{
				   if(!right_image)  D[v*width+u]=depthDataL[v*width+u];
				   if(right_image)   D[v*width+u]=depthDataR[v*width+u];
				}*/
				
            }
        }
      }
    }
    
  }

  delete[] P;
}

void Elas::leftRightConsistencyCheck(float* D1,float* D2) {
  
  // get disparity image dimensions
  int32_t D_width  = width;
  int32_t D_height = height;
  if (param.subsampling) {
    D_width  = width/2;
    D_height = height/2;
  }
  
  // make a copy of both images
  float* D1_copy = (float*)malloc(D_width*D_height*sizeof(float));
  float* D2_copy = (float*)malloc(D_width*D_height*sizeof(float));
  memcpy(D1_copy,D1,D_width*D_height*sizeof(float));
  memcpy(D2_copy,D2,D_width*D_height*sizeof(float));

  // loop variables
  uint32_t addr,addr_warp;
  float    u_warp_1,u_warp_2,d1,d2;
  
  // for all image points do
  for (int32_t u=0; u<D_width; u++) {
    for (int32_t v=0; v<D_height; v++) {
      
      // compute address (u,v) and disparity value
      addr     = getAddressOffsetImage(u,v,D_width);
      d1       = *(D1_copy+addr);
      d2       = *(D2_copy+addr);
      if (param.subsampling) {
        u_warp_1 = (float)u-d1/2;
        u_warp_2 = (float)u+d2/2;
      } else {
        u_warp_1 = (float)u-d1;
        u_warp_2 = (float)u+d2;
      }
      
      
      // check if left disparity is valid
      if (d1>=0 && u_warp_1>=0 && u_warp_1<D_width) {       
                  
        // compute warped image address
        addr_warp = getAddressOffsetImage((int32_t)u_warp_1,v,D_width);

        // if check failed
        if (fabs(*(D2_copy+addr_warp)-d1)>param.lr_threshold)
          *(D1+addr) = -10;
        
      // set invalid
      } else
        *(D1+addr) = -10;
      
      // check if right disparity is valid
      if (d2>=0 && u_warp_2>=0 && u_warp_2<D_width) {       

        // compute warped image address
        addr_warp = getAddressOffsetImage((int32_t)u_warp_2,v,D_width);

        // if check failed
        if (fabs(*(D1_copy+addr_warp)-d2)>param.lr_threshold)
          *(D2+addr) = -10;
        
      // set invalid
      } else
        *(D2+addr) = -10;
    }
  }
  
  // release memory
  free(D1_copy);
  free(D2_copy);
}

void Elas::removeSmallSegments (float* D) {
  
  // get disparity image dimensions
  int32_t D_width        = width;
  int32_t D_height       = height;
  int32_t D_speckle_size = param.speckle_size;
  if (param.subsampling) {
    D_width        = width/2;
    D_height       = height/2;
    D_speckle_size = sqrt((float)param.speckle_size)*2;
  }
  
  // allocate memory on heap for dynamic programming arrays
  int32_t *D_done     = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t *seg_list_u = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t *seg_list_v = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t seg_list_count;
  int32_t seg_list_curr;
  int32_t u_neighbor[4];
  int32_t v_neighbor[4];
  int32_t u_seg_curr;
  int32_t v_seg_curr;
  
  // declare loop variables
  int32_t addr_start, addr_curr, addr_neighbor;
  
  // for all pixels do
  for (int32_t u=0; u<D_width; u++) {
    for (int32_t v=0; v<D_height; v++) {
      
      // get address of first pixel in this segment
      addr_start = getAddressOffsetImage(u,v,D_width);
                  
      // if this pixel has not already been processed
      if (*(D_done+addr_start)==0) {
                
        // init segment list (add first element
        // and set it to be the next element to check)
        *(seg_list_u+0) = u;
        *(seg_list_v+0) = v;
        seg_list_count  = 1;
        seg_list_curr   = 0;
        
        // add neighboring segments as long as there
        // are none-processed pixels in the seg_list;
        // none-processed means: seg_list_curr<seg_list_count
        while (seg_list_curr<seg_list_count) {
        
          // get current position from seg_list
          u_seg_curr = *(seg_list_u+seg_list_curr);
          v_seg_curr = *(seg_list_v+seg_list_curr);
          
          // get address of current pixel in this segment
          addr_curr = getAddressOffsetImage(u_seg_curr,v_seg_curr,D_width);
          
          // fill list with neighbor positions
          u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
          u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
          u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
          u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;
          
          // for all neighbors do
          for (int32_t i=0; i<4; i++) {
            
            // check if neighbor is inside image
            if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<D_width && v_neighbor[i]<D_height) {
              
              // get neighbor pixel address
              addr_neighbor = getAddressOffsetImage(u_neighbor[i],v_neighbor[i],D_width);
              
              // check if neighbor has not been added yet and if it is valid
              if (*(D_done+addr_neighbor)==0 && *(D+addr_neighbor)>=0) {

                // is the neighbor similar to the current pixel
                // (=belonging to the current segment)
                if (fabs(*(D+addr_curr)-*(D+addr_neighbor))<=param.speckle_sim_threshold) {
                  
                  // add neighbor coordinates to segment list
                  *(seg_list_u+seg_list_count) = u_neighbor[i];
                  *(seg_list_v+seg_list_count) = v_neighbor[i];
                  seg_list_count++;            
                  
                  // set neighbor pixel in I_done to "done"
                  // (otherwise a pixel may be added 2 times to the list, as
                  //  neighbor of one pixel and as neighbor of another pixel)
                  *(D_done+addr_neighbor) = 1;
                }
              }
              
            } 
          }
          
          // set current pixel in seg_list to "done"
          seg_list_curr++;
          
          // set current pixel in I_done to "done"
          *(D_done+addr_curr) = 1;

        } // end: while (seg_list_curr<seg_list_count)
        
        // if segment NOT large enough => invalidate pixels
        if (seg_list_count<D_speckle_size) {
          
          // for all pixels in current segment invalidate pixels
          for (int32_t i=0; i<seg_list_count; i++) {
            addr_curr = getAddressOffsetImage(*(seg_list_u+i),*(seg_list_v+i),D_width);
            *(D+addr_curr) = -10;
          }
        }
      } // end: if (*(I_done+addr_start)==0)
      
    }
  }
  
  // free memory
  free(D_done);
  free(seg_list_u);
  free(seg_list_v);
}

void Elas::gapInterpolation(float* D) {
  
  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  int32_t D_ipol_gap_width = param.ipol_gap_width;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
    D_ipol_gap_width = param.ipol_gap_width/2+1;
  }
  
  // discontinuity threshold
  float discon_threshold = 3.0;
  
  // declare loop variables   ����ѭ������
  int32_t count,addr,v_first,v_last,u_first,u_last;
  float   d1,d2,d_ipol;
  
  // 1. Row-wise:
  // for each row do
  for (int32_t v=0; v<D_height; v++) {
    
    // init counter
    count = 0;
    
    // for each element of the row do
    for (int32_t u=0; u<D_width; u++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,D_width);
      
      // if disparity valid
      if (*(D+addr)>=0) {
        
        // check if speckle is small enough
        if (count>=1 && count<=D_ipol_gap_width) {
          
          // first and last value for interpolation
          u_first = u-count;
          u_last  = u-1;
          
          // if value in range
          if (u_first>0 && u_last<D_width-1) {
            
            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u_first-1,v,D_width));
            d2 = *(D+getAddressOffsetImage(u_last+1,v,D_width));
            if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            else                              d_ipol = min(d1,d2);
            
            // set all values to d_ipol
            for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++)
              *(D+getAddressOffsetImage(u_curr,v,D_width)) = d_ipol;
          }
          
        }
        
        // reset counter
        count = 0;
      
      // otherwise increment counter
      } else {
        count++;
      }
    }
    
    // if full size disp map requested
    if (param.add_corners) {

      // extrapolate to the left
      for (int32_t u=0; u<D_width; u++) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t u2=max(u-D_ipol_gap_width,0); u2<u; u2++)
            *(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
          break;
        }
      }

      // extrapolate to the right
      for (int32_t u=D_width-1; u>=0; u--) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t u2=u; u2<=min(u+D_ipol_gap_width,D_width-1); u2++)
            *(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
          break;
        }
      }
    }
  }

  // 2. Column-wise:
  // for each column do
  for (int32_t u=0; u<D_width; u++) {
    
    // init counter
    count = 0;
    
    // for each element of the column do
    for (int32_t v=0; v<D_height; v++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,D_width);
      
      // if disparity valid
      if (*(D+addr)>=0) {
        
        // check if gap is small enough
        if (count>=1 && count<=D_ipol_gap_width) {
          
          // first and last value for interpolation
          v_first = v-count;
          v_last  = v-1;
          
          // if value in range
          if (v_first>0 && v_last<D_height-1) {
            
            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u,v_first-1,D_width));
            d2 = *(D+getAddressOffsetImage(u,v_last+1,D_width));
            if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            else                              d_ipol = min(d1,d2);
            
            // set all values to d_ipol
            for (int32_t v_curr=v_first; v_curr<=v_last; v_curr++)
              *(D+getAddressOffsetImage(u,v_curr,D_width)) = d_ipol;
          }
          
        }
        
        // reset counter
        count = 0;
      
      // otherwise increment counter
      } else {
        count++;
      }
    }

    // added extrapolation to top and bottom since bottom rows sometimes stay unlabeled...
    // DS 5/12/2014

    // if full size disp map requested
    if (param.add_corners) {

      // extrapolate towards top
      for (int32_t v=0; v<D_height; v++) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t v2=max(v-D_ipol_gap_width,0); v2<v; v2++)
            *(D+getAddressOffsetImage(u,v2,D_width)) = *(D+addr);
          break;
        }
      }

      // extrapolate towards the bottom
      for (int32_t v=D_height-1; v>=0; v--) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t v2=v; v2<=min(v+D_ipol_gap_width,D_height-1); v2++)
            *(D+getAddressOffsetImage(u,v2,D_width)) = *(D+addr);
          break;
        }
      }
    }
  }
}

// implements approximation to bilateral filtering
void Elas::adaptiveMean (float* D) {

}
/*
    // implements approximation to bilateral filtering
void Elas::adaptiveMean (float* D) {
  
  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
  }
  
  // allocate temporary memory
  float* D_copy = (float*)malloc(D_width*D_height*sizeof(float));
  float* D_tmp  = (float*)malloc(D_width*D_height*sizeof(float));
  memcpy(D_copy,D,D_width*D_height*sizeof(float));
  
  // zero input disparity maps to -10 (this makes the bilateral
  // weights of all valid disparities to 0 in this region)
  for (int32_t i=0; i<D_width*D_height; i++) {
    if (*(D+i)<0) {
      *(D_copy+i) = -10;
      *(D_tmp+i)  = -10;
    }
  }
  
  __m128 xconst0 = _mm_set1_ps(0);
  __m128 xconst4 = _mm_set1_ps(4);
  __m128 xval,xweight1,xweight2,xfactor1,xfactor2;
  
  float *val     = (float *)_mm_malloc(8*sizeof(float),16);
  float *weight  = (float*)_mm_malloc(4*sizeof(float),16);
  float *factor  = (float*)_mm_malloc(4*sizeof(float),16);
  
  // set absolute mask
  __m128 xabsmask = _mm_set1_ps(0x7FFFFFFF);
  
  // when doing subsampling: 4 pixel bilateral filter width
  if (param.subsampling) {
  
    // horizontal filter
    for (int32_t v=3; v<D_height-3; v++) {

      // init
      for (int32_t u=0; u<3; u++)
        val[u] = *(D_copy+v*D_width+u);

      // loop
      for (int32_t u=3; u<D_width; u++) {

        // set
        float val_curr = *(D_copy+v*D_width+(u-1));
        val[u%4] = *(D_copy+v*D_width+u);

        xval     = _mm_load_ps(val);      
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];
        
        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D_tmp+v*D_width+(u-1)) = d;
        }
      }
    }

    // vertical filter
    for (int32_t u=3; u<D_width-3; u++) {

      // init
      for (int32_t v=0; v<3; v++)
        val[v] = *(D_tmp+v*D_width+u);

      // loop
      for (int32_t v=3; v<D_height; v++) {

        // set
        float val_curr = *(D_tmp+(v-1)*D_width+u);
        val[v%4] = *(D_tmp+v*D_width+u);

        xval     = _mm_load_ps(val);      
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];
        
        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D+(v-1)*D_width+u) = d;
        }
      }
    }
    
  // full resolution: 8 pixel bilateral filter width
  } else {
    
  
    // horizontal filter
    for (int32_t v=3; v<D_height-3; v++) {

      // init
      for (int32_t u=0; u<7; u++)
        val[u] = *(D_copy+v*D_width+u);

      // loop
      for (int32_t u=7; u<D_width; u++) {

        // set
        float val_curr = *(D_copy+v*D_width+(u-3));
        val[u%8] = *(D_copy+v*D_width+u);

        xval     = _mm_load_ps(val);      
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        xval     = _mm_load_ps(val+4);      
        xweight2 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight2 = _mm_and_ps(xweight2,xabsmask);
        xweight2 = _mm_sub_ps(xconst4,xweight2);
        xweight2 = _mm_max_ps(xconst0,xweight2);
        xfactor2 = _mm_mul_ps(xval,xweight2);

        xweight1 = _mm_add_ps(xweight1,xweight2);
        xfactor1 = _mm_add_ps(xfactor1,xfactor2);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];
        
        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D_tmp+v*D_width+(u-3)) = d;
        }
      }
    }
  
    // vertical filter
    for (int32_t u=3; u<D_width-3; u++) {

      // init
      for (int32_t v=0; v<7; v++)
        val[v] = *(D_tmp+v*D_width+u);

      // loop
      for (int32_t v=7; v<D_height; v++) {

        // set
        float val_curr = *(D_tmp+(v-3)*D_width+u);
        val[v%8] = *(D_tmp+v*D_width+u);

        xval     = _mm_load_ps(val);      
        xweight1 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight1 = _mm_and_ps(xweight1,xabsmask);
        xweight1 = _mm_sub_ps(xconst4,xweight1);
        xweight1 = _mm_max_ps(xconst0,xweight1);
        xfactor1 = _mm_mul_ps(xval,xweight1);

        xval     = _mm_load_ps(val+4);      
        xweight2 = _mm_sub_ps(xval,_mm_set1_ps(val_curr));
        xweight2 = _mm_and_ps(xweight2,xabsmask);
        xweight2 = _mm_sub_ps(xconst4,xweight2);
        xweight2 = _mm_max_ps(xconst0,xweight2);
        xfactor2 = _mm_mul_ps(xval,xweight2);

        xweight1 = _mm_add_ps(xweight1,xweight2);
        xfactor1 = _mm_add_ps(xfactor1,xfactor2);

        _mm_store_ps(weight,xweight1);
        _mm_store_ps(factor,xfactor1);

        float weight_sum = weight[0]+weight[1]+weight[2]+weight[3];
        float factor_sum = factor[0]+factor[1]+factor[2]+factor[3];
        
        if (weight_sum>0) {
          float d = factor_sum/weight_sum;
          if (d>=0) *(D+(v-3)*D_width+u) = d;
        }
      }
    }
  }
  
  // free memory
  _mm_free(val);
  _mm_free(weight);
  _mm_free(factor);
  free(D_copy);
  free(D_tmp);
}
*/

void Elas::median (float* D) {
  
  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
  }

  // temporary memory
  float *D_temp = (float*)calloc(D_width*D_height,sizeof(float));
  
  int32_t window_size = 3;
  
  float *vals = new float[window_size*2+1];
  int32_t i,j;
  float temp;
  
  // first step: horizontal median filter
  for (int32_t u=window_size; u<D_width-window_size; u++) 
  {
    for (int32_t v=window_size; v<D_height-window_size; v++) 
	{
      if (*(D+getAddressOffsetImage(u,v,D_width))>=0) 
	  {    
        j = 0;
        for (int32_t u2=u-window_size; u2<=u+window_size; u2++) 
		{
          temp = *(D+getAddressOffsetImage(u2,v,D_width));
          i = j-1;
          while (i>=0 && *(vals+i)>temp) 
		  {
            *(vals+i+1) = *(vals+i);
            i--;
          }
          *(vals+i+1) = temp;
          j++;
        }
        *(D_temp+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
      } 
	  else 
	  {
        *(D_temp+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
      }
        
    }
  }
  
  // second step: vertical median filter
  for (int32_t u=window_size; u<D_width-window_size; u++) 
  {
    for (int32_t v=window_size; v<D_height-window_size; v++) 
	{
      if (*(D+getAddressOffsetImage(u,v,D_width))>=0) 
	  {
        j = 0;
        for (int32_t v2=v-window_size; v2<=v+window_size; v2++) 
		{
          temp = *(D_temp+getAddressOffsetImage(u,v2,D_width));
          i = j-1;
          while (i>=0 && *(vals+i)>temp) 
		  {
            *(vals+i+1) = *(vals+i);
            i--;
          }
          *(vals+i+1) = temp;
          j++;
        }
        *(D+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
      } 
	  else 
	  {
        *(D+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
      }
    }
  }
  
  free(D_temp);
  free(vals);
}
