
#include "descriptor.h"
//#include "filter.h"
//#include <emmintrin.h>
#include <arm_neon.h>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

const double CLK_TCK = 1000.0;



typedef union {int8x8x2_t __i; int16x8_t __o;} convert;

void con_8x8_2_8x16(const int8x8_t _8, int16x8_t* _16)
{
    char zero[16] = {0};
    int8x8_t zeroN = vld1_s8((const int8_t*)zero);
    int8x8x2_t temp;
    temp = vzip_s8(_8, zeroN);
    convert con = {temp};
    *_16 = con.__o;

}

void convolve_cols_3x3( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h ) {
  using namespace std;
  const int w_chunk  = w/16;

  const unsigned char* c0 = in;
  const unsigned char* c1 = in + w_chunk*1*16;
  const unsigned char* c2 = in + w_chunk*2*16;
  int16_t* result_h = out_h + w_chunk*2*8;
  int16_t* result_v = out_v + w_chunk*2*8;
  const unsigned char* end_input = in + w_chunk*h*16;

  int8x8_t tmp_conv;
  int16x8_t h0,l0, h1,l1, h2,l2;
  int16x8_t re_h0, re_h1, re_v0, re_v1;

  int16_t op1[16], op2[16];
  int loop = 0;
  int i =0;

  for( ; c2 != end_input; c0 += 16, c1+=16, c2+=16, \
          result_h+=16, result_v+=16 )
  {

      tmp_conv = vld1_s8(c0);
      con_8x8_2_8x16(tmp_conv, &h0);
      tmp_conv = vld1_s8(c0+8);
      con_8x8_2_8x16(tmp_conv, &l0);
      tmp_conv = vld1_s8(c1);
      con_8x8_2_8x16(tmp_conv, &h1);
      tmp_conv = vld1_s8(c1+8);
      con_8x8_2_8x16(tmp_conv, &l1);
      tmp_conv = vld1_s8(c2);
      con_8x8_2_8x16(tmp_conv, &h2);
      tmp_conv = vld1_s8(c2+8);
      con_8x8_2_8x16(tmp_conv, &l2);

      re_h0 = vsubq_s16(h0, h2);
      re_h1 = vsubq_s16(l0, l2);
      vst1q_s16(result_h, (re_h0));
      vst1q_s16(result_h+16, (re_h1));
      vst1q_s8((int8_t*)result_h, vreinterpretq_s8_s16(re_h0));
      vst1q_s8((int8_t*)result_h+16, vreinterpretq_s8_s16(re_h1));
//      printf("result_h: ");
//      for( i =0 ; i <  16 ; i++)
//        {
//            printf(" %d ", *(result_h + i));
//        }
//        printf("\n");


//      re_v0 = vreinterpretq_s16_s8(h0);
      re_v0 = vaddq_s16(h0, h1);
      re_v0 =  vaddq_s16(re_v0, h1);
      re_v0 = vaddq_s16(re_v0, h2);
//      re_v1 = vreinterpretq_s16_s8(l0);
      re_v1 = vaddq_s16(l0, l1);
      re_v1 = vaddq_s16(re_v1, l1);
      re_v1 = vaddq_s16(re_v1, l2);
      vst1q_s8((int8_t*)result_v, vreinterpretq_s8_s16(re_v0));
      vst1q_s8((int8_t*)result_v+16, vreinterpretq_s8_s16(re_v1));

//    printf("result_v: ");
//    for( i =0 ; i <  16 ; i++)
//      {
//          printf(" %d ", *(result_v + i));
//      }
//      printf("\n");

//      loop++;
//      if(loop>5)
//          while(1);


    }
}

void convolve_101_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h)
{
    int16_t* i0 = in;
    int16_t* i2 = in + 2;
    uint8_t* result = out + 1;
    int16_t* end_input = in + w*h;
    int16_t offs_arr[8] = {128,128,128,128,128,128,128,128};
    int16x8_t offs = vld1q_s16(offs_arr);
    uint8x8_t lo_sa, hi_sa;
    int loop = 0;
    const int blocked_loops = (w*h-2)/16;
    for( int j = 0; j != blocked_loops; j++ )
    {
        int16x8_t lo, hi, i2_register;

        //
        lo = vld1q_s16(i0);
        i2_register = vld1q_s16(i2);
        lo = vsubq_s16(lo, i2_register);
        lo = vshrq_n_s16(lo, 2);
        lo = vaddq_s16(lo, offs);

        i0 += 8;
        i2 += 8;

        hi = vld1q_s16(i0);
        i2_register = vld1q_s16(i2);
        hi = vsubq_s16(hi, i2_register);
        hi = vshrq_n_s16(hi, 2);
        hi = vaddq_s16(hi, offs);

        i0 += 8;
        i2 += 8;

        lo_sa = vqmovun_s16(lo);
        vst1_u8((uint8_t*)result, lo_sa);
        hi_sa = vqmovun_s16(hi);
        vst1_u8((uint8_t*)result + 8, hi_sa);

        result += 16;


    }

      for(; i2 < end_input; i2++, result++)
      {
          *result = ((*(i2-2) - *i2) >> 2) + 128;
      }

}


void convolve_121_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h )
{
    const int16_t* i0 = in ;
    const int16_t* i1 = in+1;
    const int16_t* i2 = in+2;
    uint8_t* result   = out + 1;
    uint8x8_t lo_sa, hi_sa;

    const int16_t* const end_input = in + w*h;
    const size_t blocked_loops = (w*h-2)/16;
    int16_t offs_arr[8] = {128,128,128,128,128,128,128,128};
    int16x8_t offs = vld1q_s16(offs_arr);

    int loop = 0;

    for( size_t i=0; i != blocked_loops; i++ )
    {
        int16x8_t lo, hi, i1_register, i2_register;

        //
        lo = vld1q_s16(i0);
        i1_register = vld1q_s16(i0+1);
        i2_register = vld1q_s16(i0+2);
        i1_register = vaddq_s16(i1_register, i1_register);
        lo = vaddq_s16(i1_register, lo);
        lo = vaddq_s16(i2_register, lo);
        lo = vshrq_n_s16(lo, 2);
        lo = vaddq_s16(lo, offs);

        i0 += 8;

        hi = vld1q_s16(i0);
        i1_register = vld1q_s16(i0+1);
        i2_register = vld1q_s16(i0+2);
        i1_register = vaddq_s16(i1_register, i1_register);
        hi = vaddq_s16(i1_register, hi);
        hi = vaddq_s16(i2_register, hi);
        hi = vshrq_n_s16(hi, 2);
        hi = vaddq_s16(hi, offs);

        i0 += 8;

        lo_sa = vqmovun_s16(lo);
        vst1_u8((uint8_t*)result, lo_sa);
        hi_sa = vqmovun_s16(hi);
        vst1_u8((uint8_t*)result + 8, hi_sa);

        result += 16;
    }

}



void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h )
{
      int16_t* temp_h = (int16_t*)( aligned_alloc( 16, w*h*sizeof( int16_t ) ) );
      int16_t* temp_v = (int16_t*)( aligned_alloc( 16, w*h*sizeof( int16_t ) ) );
      clock_t t1,t2;

      convolve_cols_3x3( in, temp_v, temp_h, w, h );    //1.35ms
      convolve_101_row_3x3_16bit(temp_v, out_v, w, h);  //0.42ms
      convolve_121_row_3x3_16bit( temp_h, out_h, w, h );    //0.57ms

      free( temp_h );
      free( temp_v );
}


extern void* HostMalWC(void **p, long size);
extern void* HostMal(void **p, long size);
extern void cudaFreeHost_cpuaa(void *p);
extern int __createDesc_gpu(uint8_t* I_desc, uint8_t* I_du, uint8_t* I_dv );


Descriptor::Descriptor(int32_t width,int32_t height,int32_t bpl) {
//        I_desc        = (uint8_t*)aligned_alloc(16, 16*width*height*sizeof(uint8_t));
    I_desc_g    = (uint8_t*) HostMal(&I_desc, 16*width*height*sizeof(uint8_t));
    I_du_g      = (uint8_t*) HostMal((void**)&I_du, bpl*height*sizeof(uint8_t));
    I_dv_g      = (uint8_t*) HostMal((void**)&I_dv, bpl*height*sizeof(uint8_t));
}

void Descriptor::compute(uint8_t* I, int32_t width,int32_t height,int32_t bpl,bool half_resolution)
{

//    clock_t t1,t2;
//    t1=clock();
    struct timeval start, end;
    double timeuse;
    gettimeofday(&start, NULL);
    sobel3x3(I,I_du,I_dv,bpl,height);
    gettimeofday(&end, NULL);
    timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "s: " << timeuse/1000 << "ms" <<endl;

    gettimeofday(&start, NULL);
    __createDesc_gpu(I_desc_g, I_du_g, I_dv_g);
//    t2 = clock();
//    cout << "createDescriptor " << (t2-t1)/CLK_TCK << "ms" <<endl;
    gettimeofday(&end, NULL);
    timeuse = 1000000* (end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec;
    cout << "d: " << timeuse/1000 << "ms" <<endl;


//    printf("after I_desc_g:\n");
//    for(int i = 32000; i < 32032; i ++)
//        printf("%d ", *(I_desc + i));
//    printf("\n");


//      free(I_du);
//      free(I_dv);
//    cudaFreeHost_cpuaa(I_du);
//    cudaFreeHost_cpuaa(I_dv);
}

extern void allocFreeCount();

Descriptor::~Descriptor() {
    cudaFreeHost_cpuaa(I_desc);
    cudaFreeHost_cpuaa(I_du);
    cudaFreeHost_cpuaa(I_dv);
 //   std::cout<<"Descriptor deconstructor" <<std::endl;
 //   allocFreeCount();

}

void Descriptor::createDescriptor (uint8_t* I_du,uint8_t* I_dv,int32_t width,int32_t height,int32_t bpl,bool half_resolution) {

  uint8_t *I_desc_curr;
  uint32_t addr_v0,addr_v1,addr_v2,addr_v3,addr_v4;

    // create filter strip
    for (int32_t v=3; v<height-3; v++) {

      addr_v2 = v*bpl;
      addr_v0 = addr_v2-2*bpl;
      addr_v1 = addr_v2-1*bpl;
      addr_v3 = addr_v2+1*bpl;
      addr_v4 = addr_v2+2*bpl;

      for (int32_t u=3; u<width-3; u++) {
        I_desc_curr = I_desc+(v*width+u)*16;
        *(I_desc_curr++) = *(I_du+addr_v0+u+0);
        *(I_desc_curr++) = *(I_du+addr_v1+u-2);
        *(I_desc_curr++) = *(I_du+addr_v1+u+0);
        *(I_desc_curr++) = *(I_du+addr_v1+u+2);
        *(I_desc_curr++) = *(I_du+addr_v2+u-1);
        *(I_desc_curr++) = *(I_du+addr_v2+u+0);
        *(I_desc_curr++) = *(I_du+addr_v2+u+0);
        *(I_desc_curr++) = *(I_du+addr_v2+u+1);
        *(I_desc_curr++) = *(I_du+addr_v3+u-2);
        *(I_desc_curr++) = *(I_du+addr_v3+u+0);
        *(I_desc_curr++) = *(I_du+addr_v3+u+2);
        *(I_desc_curr++) = *(I_du+addr_v4+u+0);

        *(I_desc_curr++) = *(I_dv+addr_v1+u+0);
        *(I_desc_curr++) = *(I_dv+addr_v2+u-1);
        *(I_desc_curr++) = *(I_dv+addr_v2+u+1);
        *(I_desc_curr++) = *(I_dv+addr_v3+u+0);
      }
    }

}



//void Descriptor::createDescriptor (uint8_t* I_du, uint8_t* I_dv, int32_t width, \
//                                   int32_t height, int32_t bpl, bool half_resolution)
//{
//      uint8_t *I_desc_curr ;
//      uint8_t *I_dv_curr, *I_du_curr;


//      for(int v = 5; v < height - 5; v++){
//          for(int u = 5; u < width - 5; u++){
//              I_du_curr = I_du + v * width + u;
//              I_dv_curr = I_dv + v * width + u;
//              I_desc_curr = I_desc + 16 * (v * width + u);

//              *(I_desc_curr + 16 * (2 * width + 0) + 0) = *I_du_curr;
//              *(I_desc_curr + 16 * (1 * width + 2) + 1) = *I_du_curr;
//              *(I_desc_curr + 16 * (1 * width + 0) + 2) = *I_du_curr;
//              *(I_desc_curr + 16 * (1 * width - 2) + 3) = *I_du_curr;
//              *(I_desc_curr + 16 * (0 * width + 1) + 4) = *I_du_curr;
//              *(I_desc_curr + 16 * (0 * width + 0) + 5) = *I_du_curr;
//              *(I_desc_curr + 16 * (0 * width + 0) + 6) = *I_du_curr;
//              *(I_desc_curr + 16 * (0 * width - 1) + 7) = *I_du_curr;
//              *(I_desc_curr + 16 * (-1 * width + 2) + 8) = *I_du_curr;
//              *(I_desc_curr + 16 * (-1 * width + 0) + 9) = *I_du_curr;
//              *(I_desc_curr + 16 * (-1 * width - 2) + 10) = *I_du_curr;
//              *(I_desc_curr + 16 * (-2 * width + 0) + 11) = *I_du_curr;

//              *(I_desc_curr + 16 * (1 * width + 0) + 12) = *I_dv_curr;
//              *(I_desc_curr + 16 * (0 * width + 1) + 13) = *I_dv_curr;
//              *(I_desc_curr + 16 * (0 * width - 1) + 14) = *I_dv_curr;
//              *(I_desc_curr + 16 * (-1 * width + 0) + 15) = *I_dv_curr;
//          }
//      }
//}























