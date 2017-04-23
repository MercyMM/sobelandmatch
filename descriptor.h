
#ifndef __DESCRIPTOR_H__
#define __DESCRIPTOR_H__

// #include <tchar.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
  #include <stdint.h>
#else
  typedef __int8            int8_t;
  typedef __int16           int16_t;
  typedef __int32           int32_t;
  typedef __int64           int64_t;
  typedef unsigned __int8   uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef unsigned __int32  uint32_t;
  typedef unsigned __int64  uint64_t;
#endif

class Descriptor {
  
public:

  // constructor creates filters
  Descriptor(int32_t width,int32_t height,int32_t bpl);
  
  // deconstructor releases memory
  ~Descriptor();
    int a(){return 1;};
  void compute(uint8_t* I,int32_t width,int32_t height,int32_t bpl,bool half_resolution);

  // descriptors accessible from outside
  uint8_t* I_desc;
  uint8_t* I_desc_g;
  uint8_t *I_du, *I_dv;
  
private:

  // build descriptor I_desc from I_du and I_dv
  void createDescriptor(uint8_t* I_du,uint8_t* I_dv,int32_t width,int32_t height,int32_t bpl,bool half_resolution);

};

#endif
