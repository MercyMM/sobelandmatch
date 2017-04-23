kernel void left_match(constant int *ps,global int * disparity_grid,global int * grid_dims,global uchar *I1_desc,global uchar *I2_desc,global int *p_support,global int *tri,global float *tri_1,global float*tri_2,global float*D)
{
    int x = (int)get_global_id(0);  

	 int width = 715;
	 int height = 492;
	 int disp_num = grid_dims[0]-1;
	 int window_size = 2;

	 local int P[64];
	 int delta_d=0;
#pragma unroll 
	for(delta_d = 0;delta_d<64;delta_d++)
	{
		P[delta_d] = ps[delta_d];
	}
	 int plane_radius = 2;

	int c1,c2,c3;
	float plane_a,plane_b,plane_c,plane_d;

		plane_a = tri_1[x * 3 + 0];
		plane_b = tri_1[x * 3 + 1];
		plane_c = tri_1[x * 3 + 2];
		plane_d = tri_2[x * 3 + 0];


	c1 = tri[x * 3 + 0];
	c2 = tri[x * 3 + 1];
	c3 = tri[x * 3 + 2];

	float tri_u[3];
	tri_u[0] = p_support[c1 * 3 + 0];
	tri_u[1] = p_support[c2 * 3 + 0];
	tri_u[2] = p_support[c3 * 3 + 0];

	float tri_v[3] = {p_support[c1 * 3 + 1],p_support[c2 * 3 + 1],p_support[c3 * 3 + 1]};

	float tri_u_temp;
	float tri_v_temp;
	for (int j=0; j<3; j++) {
		for (int k=0; k<j; k++) {
			if (tri_u[k] > tri_u[j]) {
				tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
				tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
			}
		}
	}

	float A_u = tri_u[0]; float A_v = tri_v[0];
	float B_u = tri_u[1]; float B_v = tri_v[1];
	float C_u = tri_u[2]; float C_v = tri_v[2];

	float AB_a = 0; float AC_a = 0; float BC_a = 0;
	if ((int)(A_u)!=(int)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
	if ((int)(A_u)!=(int)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
	if ((int)(B_u)!=(int)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
	float AB_b = A_v-AB_a*A_u;
	float AC_b = A_v-AC_a*A_u;
	float BC_b = B_v-BC_a*B_u;

	bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;
	if ((int)(A_u)!=(int)(B_u)) 
	{
		for (int u=max((int)A_u,0); u<min((int)B_u,width); u++)
		{
			int v_1 = (uint)(AC_a*(float)u+AC_b);
            int v_2 = (uint)(AB_a*(float)u+AB_b);
            for (int v=min(v_1,v_2); v<max(v_1,v_2); v++) 
		    { 				
				uint d_addr;
                d_addr = v * width + u;
  

				if (u<window_size || u>=width-window_size)
					return;


				int  line_offset = 16*width*max(min(v,height-3),2);

				uchar I1[16];
#pragma unroll 
				for(int i= 0;i < 16;i++)
				{
					I1[i] = *(I1_desc + line_offset + u*16 +i);
				}
				int sum = 0;
#pragma unroll 
				for(int m = 0;m < 16;m++)
				{
					sum = sum + abs((int)I1[m] - 128);
				}

				if (sum < 1)
					continue;

				int d_plane     = (int)(plane_a*(float)u+plane_b*(float)v+plane_c);
				int d_plane_min = max(d_plane-plane_radius,0);
				int d_plane_max = min(d_plane+plane_radius,disp_num-1);


				int  grid_x    = (int)floor((float)u/20.0f);
				int  grid_y    = (int)floor((float)v/20.0f);  
				uint grid_addr = (grid_y * grid_dims[1] + grid_x) * grid_dims[0];
				int  num_grid  = disparity_grid[grid_addr];
				int  d_grid[64];
				for(int z = 0;z < num_grid; z++)
				{
					d_grid[z] = disparity_grid[grid_addr + 1 + z];	
				}
  

				int d_curr, u_warp, val;
				int min_val = 10000;
				int min_d   = -1;
				for (int i=0; i<num_grid; i++) {
					d_curr = d_grid[i];
					if (d_curr<d_plane_min || d_curr>d_plane_max) {
						u_warp = u-d_curr;
						if (u_warp<window_size || u_warp>=width-window_size)
							continue;
						val = 0;
						for(int m = 0;m < 16;m++)
						{
							val = val + abs((int)I1[m]-(int)(*(I2_desc+line_offset+u_warp*16+m)));
						}
						
						if(val < min_val)
						{
							min_val = val;
							min_d = d_curr;
						}
					}
				}
				for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) 
				{
					u_warp = u-d_curr;
					if (u_warp<window_size || u_warp>=width-window_size)
						continue;
					val = 0;
					for(int m = 0;m < 16;m++)
					{
						val = val + abs((int)I1[m]-(int)(*(I2_desc+line_offset+u_warp*16+m)));
					}
					
					if(valid)
					{
					 	val = val + P[abs(d_curr-d_plane)];
					}
					if(val < min_val)
					{
						min_val = val;
						min_d = d_curr;
					}
				}
				if(min_d>=0) *(D+d_addr) = min_d;
				else         *(D+d_addr) = -1;
			}
		}
	}
	if ((int)(B_u)!=(int)(C_u)) 
	{
		for (int u=max((int)B_u,0); u<min((int)C_u,width); u++)
		{
			int v_1 = (uint)(AC_a*(float)u+AC_b);
            int v_2 = (uint)(BC_a*(float)u+BC_b);
            for (int v=min(v_1,v_2); v<max(v_1,v_2); v++)
		    {
				uint d_addr;
                d_addr = v * width + u;
  

				if (u<window_size || u>=width-window_size)
					return;


				int  line_offset = 16*width*max(min(v,height-3),2);
				uchar I1[16];
#pragma unroll 
				for(int i= 0;i < 16;i++)
				{
					I1[i] = *(I1_desc + line_offset + u*16 +i);
				}
				int sum = 0;
#pragma unroll 
				for(int m = 0;m < 16;m++)
				{
					sum = sum + abs((int)I1[m] - 128);
				}

				if (sum < 1)
					continue;

				int d_plane     = (int)(plane_a*(float)u+plane_b*(float)v+plane_c);
				int d_plane_min = max(d_plane-plane_radius,0);
				int d_plane_max = min(d_plane+plane_radius,disp_num-1);


				int  grid_x    = (int)floor((float)u/20.0f);
				int  grid_y    = (int)floor((float)v/20.0f);  
				uint grid_addr = (grid_y * grid_dims[1] + grid_x) * grid_dims[0];
				int  num_grid  = disparity_grid[grid_addr];
				int  d_grid[64];
				for(int z = 0;z < num_grid; z++)
				{
					d_grid[z] = disparity_grid[grid_addr + 1 + z];	
				}
  

				int d_curr, u_warp, val;
				int min_val = 10000;
				int min_d   = -1;

				for (int i=0; i<num_grid; i++) {
					d_curr = d_grid[i];
					if (d_curr<d_plane_min || d_curr>d_plane_max) {
						u_warp = u-d_curr;
						if (u_warp<window_size || u_warp>=width-window_size)
							continue;
						val = 0;
						for(int m = 0;m < 16;m++)
						{
							val = val + abs((int)I1[m]-(int)(*(I2_desc+line_offset+u_warp*16+m)));
						}
						if(val < min_val)
						{
							min_val = val;
							min_d = d_curr;
						}
					}
				}
				for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
					u_warp = u-d_curr;
					if (u_warp<window_size || u_warp>=width-window_size)
						continue;
					val = 0;
					for(int m = 0;m < 16;m++)
					{
						val = val + abs((int)I1[m]-(int)(*(I2_desc+line_offset+u_warp*16+m)));
					}

					if(valid)
					{
						val = val + P[abs(d_curr-d_plane)];
					}
					if(val < min_val)
					{
						min_val = val;
						min_d = d_curr;
					}
				}

				if(min_d>=0) *(D+d_addr) = min_d;
				else         *(D+d_addr) = -1;
			}
        }
	 }
}

kernel void right_match(global int *ps,global int * disparity_grid,global int * grid_dims,global uchar *I1_desc,global uchar *I2_desc,global int *p_support,global int *tri,global float *tri_1,global float*tri_2,global float*D)
{
    int x = (int)get_global_id(0);  

	int width = 715;
	int height = 492;
	int disp_num = grid_dims[0]-1;
	int window_size = 2;

	int P[64]={0};
	int delta_d=0;
	for(delta_d = 0;delta_d<disp_num;delta_d++)
	{
		P[delta_d] = ps[delta_d];
	}
	int plane_radius = 2;

	int c1,c2,c3;
	float plane_a,plane_b,plane_c,plane_d;

		plane_a = tri_2[x * 3 + 0];
		plane_b = tri_2[x * 3 + 1];
		plane_c = tri_2[x * 3 + 2];
		plane_d = tri_1[x * 3 + 0];


	c1 = tri[x * 3 + 0];
	c2 = tri[x * 3 + 1];
	c3 = tri[x * 3 + 2];

	float tri_u[3];
		tri_u[0] = p_support[c1 * 3 + 0] - p_support[c1 * 3 + 2];
		tri_u[1] = p_support[c2 * 3 + 0] - p_support[c2 * 3 + 2];
		tri_u[2] = p_support[c3 * 3 + 0] - p_support[c3 * 3 + 2];
	
	float tri_v[3] = {p_support[c1 * 3 + 1],p_support[c2 * 3 + 1],p_support[c3 * 3 + 1]};

	for (int j=0; j<3; j++) {
      for (int k=0; k<j; k++) {
        if (tri_u[k]>tri_u[j]) {
          float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
          float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
        }
      }
    }
	float A_u = tri_u[0]; float A_v = tri_v[0];
    float B_u = tri_u[1]; float B_v = tri_v[1];
    float C_u = tri_u[2]; float C_v = tri_v[2];

    float AB_a = 0; float AC_a = 0; float BC_a = 0;
    if ((int)(A_u)!=(int)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
    if ((int)(A_u)!=(int)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
    if ((int)(B_u)!=(int)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
    float AB_b = A_v-AB_a*A_u;
    float AC_b = A_v-AC_a*A_u;
    float BC_b = B_v-BC_a*B_u;

	bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;

	if ((int)(A_u)!=(int)(B_u)) 
	{
		for (int u=max((int)A_u,0); u<min((int)B_u,width); u++)
		{
			int v_1 = (uint)(AC_a*(float)u+AC_b);
            int v_2 = (uint)(AB_a*(float)u+AB_b);
            for (int v=min(v_1,v_2); v<max(v_1,v_2); v++) 
		    { 				
				uint d_addr;
                d_addr = v * width + u;
  

				if (u<window_size || u>=width-window_size)
					return;


				int  line_offset = 16*width*max(min(v,height-3),2);
			//	int  line_offset_16 = width*max(min(v,height-3),2);
				
			//	uchar16 xmm0 = vload16(line_offset_16+u,I1_desc);

				int sum = 0;
				for(int m = 0;m < 16;m++)
				{
					sum = sum + abs((int)(*(I2_desc+line_offset+u*16+m)) - 128);
				}
				
				if (sum < 1)
					continue;

				int d_plane     = (int)(plane_a*(float)u+plane_b*(float)v+plane_c);
				int d_plane_min = max(d_plane-plane_radius,0);
				int d_plane_max = min(d_plane+plane_radius,disp_num-1);


				int  grid_x    = (int)floor((float)u/20.0f);
				int  grid_y    = (int)floor((float)v/20.0f);  
				uint grid_addr = (grid_y * grid_dims[1] + grid_x) * grid_dims[0];
				int  num_grid  = disparity_grid[grid_addr];
				int  d_grid[64];
				for(int z = 0;z < num_grid; z++)
				{
					d_grid[z] = disparity_grid[grid_addr + 1 + z];	
				}
  

				int d_curr, u_warp, val;
				int min_val = 10000;
				int min_d   = -1;
			//	uchar16 xmm1 = vload16(line_offset_16+u,I1_desc);
			//	uchar16 xmm2;
			//	uchar xm1[16];
				for (int i=0; i<num_grid; i++) {
					d_curr = d_grid[i];
					if (d_curr<d_plane_min || d_curr>d_plane_max) {
						u_warp = u+d_curr;
						if (u_warp<window_size || u_warp>=width-window_size)
							continue;
						val = 0;
						for(int m = 0;m < 16;m++)
						{
							val = val + abs((int)(*(I2_desc+line_offset+u*16+m))-(int)(*(I1_desc+line_offset+u_warp*16+m)));
						}
				
						if(val < min_val)
						{
							min_val = val;
							min_d = d_curr;
						}
					}
				}
				for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) 
				{
					u_warp = u+d_curr;
					if (u_warp<window_size || u_warp>=width-window_size)
						continue;
					val = 0;
					for(int m = 0;m < 16;m++)
					{
						val = val + abs((int)(*(I2_desc+line_offset+u*16+m))-(int)(*(I1_desc+line_offset+u_warp*16+m)));
					}
			
					if(valid)
					{
					 	val = val + P[abs(d_curr-d_plane)];
					}
					if(val < min_val)
					{
						min_val = val;
						min_d = d_curr;
					}
				}
			
				if(min_d>=0) *(D+d_addr) = min_d;
				else         *(D+d_addr) = -1;
			}
		}
	}
	if ((int)(B_u)!=(int)(C_u)) 
	{
		for (int u=max((int)B_u,0); u<min((int)C_u,width); u++)
		{
			int v_1 = (uint)(AC_a*(float)u+AC_b);
            int v_2 = (uint)(BC_a*(float)u+BC_b);
            for (int v=min(v_1,v_2); v<max(v_1,v_2); v++)
		    {
				uint d_addr;
                d_addr = v * width + u;
  

				if (u<window_size || u>=width-window_size)
					return;


				int  line_offset = 16*width*max(min(v,height-3),2);
			//	int  line_offset_16 = width*max(min(v,height-3),2);
				
				//uchar16 xmm0 = vload16(line_offset_16+u,I1_desc);

				int sum = 0;
				for(int m = 0;m < 16;m++)
				{
					sum = sum + abs((int)(*(I2_desc+line_offset+u*16+m)) - 128);
				}

				if (sum < 1)
					continue;

				int d_plane     = (int)(plane_a*(float)u+plane_b*(float)v+plane_c);
				int d_plane_min = max(d_plane-plane_radius,0);
				int d_plane_max = min(d_plane+plane_radius,disp_num-1);


				int  grid_x    = (int)floor((float)u/20.0f);
				int  grid_y    = (int)floor((float)v/20.0f);  
				uint grid_addr = (grid_y * grid_dims[1] + grid_x) * grid_dims[0];
				int  num_grid  = disparity_grid[grid_addr];
				int  d_grid[64];
				for(int z = 0;z < num_grid; z++)
				{
					d_grid[z] = disparity_grid[grid_addr + 1 + z];	
				}
  

				int d_curr, u_warp, val;
				int min_val = 10000;
				int min_d   = -1;

			//	uchar16 xmm2;
			//	uchar16 xmm1 = vload16(line_offset_16+u,I1_desc);

				for (int i=0; i<num_grid; i++) {
					d_curr = d_grid[i];
					if (d_curr<d_plane_min || d_curr>d_plane_max) {
						u_warp = u+d_curr;
						if (u_warp<window_size || u_warp>=width-window_size)
							continue;
						val = 0;
						for(int m = 0;m < 16;m++)
						{
							val = val + abs((int)(*(I2_desc+line_offset+u*16+m))-(int)(*(I1_desc+line_offset+u_warp*16+m)));
						}
			
						if(val < min_val)
						{
							min_val = val;
							min_d = d_curr;
						}
					}
				}
				for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
					u_warp = u+d_curr;
					if (u_warp<window_size || u_warp>=width-window_size)
						continue;
					val = 0;
					for(int m = 0;m < 16;m++)
					{
						val = val + abs((int)(*(I2_desc+line_offset+u*16+m))-(int)(*(I1_desc+line_offset+u_warp*16+m)));
					}
		
					if(valid)
					{
						val = val + P[abs(d_curr-d_plane)];
					}
					if(val < min_val)
					{
						min_val = val;
						min_d = d_curr;
					}
				}
				/*	if (min_d>=0) atomic_xchg(D+d_addr,min_d); 
				else          atomic_xchg(D+d_addr,-1); */ 
				if(min_d>=0) *(D+d_addr) = min_d;
				else         *(D+d_addr) = -1;
			}
        }
	 }
}