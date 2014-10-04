
#define NOMINMAX
#include<optix_world.h>
#include<stdio.h>
#include <omp.h>
#include <float.h>

using optix::float2;
using optix::float3;
using optix::float4;
using optix::make_float2;
using optix::make_float3;
using optix::make_float4;
using optix::Matrix3x3;
using optix::normalize;
using optix::length;
// using optix::abs;
// using optix::min;
// using optix::max;

#undef optix::Ray

typedef float4 vec4;
//typedef float3 vec3;
//typedef float2 vec2;
typedef Matrix3x3 mat3;

struct vec2 : public optix::float2
{
	vec2(){}
	vec2(optix::float2& t){x=t.x;y=t.y;}
	float2 swizzle_xy(){return make_float2(x,y);}
};
struct vec3 : public optix::float3
{
	vec3(){}
	vec3(optix::float3& t){x=t.x;y=t.y;z=t.z;}
	float2 swizzle_xy(){return make_float2(x,y);}
};

#define xy swizzle_xy()

float min(float a,float b)
{
	return a<b?a:b;
}
float max(float a,float b)
{
	return a>b?a:b;
}
vec2 max(vec2 a,float b)
{
	return make_float2(max(a.x,b),max(a.y,b));
}
vec3 max(vec3 a,float b)
{
	return make_float3(max(a.x,b),max(a.y,b),max(a.z,b));
}
vec3 abs(vec3 a)
{
	return make_float3(fabsf(a.x),fabsf(a.y),fabsf(a.z));
}
vec2 abs(vec2 a)
{
	return make_float2(fabsf(a.x),fabsf(a.y));
}

float sign(float x)
{
	return x<0.f?-1.f:1.f;
}

__forceinline__ bool _valid(float v)
{
	return _finite(v)&&!_isnan(v);
}
__forceinline__ bool _valid(vec3 v)
{
	return _valid(v.x)&&_valid(v.y)&&_valid(v.z);
}

// global variables
vec2 resolution=make_float2(800.,450.);
float time=10.;

#define IN_PARAM(_type) _type&
#define OUT_PARAM(_type) _type&
#define SET_COL(_mat,_i,_col) (_mat).setCol((_i),(_col))

mat3 INIT_MAT3(float _m1,float _m2,float _m3,float _m4,float _m5,float _m6,float _m7,float _m8,float _m9)
{
	//float data[]={_m1,_m2,_m3,_m4,_m5,_m6,_m7,_m8,_m9};
	float data[]={_m1,_m4,_m7,_m2,_m5,_m8,_m3,_m6,_m9}; // '
	return mat3(data);
}

float rnd()
{
	return (float)rand()/(float)RAND_MAX;
}

vec4 renderMain(vec2);

bool is_little_edian()
{
	int num = 1;
	return (*(char *)&num == 1);
}
bool write_pfm(const char* filename,void* data,int width,int height,int stride,float scale)
{
	FILE* fp=fopen(filename,"wb");
	if(!fp)
	{
		printf("Error opening pfm file for writing.\n");
		return false;
	}

	fprintf(fp,"PF\n%d %d\n%f\n",width,height,is_little_edian()?-1.f:1.f);

	float3* dat=new float3[width*height];

	unsigned char* px=(unsigned char*)data;

	int pxs=width*height;
	for(int i=0;i<pxs;++i)
	{
		dat[i]=scale * (*(float3*)px);
		//fwrite(px,3*sizeof(float),1,fp);
		px+=stride;
	}
	fwrite(dat,sizeof(float3),width*height,fp);
	fclose(fp);

	delete dat;

	return true;
}

int main()
{
	char filename[]="res.pfm";

	float3* output_buf=new float3[resolution.x*resolution.y];
	float3* pres=output_buf;


	//#define DEBUG_PIXEL
#ifdef DEBUG_PIXEL
	//{{
		//int x=389,y=(int)resolution.y - 195 -1;
	for(int y=(int)resolution.y - 1-357;y<=(int)resolution.y - 1-340;++y)
	{
		for(int x=260;x<=274;++x){

			if(x==344&&y==(int)resolution.y - 1-241)
			{
				x=x;y=y;
			}
#else
#pragma omp parallel for
	for(int y=0;y<(int)resolution.y;++y)
	{
		for(int x=0;x<resolution.x;++x)
		{
#endif		

			int idx=y*(int)resolution.x+x;
			if(omp_get_thread_num()==0) printf("\r%.1f    ", omp_get_num_threads()*100.f*(float)idx/(resolution.x*resolution.y) );
			//int x=461,y=297;

			float4 res=renderMain(make_float2(x,y));
			//if(x==461&&y==297) res=make_float4(10000.,0.,0.,1.);
			output_buf[idx]=make_float3(res);
		}
	}

	write_pfm(filename,output_buf,resolution.x,resolution.y,sizeof(float3),1.f);

	delete [] output_buf;

	return 0;
}
