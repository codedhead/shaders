#define _CODE_GPU_
#ifndef _CODE_GPU_

#define NOMINMAX
#include<optix_world.h>
#include<stdio.h>

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

// global variables
vec4 gl_FragColor=make_float4(0.);
vec2 gl_FragCoord=make_float2(256.,256.);
vec2 resolution=make_float2(512.,512.);//800.,450.);
float time=10.;

#define IN_PARAM(_type) _type&
#define OUT_PARAM(_type) _type&
#define SET_COL(_mat,_i,_col) (_mat).setCol((_i),(_col))

#else

//#ifdef GL_ES
//precision mediump float;
//#endif

#define resolution iResolution
#define time iGlobalTime

#define IN_PARAM(_type) in _type
#define OUT_PARAM(_type) out _type
#define SET_COL(_mat,_i,_col) (_mat)[_i]=(_col)
#define make_float2 vec2
#define make_float3 vec3
#define make_float4 vec4

#endif



///////////////////////////////////////////////////////////////////////////
// glsl code

#define PI 3.14159
#define D2R(_d) ((_d)*PI/180.)

const int max_depth=6;
const int rr_depth=5;

vec3 cam_origin=make_float3(272.691711, 277.386017, -760.679871);
vec3 cam_target=make_float3(272.696594, 277.381134, -759.679871);
vec3 cam_up=make_float3(0.,1.,0.);
float cam_vfov=D2R(39.3077);

#define OBJ_COUNT 1
#define OBJ_NONE -1.
#define OBJ_VIRTUAL -2.

#define FLT_MAX 3.402823466e+38

#define RAY_MARCHING_EPS 1.
#define RAY_MARCHING_MAX_ITER 64

const float scene_scale=5500.;

#undef optix::Ray;
struct Ray
{
	vec3 origin;
	vec3 dir;
};

///////////////////////////////////////////////////////////////////////
// camera

vec2 to_imageplane(const vec2 pixel)
{
	vec2 iplane_size;
	iplane_size.y=2.*tan(0.5*cam_vfov);
	float aspectRatio=resolution.x/resolution.y;
	iplane_size.x=aspectRatio*iplane_size.y;
	
	return (pixel/resolution.xy - 0.5)*iplane_size;
}

// camera -> world
mat3 cam_mat;
void setupCamera()
{
	vec3 n=normalize(cam_origin-cam_target);
	vec3 s=normalize(cross(cam_up,n));
	vec3 t=cross(n,s);
	SET_COL(cam_mat,0,s);
	SET_COL(cam_mat,1,t);
	SET_COL(cam_mat,2,n);
}


Ray genRay(vec2 pixel)
{

	vec2 ixy=to_imageplane(pixel);
	vec3 cam_dir=normalize(make_float3(ixy.x,ixy.y,-1.));
	vec3 world_dir=cam_mat*cam_dir;

	Ray ray;
	ray.dir=world_dir;
	ray.origin=cam_origin;
	
	return ray;
}


///////////////////////////////////////////////////////////////////////
// miss

void rayMiss()
{
}


///////////////////////////////////////////////////////////////////////
// intersection

float udBox(vec3 p, vec3 b, vec3 ray_dir)
{
	return length(max(abs(p)-b,0.0));
}

float sdRect2(vec2 p, vec2 b)
{
	vec2 d = abs(p) - b;
  	return min(max(d.x,d.y),0.0) +
         length(max(d,0.0));
}

float sdBox(vec3 p, vec3 b, vec3 ray_dir)
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}

// -z open
float sdOpenBox(vec3 p, vec3 b, vec3 ray_dir)
{
	if(p.z<-b.z-RAY_MARCHING_EPS) // get to -z face first
		return sdBox(p,b,ray_dir);
	else
		return sdBox(make_float3(p.xy,p.z+b.z),make_float3(b.xy,2.*b.z),ray_dir);
}

vec2 dist(int objId,vec3 p,vec3 ray_dir)
{
	float d=FLT_MAX;
	float isvirtual=0.;
	
	// wall box
	if(objId==0)
	{
		vec3 b=make_float3(550.*0.5);
		vec3 q=p-b;
		//d=sdOpenBox(q,b,ray_dir);
		
		if(q.z<-b.z-RAY_MARCHING_EPS) // get to -z face first
		{
			d=sdBox(q,b,ray_dir);
			isvirtual=1.;
		}
		else
			d=sdBox(make_float3(q.xy,q.z+b.z),make_float3(b.xy,2.*b.z),ray_dir);
		
	}
	// short box
	/*else if(objId==1)
		;
	// tall box
	else if(objId==2)
		;
	// light
	else if(objId==3)
		;*/
	return make_float2(d,isvirtual);
}

float dist(int objId,vec3 p)
{
	return dist(objId,p,make_float3(0.)).x;
}

int intersectm(Ray ray,float maxDist,OUT_PARAM(vec3) position,OUT_PARAM(vec3) normal)
{
	position=make_float3(0.);normal=make_float3(0.);
	float t=abs(ray.origin.z)/abs(ray.dir.z);
	vec3 p=ray.origin+t*ray.dir;
	if(p.x>=0.&&p.x<550.&&p.y>=0.&&p.y<=550.)
		return 0;
	return -1;
}

vec2 closestObj(vec3 p,vec3 ray_dir,float maxDist)
{
	float dmin=maxDist;
	float hit_obj=OBJ_NONE;
	for(int obji=0;obji<OBJ_COUNT;obji++)
	{
		vec2 hobj=dist(obji,p,ray_dir);
		float d=abs(hobj.x);
		
		if(d>-RAY_MARCHING_EPS&&d<dmin)
		{
			dmin=d;
			hit_obj=(hobj.y==0.?float(obji):OBJ_VIRTUAL);
		}
	}
	return make_float2(hit_obj,dmin);
}

vec2 closestObj(vec3 p)
{
	return closestObj(p,make_float3(0.),scene_scale);
}


#define GRADIENT_DELTA 1.
/*
vec3 gradNormal(int objId,vec3 p)
{
	if(p.y>=550.-2.0)
		return make_float3(0.,-1.,0.);
	else if(p.y<=2.0)
		return make_float3(0.,1.,0.);
	else if(p.z>=550.-2.)
		return make_float3(0.,0.,-1.);
	else if(p.z<=2.)
		return make_float3(0.,0.,1.);
	else if(p.x<2.0)
		return make_float3(1.,0.,0.);
	else
		return make_float3(-1.,0.,0.);
}

/**/
vec3 gradNormal(int objId,vec3 p) {
	//return make_float3(
	//	dist(objId,p + make_float3(0, 0, GRADIENT_DELTA)) - dist(objId,p - make_float3(0, 0, GRADIENT_DELTA)),
	//	0.,0.);
	vec3 g=make_float3(
		dist(objId,p + make_float3(GRADIENT_DELTA, 0, 0)) - dist(objId,p - make_float3(GRADIENT_DELTA, 0, 0)),
		dist(objId,p + make_float3(0, GRADIENT_DELTA, 0)) - dist(objId,p - make_float3(0, GRADIENT_DELTA, 0)),
		dist(objId,p + make_float3(0, 0, GRADIENT_DELTA)) - dist(objId,p - make_float3(0, 0, GRADIENT_DELTA)));

    return normalize(g);
}
/**/

float intersect(Ray ray,float maxDist,OUT_PARAM(vec3) p,OUT_PARAM(vec3) normal)
{
	p=ray.origin;
	
	float accum_dist=0.;	
	vec2 cobj=make_float2(OBJ_NONE,FLT_MAX);
	
	for(int rmi=0;rmi<RAY_MARCHING_MAX_ITER;rmi++)
	{
		cobj=closestObj(p,ray.dir,maxDist-accum_dist);
		
		if(cobj.x==OBJ_NONE||(cobj.x!=OBJ_VIRTUAL&&cobj.y<RAY_MARCHING_EPS))
		{
			break;
		}
		
		// else: ray marching		
		accum_dist+=cobj.y;
		p+=cobj.y*ray.dir;// more accurate?
	}
	
	normal=gradNormal(int(cobj.x),p);
	return cobj.x;
}

///////////////////////////////////////////////////////////////////////
// material
vec3 shade(float objId,vec3 p,vec3 n)
{
	if(objId==0.)
	{
		//return make_float3(1.0,1.,0.);
		//return make_float3(p.x/550.,0.,0.);
		//return make_float3(100000.*n.x,0.,0.);
		return 0.5*(n+1.);
		//vec3 t=n/800.;		
		//if(t.x>0.95)
		//	return make_float3(1.,0.,0.);
		//return make_float3(0.);
	}
	else 
		return make_float3(0.);
}

void sampleBSDF(float objId)
{
}



void main()
{
	setupCamera();
	Ray ray=genRay(gl_FragCoord.xy);
	
	//gl_FragColor=make_float4(0.5*(1.0+ray.dir),1.);
	//gl_FragColor=make_float4(ray.dir.y,0.,0.,1.);
	//return;
	
	vec4 res=make_float4(0.);
	vec3 p,n;
	
	for(int d=1;d<max_depth;++d)
	{
		//int oo=intersectm(ray,scene_scale,p,n);
		float obj=intersect(ray,scene_scale,p,n);
		//int obj=0;
		//n=make_float3(length(ray.origin));
		//gl_FragColor=make_float4(heatMap(float(obj+1)/10.),1.);
		if(obj==OBJ_NONE)
		{
			//res=make_float4(1.);
			//rayMiss(ray);
			//break;
		}
		res+=make_float4(shade(obj,p,n),1.);
		
		//ray=sampleRay(obj);
		break;
	}
	
	gl_FragColor=res;
}