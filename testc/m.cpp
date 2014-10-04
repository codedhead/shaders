#define _CODE_GPU_
#ifndef _CODE_GPU_
//#include "cpuv.h"
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
#define INIT_MAT3 mat3

float seed;
float rnd() { return fract(sin(seed++)*43758.5453123); }

/*float rand(vec2 co)
{
	seed=fract(sin(dot(co.xy,vec2(12.9898,78.233))) * 43758.5453);
	return seed;
}
float rnd()
{
	return rand(vec2(gl_FragCoord)+seed);
}*/

#endif


///////////////////////////////////////////////////////////////////////////
// glsl code

#define PI 3.14159
#define INV_PI (1./PI)
#define D2R(_d) ((_d)*PI/180.)

vec3 cam_origin=make_float3(272.691711, 277.386017, -760.679871);
vec3 cam_target=make_float3(272.696594, 277.381134, -759.679871);
vec3 cam_up=make_float3(0.,1.,0.);
float cam_vfov=D2R(39.3077);

#define GAMMA 2.2
#define SPP 8
#define INV_SPP 0.125
#define MAX_DEPTH 4
#define OBJ_COUNT 4
#define OBJ_NONE -1.
#define OBJ_VIRTUAL -2.

#define OBJ_WALL 0
#define OBJ_SBOX 1
#define OBJ_TBOX 2
#define OBJ_LIGHT 3
#define OBJ_LIGHTf 3.

#define FLT_MAX 3.402823466e+38

#define EPS 1e-2
#define RAY_MARCHING_MAX_ITER 128

const float scene_scale=5500.;
const vec3 Lradiance=make_float3(28.4);

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
    //vec3 newo = cam_origin + vec3(cos(time * 0.8) * 180., cos(time * 0.9) * 180., (cos(time * .3) + 1.) * 390.);
    //vec3 o=mix(cam_origin, newo, smoothstep(0., 1., (time - 5.) * .1));

    
    
	//float angZ = smoothstep(0., 1., (time - 5.) * .1) * sin(time * 1.1 + .77) * .05;
	//	ray_dir = rotateZ(ray_dir, angZ);
    
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

const vec3 vec30=make_float3(0.);

// -z open
float sdOpenBox(vec3 p, vec3 b, vec3 ray_dir)
{
	if(p.z<-b.z-EPS) // get to -z face first
		return sdBox(p,b,ray_dir);
	else
		return sdBox(make_float3(p.xy,p.z+b.z),make_float3(b.xy,2.*b.z),ray_dir);
}

vec2 dist(int objId,vec3 p,vec3 ray_dir)
{
	float d=FLT_MAX;
	float isvirtual=0.;
	
	// wall box
	if(objId==OBJ_WALL)
	{
		vec3 b=make_float3(550.*0.5);
		vec3 q=p-b;
		//d=sdOpenBox(q,b,ray_dir);
        
        /*
        float inv= (ray_dir.z>0.?(1./ray_dir.z):0.);
        float dt=max(-b.z-q.z,0.)*inv;///ray_dir.z;
       	q+=dt*ray_dir;
        float dt2=sdBox(make_float3(q.xy,q.z+b.z),make_float3(b.xy,2.*b.z),ray_dir);
        d= (dt>0.? (dt+abs(dt2)): -dt2);
        */

        if(q.z<-b.z-EPS) // get to -z face first
		{
			// flip side
			d=-sdBox(q,b,ray_dir);
			isvirtual=1.;
		}
		else
		{
			// flip side
			d=-sdBox(make_float3(q.xy,q.z+b.z),make_float3(b.xy,2.*b.z),ray_dir);
		}
		
	}
	// short box
	else if(objId==OBJ_SBOX)
	{
#ifdef _CODE_GPU_
		mat3 w2o=INIT_MAT3(
			0.953400, 		0,		-0.285121,
			0, 				1.,		0,
			0.301709,		0.,		0.958492);
#else
		mat3 w2o=INIT_MAT3(
			0.953400, 		0,		0.301709,
			0, 				1.,		0,
			-0.285121,		0.,		0.958492);
#endif
		vec3 b=make_float3(165.*0.5);
		vec3 q=w2o*(p-make_float3(175.,82.5,168.5));		
		d=udBox(q,b,ray_dir);
	}
	// tall box
	else if(objId==OBJ_TBOX)
	{
#ifdef _CODE_GPU_
		mat3 w2o=INIT_MAT3(
			0.955649, 		0,		0.301709,
			0, 				1., 	0,
			-0.294508, 		0, 		0.953400);
#else
		mat3 w2o=INIT_MAT3(
			0.955649, 		0,		-0.294508,
			0, 				1., 	0,
			0.301709, 		0, 		0.953400);
#endif
		vec3 b=make_float3(165.*0.5,165.,165.*0.5);
		vec3 q=w2o*(p-make_float3(368.5,165.,351.5));		
		d=udBox(q,b,ray_dir);
	}		
	// light
	else if(objId==OBJ_LIGHT)
	{
		vec3 q=p-make_float3(556.*0.5,548.8,559.*0.5);
		vec3 b=make_float3(65.,0.,52.5);
		d=udBox(q,b,ray_dir);
	}
	return make_float2(d,isvirtual);
}

float dist(int objId,vec3 p)
{
	return dist(objId,p,make_float3(0.)).x;
}

// int intersectm(Ray ray,float maxDist,OUT_PARAM(vec3) position,OUT_PARAM(vec3) normal)
// {
// 	position=make_float3(0.);normal=make_float3(0.);
// 	float t=abs(ray.origin.z)/abs(ray.dir.z);
// 	vec3 p=ray.origin+t*ray.dir;
// 	if(p.x>=0.&&p.x<550.&&p.y>=0.&&p.y<=550.)
// 		return 0;
// 	return -1;
// }

vec2 closestObj(vec3 p,vec3 ray_dir,float maxDist)
{
	float dmin=maxDist;
	float hit_obj=OBJ_NONE;
	for(int obji=0;obji<OBJ_COUNT;obji++)
	{
		vec2 hobj=dist(obji,p,ray_dir);
		float d=abs(hobj.x);
		
		if(/*d>-RAY_MARCHING_EPS&&*/d<dmin)
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

bool ishit(vec2 cobj,float accum_dist,float eps,float maxDist)
{
	return (cobj.x!=OBJ_VIRTUAL&&
		(accum_dist>eps&&accum_dist<maxDist)
		&&cobj.y<eps);
}

float intersect(Ray ray,float minDist,float maxDist,OUT_PARAM(vec3) p,OUT_PARAM(vec3) normal)
{
	p=ray.origin;
	
	float accum_dist=0.;
	vec2 cobj;
	bool ish=false;

	for(int rmi=0;rmi<RAY_MARCHING_MAX_ITER;rmi++)
	{
		cobj=closestObj(p,ray.dir,maxDist-accum_dist);
		
		accum_dist+=cobj.y;
		p+=cobj.y*ray.dir;// more accurate?

		if(cobj.x==OBJ_NONE||(ish=ishit(cobj,accum_dist,minDist,maxDist)))
			break;
	}
	
	float ret_type=OBJ_NONE;
	if(ish)
	{
		normal=gradNormal(int(cobj.x),p);
		ret_type=cobj.x;
	}
	else
	{
		normal=make_float3(0.);
	}
	return ret_type;
}

bool isShadowed(Ray ray,float minDist,float maxDist)
{
	vec3 p=ray.origin;
	
	float accum_dist=0.;
	vec2 cobj;
	bool ish=false;

	for(int rmi=0;rmi<RAY_MARCHING_MAX_ITER;rmi++)
	{
		cobj=closestObj(p,ray.dir,maxDist-accum_dist);
		
		accum_dist+=cobj.y;
		p+=cobj.y*ray.dir;// more accurate?

		if(cobj.x==OBJ_NONE|| (ish=ishit(cobj,accum_dist,minDist,maxDist)))
		{
			break;
		}		
	}
	
	return ish;
}

///////////////////////////////////////////////////////////////////////
// material

void sampleLight(vec3 ref_p,OUT_PARAM(vec3) L,OUT_PARAM(vec3) Lp,
				 OUT_PARAM(vec3) Ln,OUT_PARAM(float) pdf)
{
	vec2 uv=make_float2(rnd(),rnd());
	Lp=make_float3(213.,548.8,227.)+make_float3(uv.x*130.,0.,uv.y*105.);
	Ln=make_float3(0.,-1.,0.);
	L=Lradiance;
	pdf=1./(130.*105.);
}

vec3 shade(float objId,vec3 p,vec3 n,vec3 alpha,OUT_PARAM(vec3) f)
{
	f=make_float3(0.);
    if(objId==0.)
    {
        float x=0.5*(n.x+1.);
        float y=abs(n.y)+abs(n.z);
        f=make_float3(0.7*(1.-x+0.5*y),0.7*(x+0.5*y),0.4*y);
    }
    else if(objId==1.||objId==2.)
    {
        f=make_float3(0.7,0.7,0.4);
    }
    else if(objId==OBJ_LIGHTf)
    {
        f=make_float3(0.5);
    }
	
	vec3 res=make_float3(0.);
	if(objId==OBJ_LIGHTf)
    {
        res=alpha*Lradiance;
    }
    else
    {
        vec3 L,Lp,Ln;
        float pdf;
        sampleLight(p,L,Lp,Ln,pdf);

        Ray shadow_ray;
        // 	float3 absp=abs(p);
        // 	shadow_ray.dir=normalize(Lp-p);
        // 	shadow_ray.origin=p+EPS*max(max(absp.x,absp.y),absp.z)*shadow_ray.dir;	
        // 	
        // 	if(dot(shadow_ray.dir,Ln)>=0.) return res;
        // 	
        // 	float max_dist=(1.-EPS)*length(Lp-p);
        // 	bool shadowed=isShadowed(shadow_ray,RAY_MARCHING_EPS,max_dist);

        vec3 absp=abs(Lp);
        shadow_ray.dir=normalize(p-Lp);
        shadow_ray.origin=Lp+EPS*max(max(absp.x,absp.y),absp.z)*shadow_ray.dir;	// in case point is under the surface

        if(dot(shadow_ray.dir,Ln)<=0.) return res;

        float max_dist=(1.-EPS)*length(shadow_ray.origin-p);
        bool shadowed=isShadowed(shadow_ray,EPS,max_dist);


        if(!shadowed)
        {
            float g=abs(dot(shadow_ray.dir,n))*
                    abs(dot(shadow_ray.dir,Ln))/(max_dist*max_dist);
            res=alpha*f*INV_PI*g*L/pdf;
        }
    }
	
	return res;
}


vec3 uniformHemisphere(float u1, float u2)
{
	float z=u1;
	float r=sqrt(1.-z*z);
	float phi=2.*PI*u2;
	return make_float3(r*cos(phi),r*sin(phi),z);
}

vec3 l2w(vec3 l,vec3 normal)
{
	vec3 binormal,tangent;
	
	if( abs(normal.x) > abs(normal.z) )
	{
		binormal.x = -normal.y;
		binormal.y =  normal.x;
		binormal.z =  0.;
	}
	else
	{
		binormal.x =  0.;
		binormal.y = -normal.z;
		binormal.z =  normal.y;
	}
	
	binormal = normalize(binormal);
	tangent = cross( binormal, normal );
	
	return l.x*tangent+l.y*binormal+l.z*normal;
}

vec3 sampleBSDF(float objId,vec3 p,vec3 n)
{
	vec3 dir=uniformHemisphere(rnd(),rnd());
	return l2w(dir,n);
}


#ifdef _CODE_GPU_
void main()
{
	//seed = sin(iGlobalTime)*(gl_FragCoord.y*resolution.x+gl_FragCoord.x);
    seed = iGlobalTime + iResolution.y * gl_FragCoord.x / iResolution.x + gl_FragCoord.y / iResolution.y;
#else
vec4 renderMain(vec2 gl_FragCoord)
{
#endif
	
	setupCamera();
	
	
	//gl_FragColor=make_float4(0.5*(1.0+ray.dir),1.);
	//gl_FragColor=make_float4(ray.dir.y,0.,0.,1.);
	//return;
	
	vec3 res=make_float3(0.);
	vec3 p,n;

	for(int si=1;si<=SPP;++si){

		vec3 alpha=make_float3(1.);
		Ray ray=genRay(gl_FragCoord.xy+make_float2(rnd(),rnd()));
		
		for(int d=1;d<MAX_DEPTH;++d)
		{
			float obj=intersect(ray,EPS,scene_scale,p,n);
			if(obj==OBJ_NONE)
			{
				//rayMiss(ray);
				break;
			}
			else{
				vec3 f;
				res+=shade(obj,p,n,alpha,f);
				//!! actually *f should happen in sampleBSDF, but for diffuse, dosen't matter
				alpha*=f;

				//res=make_float4( 0.25+obj*0.25); 
				//res=0.5*(make_float3(1.)+n);break;
			}
			

			ray.dir=sampleBSDF(obj,p,n);
			vec3 absp=abs(p);
			float eps=EPS*max(max(absp.x,absp.y),absp.z);
			ray.origin=p+eps*ray.dir;
		}

	}

    res = pow(res*INV_SPP, vec3(1. / GAMMA));

#ifdef _CODE_GPU_
	gl_FragColor=make_float4(res,1.); 
#else
	return make_float4(res,1.); 
#endif
}