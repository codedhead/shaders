/*

Porting the MIS path tracer to GLSL.

part of ray marching code adpated from:
https://www.shadertoy.com/view/4ssGzS

Jun 2014.10.4

*/

// some macros to make code compatible with CPP version
#define IN_PARAM(_type) in _type
#define OUT_PARAM(_type) out _type
#define SET_COL(_mat,_i,_col) (_mat)[_i]=(_col)
#define make_float2 vec2
#define make_float3 vec3
#define make_float4 vec4
#define INIT_MAT3 mat3

///////////////////////////////////////////////////////////////////////////
// main code

#define GAMMA 2.2
#define SPP 16
#define INV_SPP 0.0625
#define MAX_DEPTH 4
    
#define OBJ_COUNT 4
#define OBJ_NONE -1.
#define OBJ_VIRTUAL -2.
#define OBJ_WALL 0
#define OBJ_SBOX 1
#define OBJ_TBOX 2
#define OBJ_LIGHT 3

#define EPS 5e-3
#define RAY_MARCHING_MAX_ITER 128
#define GRADIENT_DELTA 0.1
    
#define PI 3.14159
#define INV_PI (1./PI)
#define INV_2PI (0.5/PI)
#define D2R(_d) ((_d)*PI/180.)

// scene description
const float scene_scale=5500.;
vec3 sbox_center=make_float3(175.,82.5,168.5);
vec3 tbox_center=make_float3(368.5,165.,351.5);
vec3 light_center=make_float3(556.*0.5,548.8,559.*0.5);
vec3 sbox_halfsize=0.5*make_float3(165.);
vec3 tbox_halfsize=0.5*make_float3(165.,330.,165.);
vec3 light_halfsize=make_float3(65.,0.,52.5);
mat3 sbox_w2o=INIT_MAT3(
    0.953400, 		0,		-0.285121,
    0, 				1.,		0,
    0.301709,		0.,		0.958492);
mat3 tbox_w2o=INIT_MAT3(
    0.955649, 		0,		0.301709,
    0, 				1., 	0,
    -0.294508, 		0, 		0.953400);
vec3 box_mtl=make_float3(0.7,0.7,0.4);

///////////////////////////////////////////////////////////////////////
// helpers

struct Ray
{
	vec3 origin;
	vec3 dir;
};
    
float seed;
float rnd() { return fract(sin(seed++)*43758.5453123); }

float pdfA2W(float pdf_area,vec3 p,vec3 p_next,vec3 n_next)
{
	vec3 w=p_next-p;
	float dist2=dot(w,w);
	w*=sqrt(dist2);
	return pdf_area*dist2/(abs(dot(n_next,w)));
}

vec3 uniformHemisphere(float u1, float u2)
{
	float r=sqrt(1.-u1*u1);
	float phi=2.*PI*u2;
	return make_float3(r*cos(phi),r*sin(phi),u1);
}

vec3 l2w(vec3 l,vec3 normal)
{
	vec3 binormal,tangent;
	if( abs(normal.x) > abs(normal.z) )
	{
		binormal.x = -normal.y;binormal.y =  normal.x;binormal.z =  0.;
	}
	else
	{
		binormal.x =  0.;binormal.y = -normal.z;binormal.z =  normal.y;
	}
	binormal = normalize(binormal);
	tangent = cross( binormal, normal );
	return l.x*tangent+l.y*binormal+l.z*normal;
}

///////////////////////////////////////////////////////////////////////
// camera

vec3 cam_origin=make_float3(272.691711, 277.386017, -760.679871);
vec3 cam_target=make_float3(272.696594, 277.381134, -759.679871);
vec3 cam_up=make_float3(0.,1.,0.);
float cam_vfov=D2R(39.3077);
vec2 iplane_size=2.*tan(0.5*cam_vfov)*make_float2(iResolution.x/iResolution.y,1.);
mat3 cam_mat;// camera -> world

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
	vec2 ixy=(pixel/iResolution.xy - 0.5)*iplane_size;
	vec3 cam_dir=normalize(make_float3(ixy.x,ixy.y,-1.));
	vec3 world_dir=cam_mat*cam_dir;
	Ray ray;
	ray.dir=world_dir;
	ray.origin=cam_origin;	
	return ray;
}

///////////////////////////////////////////////////////////////////////
// intersection

float udBox(vec3 p, vec3 b, vec3 ray_dir)
{
	return length(max(abs(p)-b,0.0));
}

float sdBox(vec3 p, vec3 b, vec3 ray_dir)
{
    vec3 d = abs(p) - b;
	return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

vec2 dist(int objId,vec3 p,vec3 ray_dir)
{
	float d=scene_scale;
	float isvirtual=0.;
	
	// wall box
	if(objId==OBJ_WALL)
	{
		vec3 b=make_float3(550.*0.5);
		vec3 q=p-b;
        if(q.z<-b.z-EPS) // get to -z face first
		{
			d=-sdBox(q,b,ray_dir);
			isvirtual=1.;
		}
		else
		{
			d=-sdBox(make_float3(q.xy,q.z+b.z),make_float3(b.xy,2.*b.z),ray_dir);
		}
	}
	// short box
	else if(objId==OBJ_SBOX)
	{
		vec3 q=sbox_w2o*(p-sbox_center);		
		d=udBox(q,sbox_halfsize,ray_dir);
	}
	// tall box
	else if(objId==OBJ_TBOX)
	{
		vec3 q=tbox_w2o*(p-tbox_center);		
		d=udBox(q,tbox_halfsize,ray_dir);
	}		
	// light
	else if(objId==OBJ_LIGHT)
	{
		vec3 q=p-light_center;
		d=-sign(q.y)*udBox(q,light_halfsize,ray_dir);
	}
    
	return make_float2(d,isvirtual);
}

float dist(int objId,vec3 p)
{
	return dist(objId,p,make_float3(0.)).x;
}

vec2 closestObj(vec3 p,vec3 ray_dir,float maxDist)
{
	float dmin=maxDist;
	float hit_obj=OBJ_NONE;
	for(int obji=0;obji<OBJ_COUNT;obji++)
	{
		vec2 hobj=dist(obji,p,ray_dir);
		float d=abs(hobj.x);
		
		if(d<dmin)
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

vec3 gradNormal(int objId,vec3 p) {
	vec3 g=make_float3(
		dist(objId,p + make_float3(GRADIENT_DELTA, 0, 0)) - dist(objId,p - make_float3(GRADIENT_DELTA, 0, 0)),
		dist(objId,p + make_float3(0, GRADIENT_DELTA, 0)) - dist(objId,p - make_float3(0, GRADIENT_DELTA, 0)),
		dist(objId,p + make_float3(0, 0, GRADIENT_DELTA)) - dist(objId,p - make_float3(0, 0, GRADIENT_DELTA)));

    return normalize(g);
}

bool ishit(vec2 cobj,float accum_dist,float eps,float maxDist)
{
	return (cobj.x!=OBJ_VIRTUAL&&
		(accum_dist>eps&&accum_dist<maxDist)
		&&cobj.y<eps);
}

float intersect(Ray ray,float minDist,float maxDist,OUT_PARAM(vec3) p)
{
	p=ray.origin;
	
	float accum_dist=0.;
	vec2 cobj;
	bool ish=false;

	for(int rmi=0;rmi<RAY_MARCHING_MAX_ITER;rmi++)
	{
		cobj=closestObj(p,ray.dir,maxDist-accum_dist);
		
		accum_dist+=cobj.y;
		p+=cobj.y*ray.dir;

		if(cobj.x==OBJ_NONE||(ish=ishit(cobj,accum_dist,minDist,maxDist)))
			break;
	}
	
    return ish?cobj.x:OBJ_NONE;
}

float intersect(Ray ray,float minDist,float maxDist,OUT_PARAM(vec3) p,OUT_PARAM(vec3) normal)
{
	float hit_obj=intersect(ray,minDist,maxDist,p);
    if(hit_obj!=OBJ_NONE)
        normal=gradNormal(int(hit_obj),p);
	return hit_obj;
}

bool isShadowed(Ray ray,float minDist,float maxDist)
{
	vec3 p;
    return intersect(ray,minDist,maxDist,p)!=OBJ_NONE;
}

///////////////////////////////////////////////////////////////////////
// light

const vec3 Lradiance=make_float3(28.4);
const float Lpdf=1./(130.*105.);
void sampleLight(vec3 ref_p,OUT_PARAM(vec3) Lp,OUT_PARAM(vec3) Ln)
{
	vec2 uv=make_float2(rnd(),rnd());
	Lp=make_float3(213.,548.8,227.)+make_float3(uv.x*130.,0.,uv.y*105.);
	Ln=make_float3(0.,-1.,0.);
}

///////////////////////////////////////////////////////////////////////
// bsdf

float bsdfPdf(vec3 w,vec3 n)
{
    return dot(w,n)>0.?INV_2PI:0.;
}

vec3 sampleBSDF(float objId,vec3 p,vec3 n)
{
	vec3 dir=uniformHemisphere(rnd(),rnd());
	return l2w(dir,n);
}

vec3 shade(int objId,vec3 p,vec3 n,Ray ray,int depth,OUT_PARAM(vec3) alpha)
{
	vec3 f=make_float3(0.);
    if(objId==OBJ_WALL)
    {
        float x=0.5*(n.x+1.);
        float y=abs(n.y)+abs(n.z);
        f=make_float3(0.7*(1.-x+0.5*y),0.7*(x+0.5*y),0.4*y);
    }
    else if(objId==OBJ_SBOX||objId==OBJ_TBOX)
    {
        f=box_mtl;
    }
    else if(objId==OBJ_LIGHT)
    {
        f=make_float3(0.5);
    }
	
	vec3 res=make_float3(0.);
	if(objId==OBJ_LIGHT)
    {
		float wgt=1.;// if depth>=2&&not specular, do mis
        if(depth>1)
        {
            float lgt_pdf=pdfA2W(Lpdf,ray.origin,p,n);
            float bsdf_pdf=bsdfPdf(p-ray.origin,n);
            wgt=(bsdf_pdf*bsdf_pdf)/(lgt_pdf*lgt_pdf+bsdf_pdf*bsdf_pdf);
        }
		res=alpha*wgt*Lradiance;
    }
    else
    {
        vec3 Lp,Ln;
        sampleLight(p,Lp,Ln);
        Ray shadow_ray;        
        shadow_ray.dir=normalize(p-Lp);
        if(dot(shadow_ray.dir,Ln)>0.)
        {
            vec3 absp=abs(Lp);
            shadow_ray.origin=Lp+EPS*max(max(absp.x,absp.y),absp.z)*shadow_ray.dir;	// in case point is under the surface
            float max_dist=(1.-EPS)*length(shadow_ray.origin-p);
            bool shadowed=isShadowed(shadow_ray,EPS,max_dist);
            if(!shadowed)
            {
                float g=abs(dot(shadow_ray.dir,n))*
                        abs(dot(shadow_ray.dir,Ln))/(max_dist*max_dist);
                float bsdf_pdf=bsdfPdf(Lp-p,Ln);
                float lgt_pdf=pdfA2W(Lpdf,p,Lp,Ln);
                float wgt=(lgt_pdf*lgt_pdf)/(lgt_pdf*lgt_pdf+bsdf_pdf*bsdf_pdf);
                res=wgt*alpha*f*INV_PI*g*Lradiance/Lpdf;
            }
        }
    }
	alpha*=f; //!! actually *f should take place in sampleBSDF, but for diffuse, doesn't matter
	return res;
}

void main()
{
    seed = /*iGlobalTime +*/ iResolution.y * gl_FragCoord.x / iResolution.x + gl_FragCoord.y / iResolution.y;	
	setupCamera();
	
	vec3 res=make_float3(0.);
	for(int si=1;si<=SPP;++si)
    {
		vec3 alpha=make_float3(1.); // throughput
		Ray ray=genRay(gl_FragCoord.xy+make_float2(rnd(),rnd()));
		
		for(int d=1;d<MAX_DEPTH;++d)
		{
            vec3 p,n;
			float obj=intersect(ray,EPS,scene_scale,p,n);
			if(obj==OBJ_NONE)
                break;
			else
				res+=shade(int(obj),p,n,ray,d,alpha);

			ray.dir=sampleBSDF(obj,p,n);
			alpha*=2.*abs(dot(ray.dir,n)); // correct the alpha, since we are uniform-sampling
			vec3 absp=abs(p);
			float eps=EPS*max(max(absp.x,absp.y),absp.z);
			ray.origin=p+eps*ray.dir;
		}
	}
    res = pow(res*INV_SPP, vec3(1. / GAMMA));
	gl_FragColor=make_float4(res,1.); 
}