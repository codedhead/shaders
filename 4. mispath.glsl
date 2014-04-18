//#ifdef GL_ES
//precision mediump float;
//#endif


#define resolution iResolution
#define time iGlobalTime

#define PI 3.14159
#define D2R(_d) ((_d)*PI/180.)

const int max_depth=6;
const int rr_depth=5;

vec3 cam_origin=vec3(272.691711, 277.386017, -760.679871);
vec3 cam_target=vec3(272.696594, 277.381134, -759.679871);
vec3 cam_up=vec3(0.,1.,0.);
float cam_vfov=D2R(39.3077);

#define OBJ_COUNT 1
#define OBJ_NONE -1

#define FLT_MAX 3.402823466e+38

#define RAY_MARCHING_EPS 1.
#define RAY_MARCHING_MAX_ITER 128

const float scene_scale=550.;

float seed = 0.;
float rand() { return fract(sin(seed++)*43758.5453123); }

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
	cam_mat[0]=s;
	cam_mat[1]=t;
	cam_mat[2]=n;
}



Ray genRay(vec2 pixel)
{

	vec2 ixy=to_imageplane(pixel);
	vec3 cam_dir=normalize(vec3(ixy.x,ixy.y,-1.));
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
	return length(p);
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}

// -z open
float sdOpenBox(vec3 p, vec3 b, vec3 ray_dir)
{
	//if(p.z>b.z||p.z<-b.z) // get to -z face first
		return sdBox(p,b,ray_dir);
	//else
	//	return sdBox(vec3(p.xy,p.z+b.z),vec3(b.xy,2.*b.z),ray_dir);
}

float dObject(int objId,vec3 p,vec3 ray_dir)
{
	float d=FLT_MAX;
	
	// wall box
	if(objId==0)
	{
		vec3 q=p-vec3(550.*0.5);
		float d1=sdBox(q,vec3(550.*0.5),ray_dir);
		
		//q=p-vec3(550.0*0.5,550.0*0.5,0.);
		//float d2=udBox(q,vec3(550.*0.5,550.*0.5,0.),ray.dir);
		d=d1;
		//d=max(-d1,d2);
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
	return d;
}

#define GRADIENT_EPS 0.1
vec3 gradientNormal(vec3 p)
{
	if(p.y>=550.-2.0)
		return vec3(0.,-1.,0.);
	else if(p.y<=2.0)
		return vec3(0.,1.,0.);
	else if(p.z>=550.-2.)
		return vec3(0.,0.,-1.);
	else if(p.x<2.0)
		return vec3(1.,0.,0.);
	else
		return vec3(-1.,0.,0.);
}

int intersect(Ray ray,float maxDist,out vec3 position,out vec3 normal)
{
	position=vec3(0.);normal=vec3(0.);
	
	float accum_dist=0.;
	vec3 p=ray.origin;
	
	for(int rmi=0;rmi<RAY_MARCHING_MAX_ITER;++rmi)
	{
		float dmin=10000000.;//FLT_MAX;
		int hit_obj=OBJ_NONE;

		for(int obji=0;obji<OBJ_COUNT;++obji)
		{			
			float d=dObject(obji,p,ray.dir);
			
			if(d>-RAY_MARCHING_EPS&&(accum_dist<1.)&&d<dmin)
			{
				dmin=d;
				hit_obj=obji;
			}
		}
		
		
		if(hit_obj==OBJ_NONE)
		{
			return OBJ_NONE;
		}
		if(dmin<RAY_MARCHING_EPS)
		{
			position=p;
			normal=gradientNormal(position);
			return hit_obj;
		}
		
		// else: ray marching
		
		// more accurate?
		accum_dist+=100.;dmin;
		p+=dmin*ray.dir;
	}
	
	return OBJ_NONE;
}

///////////////////////////////////////////////////////////////////////
// material
vec3 shade(int objId,vec3 p,vec3 n)
{
	if(objId==0)
	{
		return vec3(1.0,1.,0.);
		//return vec3(p.x/550.,0.,0.);
		//return 0.5*(n+1.);
		//return n/2900.;		
	}
	else 
		return vec3(0.);
}

void sampleBSDF(int objId)
{
}



void main()
{
	setupCamera();
	Ray ray=genRay(gl_FragCoord.xy);
	
	//gl_FragColor=vec4(0.5*(1.0+ray.dir),1.);
	//gl_FragColor=vec4(ray.dir.y,0.,0.,1.);
	//return;
	
	vec4 res=vec4(0.);
	vec3 p,n;
	
	for(int d=1;d<max_depth;++d)
	{
		int obj=intersect(ray,scene_scale,p,n);
		//int obj=0;
		//n=vec3(length(ray.origin));
		//gl_FragColor=vec4(heatMap(float(obj+1)/10.),1.);
		if(obj==OBJ_NONE)
		{
			//res=vec4(1.);
			//rayMiss(ray);
			//break;
		}
		res+=vec4(shade(obj,p,n),1.);
		
		//ray=sampleRay(obj);
		break;
	}
	
	gl_FragColor=res;
}