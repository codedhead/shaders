#ifdef GL_ES
precision mediump float;
#endif

uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;

const int max_depth=6;
const int rr_depth=5;

vec3 cam_origin;
vec3 cam_target;

const int objCount=10;
#define OBJ_NONE -1


struct Ray
{
	vec3 origin;
	vec3 dir;
};

///////////////////////////////////////////////////////////////////////
// camera

void setupCamera()
{
}

vec3 genRay()
{
	vec3 dir;
	return dir;
}


///////////////////////////////////////////////////////////////////////
// miss

void rayMiss()
{
}


///////////////////////////////////////////////////////////////////////
// intersection

const float box_dim=550.;

bool bboxIntersect(vec3 bmin,vec3 bmax,Ray ray)
{
}
		
bool sphereIntersect(vec4 sphere,
		Ray ray,inout float timin,inout float tmax)
{
	vec3 center=vec3(sphere);
	float r=sphere.w;
	if(!bboxIntersect(center-vec3(r),center+vec3(r),ray))
		return false;
	
}

bool rectIntersect(vec4 plane,
		Ray ray,inout float timin,inout float tmax)
{
	
}

//bool cubeIntersect(


bool objIntersect(int objId,Ray ray,inout float timin,inout float tmax)
{
	switch(objId)
	{
	case 0:
		Rect().intersect(ray,);
		break;
	
	}
	return false;
}

int intersect(Ray ray,inout vec3 position,inout vec3 normal)
{
	float tmin=FLT_MAX,tmax=-FLT_MAX;
	// brute force
	for(int obji=0;obji<objCount;++obji)
	{
		objIntersect(obji,ray,tmin,tmax);
	}
	return OBJ_NONE;
}

///////////////////////////////////////////////////////////////////////
// material
vec3 shade(int objId)
{
}

vec3 sampleBSDF(int objId)
{
}



void main()
{
	setupCamera();
	//vec3 ray=genRay();
	//vec4 res;
	
	/*for(int d=1;d<max_depth;++d)
	{
		obj=intersect(ray);
		if(obj==OBJ_NONE)
		{
			rayMiss(ray);
			break;
		}
		shade(obj);
		
		ray=sampleRay(obj);
	}*/
	
	gl_FragColor+=vec4(0.1,0.0,0.0,1.0);
}