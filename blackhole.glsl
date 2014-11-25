// reference: http://www.photon.at/~werner/bh/gvsim.html

vec3 bh=vec3(0.0);
float bh_r=4.0;
float bh_M=1.;

vec3 eye=vec3(0.,0.,10.);
vec3 up=vec3(0.,1.,0.);
mat3 cam_mat;// camera -> world
float tan_half_vfov=1.0;
vec2 iplane_size=2.*tan_half_vfov*vec2(iResolution.x/iResolution.y,1.);

void setupCamera()
{
	vec3 n=normalize(eye-bh);
	vec3 s=normalize(cross(up,n));
	vec3 t=cross(n,s);
	cam_mat[0]=s;
    cam_mat[1]=t;
    cam_mat[2]=n;
}

vec3 rot_y(vec3 v,float theta)
{
    float s=sin(theta),c=cos(theta);
    return vec3(c*v.x+s*v.z,v.y,c*v.z-s*v.x);    
}

void main(void)
{
    setupCamera();
    
    vec2 ixy=(gl_FragCoord.xy/iResolution.xy - 0.5)*iplane_size;
    vec3 ray_dir=cam_mat*normalize(vec3(ixy.x,ixy.y,-1.));   
   
	vec3 h2e=eye-bh; float l2_h2e=dot(h2e,h2e);
    float d=length(cross(ray_dir,h2e));
	if(d>3.*bh_M)
    {
        // closest point
        vec3 cp= eye+ray_dir*sqrt(l2_h2e-d*d);
		// bend
		float alpha=4.*bh_M/d;
        ray_dir=normalize(ray_dir+tan(alpha)*normalize(bh-cp));
        
        gl_FragColor=textureCube(iChannel0,rot_y(ray_dir,-0.2*iGlobalTime));
    }
    else
        gl_FragColor=vec4(0.);
}