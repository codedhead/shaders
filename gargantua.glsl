/*

Black hole with gravitational lensing and accretion disc.

Reference: 
  Bozza, Valerio. "Gravitational lensing by black holes." General Relativity and Gravitation 42.9 (2010): 2269-2300.

Jun 2014.11.24

*/

//#define MOVE_CAMERA
#define GRAV_LENSING
#define PI 3.14159

vec3 bh=vec3(0.0);
float bh_M=1.; // G=1,c=1
float r_bar=3.0*bh_M;
float disc_r_orig=r_bar * 2.1;
float disc_r=disc_r_orig;
vec3 disc_n=vec3(0.,1.,0.);
vec3 disc_s=vec3(1.,0.,0.);
vec3 eye=vec3(0.,1.2,10.);
vec3 up=normalize(vec3(0.05,1.,0.));
mat3 cam_mat;// camera -> world
float tan_half_vfov=1.0;
vec2 iplane_size=2.*tan_half_vfov*vec2(iResolution.x/iResolution.y,1.);

vec3 rot_x(vec3 v,float theta)
{
    float s=sin(theta),c=cos(theta);
    return vec3(v.x,c*v.y-s*v.z,s*v.y+c*v.z);
}
vec3 rot_y(vec3 v,float theta)
{
    float s=sin(theta),c=cos(theta);
    return vec3(c*v.x+s*v.z,v.y,c*v.z-s*v.x);    
}
vec3 rot_z(vec3 v,float theta)
{
    float s=sin(theta),c=cos(theta);
    return vec3(c*v.x-s*v.y,s*v.x+c*v.y,v.z);
}
void setupCamera()
{
#ifdef MOVE_CAMERA
    vec2 rxy=(iMouse.xy / iResolution.xy-0.5) * PI;
    eye=rot_x(eye,rxy.y); // fixed target(0,0,0)
    eye=rot_y(eye,-rxy.x);
    //up=rot_z(up,-rxy.x);
#endif
	vec3 n=normalize(eye-bh);
	vec3 s=normalize(cross(up,n));
	vec3 t=cross(n,s);
    cam_mat[0]=s;
    cam_mat[1]=t;
    cam_mat[2]=n;
}

vec4 disc_color(vec3 p_disc)
{
    vec3 v=p_disc-bh;
    float d=length(v);v/=d;
    if(d<disc_r)
    {   
        //return vec4(1.0,0.,0.,1.);
   	 	vec2 uv=vec2((atan(dot(v,disc_s),dot(v,cross(disc_s,disc_n)))/(2.*PI)*1.-iGlobalTime*0.3),
                 (d-r_bar)/(disc_r-r_bar));    
    	return 3.*texture2D(iChannel0,uv)*smoothstep(disc_r,r_bar,d);
    }
    else return vec4(0.);
}

void main(void)
{
    setupCamera();
	float ruv = length((gl_FragCoord.xy-0.5*iResolution.xy)/iResolution.y);
    vec4 color=vec4(0.7,0.6,0.8,1.) * exp(-ruv*1.5);
    
    vec2 ixy=(gl_FragCoord.xy/iResolution.xy - 0.5)*iplane_size;
    vec3 ray_dir=cam_mat*normalize(vec3(ixy,-1.));
    vec3 h2e=eye-bh;
    float l2_h2e=dot(h2e,h2e);
    float rm=length(cross(ray_dir,h2e)); // smallest distance
    float t_cp=sqrt(l2_h2e-rm*rm); // t of closest point
    
    //if(rm>r_bar)
    {
#ifdef GRAV_LENSING
        float alpha=4.*bh_M/rm;   
        disc_r=disc_r_orig;
        if(rm<r_bar) // hack
            alpha*=(1.-abs(dot(disc_n,up))),disc_r*=1.25;
#else
        float alpha=0.;
#endif
        float tan_a_2=tan(alpha*0.5);
        
        vec3 cp=eye+ray_dir*t_cp;// closest point
        vec3 coord_origin=cp+ray_dir*(rm*tan_a_2);
        vec3 x_axis=normalize(bh-coord_origin);
        vec3 y_axis=normalize(ray_dir+tan_a_2*normalize(bh-cp));
       	vec3 z_axis=cross(x_axis,y_axis);
       	
        float c=length(bh-coord_origin);
        float k=tan_a_2; // a/b
        
        // the intersection line pass through bh
        vec3 iline_r=normalize(cross(z_axis,disc_n));
        
        float x1=-1.,x2=-1.,y1,y2;

#ifdef GRAV_LENSING
		float k2=k*k;
        float b2=c*c/(1.0+k2);
        float a2=k2*b2;
        float a=sqrt(a2);
        // x^2/a2 - y^2/b2 = 1

        float denom=dot(x_axis,iline_r);
        if(denom==0.)
        {
            x1=x2=c;
            y1=-b2/sqrt(a2);
            y2=-y1;
        }
        else
        {
            float slope=dot(y_axis,iline_r)/denom; // y=slope*(x-c)
            k2=slope*slope; // override k2
            float A=a2*k2-b2;
            float B=-2.*a2*k2*c;
            float C=a2*(k2*c*c+b2);
            // B*B-4AC>=0
            float delta=sqrt(B*B-4.*A*C);
            x1=(-B-delta)/(2.*A);
			x2=(-B+delta)/(2.*A);
			y1=slope*(x1-c);
            y2=slope*(x2-c);
        }
#else
        float denom=dot(x_axis,iline_r);
        if(denom!=0.) // else no intersection
        {
            float slope=dot(y_axis,iline_r)/denom;
            x1=0.;y1=-slope*c;
        }
#endif
        
        vec3 o2e=eye-coord_origin;
        float yeye=dot(o2e,y_axis);
        vec3 p1=coord_origin+x1*x_axis+y1*y_axis,
            p2=coord_origin+x2*x_axis+y2*y_axis;
        if(x1>=0.&&y1>=yeye && ((y1<0.&&length(p1-bh)>r_bar)||rm>r_bar))
        {
           color+=disc_color(p1);
        }
        if(x2>=0.&&y2>=yeye && ((y2<0.&&length(p2-bh)>r_bar) ||rm>r_bar))
        {
           color+=disc_color(p2);
        }
    }
    
    gl_FragColor=color;
}