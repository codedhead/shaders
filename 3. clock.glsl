// 2014.4.14
// https://www.shadertoy.com/view/lsfSRN

#define CLOCK_CX 300.0
#define CLOCK_CY 200.0
#define CLOCK_R 100.0
#define CLOCK_BORDER 2.0
#define POINTER_WIDTH 2.0

#define PI 3.14159
#define D2R(_d) ((_d)*PI/180.0)

bool draw_pointer(vec2 rel_pos,float angle,float pt_len)
{
	float r=length(rel_pos);
	vec2 pt_axis=vec2(cos(angle),sin(angle));
	vec2 pt_perp_axis=vec2(-sin(angle),cos(angle));
	
	if( r<pt_len && dot(rel_pos,pt_axis)>0. && abs(dot(rel_pos,pt_perp_axis))<POINTER_WIDTH )
	{
		return true;
	}
	return false;
}

void main( void ) {

	vec2 rel_pos=gl_FragCoord.xy-vec2(CLOCK_CX,CLOCK_CY);
	float r=length(rel_pos);
	if(abs(r-CLOCK_R)<CLOCK_BORDER)
		gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
	else
	{
		float t=iDate.w;
		float h=floor(t/3600.0);
		t-=h*3600.0;if(h>12.0) h-=12.0;
		float m=floor(t/60.0);
		float s=t-m*60.0;
		float h_angle=90.0-h*360.0/12.0;
		float m_angle=90.0-m*6.0;
		float s_angle=90.0-s*6.0;
		
		if(draw_pointer(rel_pos,D2R(s_angle),0.9*CLOCK_R))
			gl_FragColor = vec4( 0.0, 1.0, 0.0, 1.0 );
		else if(draw_pointer(rel_pos,D2R(m_angle),0.7*CLOCK_R))
			gl_FragColor = vec4( 0.0, 1.0, 1.0, 1.0 );
		else if(draw_pointer(rel_pos,D2R(h_angle),0.5*CLOCK_R))
			gl_FragColor = vec4( 1.0, 1.0, 1.0, 1.0 );
		else
			gl_FragColor = vec4( 0.0 );
	}
}