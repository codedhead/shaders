// learning to do some sin wave

#ifdef GL_ES
precision mediump float;
#endif

uniform float time;

uniform vec2 resolution;

#define PI 3.14159

void main( void ) {

	vec2 p = ( gl_FragCoord.xy / resolution.xy )-0.5;
	
	float scl = 0.1*sin(time*2.0);
	float sx = scl*sin( 25.0 * p.x  -time );
	
	float dy = 1./ ( 100. * abs(p.y - sx));
	
	float c_scl = (scl+0.1)*5.0;
	gl_FragColor = vec4( c_scl*(0.5) * dy, 0.5 * dy, c_scl*dy, 1.0 );
