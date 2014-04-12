#ifdef GL_ES
precision mediump float;
#endif

uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;

void main( void ) {
	vec2 pos = ( gl_FragCoord.xy / resolution.xy );
	float rec_sz = (10.0)/resolution.x;
	if(abs(pos.x-mouse.x)<rec_sz&&abs(pos.y-mouse.y)<rec_sz)
		gl_FragColor=vec4(1.0,0.0,0.0,0.0);
	else
		gl_FragColor=vec4(0.0);

}