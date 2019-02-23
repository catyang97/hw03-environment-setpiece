#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

int maxSteps = 500;
float epsilon = 0.0001;
float maxDist = 100.0;
const float PI = 3.14159265359;


float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

float interpNoise2D(float x, float y) { // from slides
    float intX = floor(x);
    float fractX = fract(x);
    float intY = floor(y);
    float fractY = fract(y);

    float v1 = random1(vec2(intX, intY), vec2(1.f, 1.f));
    float v2 = random1(vec2(intX + 1.0f, intY), vec2(1.f, 1.f));
    float v3 = random1(vec2(intX, intY + 1.0f), vec2(1.f, 1.f));
    float v4 = random1(vec2(intX + 1.0, intY + 1.0), vec2(1.f, 1.f));

    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);

    return mix(i1, i2, fractY);
}

float fbm(float x, float y) { // from slides
  float total = 0.0f;
  float persistence = 0.5f;
  float octaves = 10.0;

  for (float i = 0.0; i < octaves; i = i + 1.0) {
      float freq = pow(2.0f, i);
      float amp = pow(persistence, i);
      total += interpNoise2D(x * freq, y * freq) * amp;
  }
  return total;
}

// Worley-Perlin Noise from https://www.shadertoy.com/view/MdGSzt
vec3 hash(vec3 p3) {
	p3 = fract(p3 * vec3(0.1031, 0.11369, 0.13787));
  p3 += dot(p3, p3.yxz + 20.0);
  return -1.0 + 2.0 * fract(vec3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

float perlinNoise(vec3 point) {
  vec3 pi = floor(point);
  vec3 pf = point - pi;
  vec3 w = pf * pf * (3.0 - 2.0 * pf);
  
  return 	mix(mix(mix(dot(pf - vec3(0, 0, 0), hash(pi + vec3(0, 0, 0))), 
                      dot(pf - vec3(1, 0, 0), hash(pi + vec3(1, 0, 0))),
                      w.x),
                  mix(dot(pf - vec3(0, 0, 1), hash(pi + vec3(0, 0, 1))), 
                      dot(pf - vec3(1, 0, 1), hash(pi + vec3(1, 0, 1))),
                      w.x),
                  w.z),
              mix(mix(dot(pf - vec3(0, 1, 0), hash(pi + vec3(0, 1, 0))), 
                      dot(pf - vec3(1, 1, 0), hash(pi + vec3(1, 1, 0))),
                      w.x),
                  mix(dot(pf - vec3(0, 1, 1), hash(pi + vec3(0, 1, 1))), 
                      dot(pf - vec3(1, 1, 1), hash(pi + vec3(1, 1, 1))),
                      w.x),
                w.z),
              w.y);
}

// Toolbox Function
float squareWave(float x, float freq, float amp) {
    return abs(mod(floor(x * freq), 2.0) * amp);
}

// Combinations
float opUnion(float d1, float d2) { return min(d1,d2); }

float opSubtraction(float d1, float d2) { return max(-d1,d2); }

float opIntersection(float d1, float d2) { return max(d1,d2); }

float opSmoothUnion(float d1, float d2, float k) {
  float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
  return mix( d2, d1, h ) - k*h*(1.0-h); 
}

float opSmoothSubtraction(float d1, float d2, float k) {
  float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
  return mix( d2, -d1, h ) + k*h*(1.0-h); 
}

// Primitives
float sdBox(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0); 
}

float sdRoundBox(vec3 p, vec3 b, float r) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}

float sdRoundedCylinder(vec3 p, float ra, float rb, float h) {
    vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float sdTriPrism(vec3 p, vec2 h) {
  vec3 q = abs(p);
  return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

// Rotation Matrices
mat3 rotX(float angle) {
  return mat3(vec3(1, 0, 0),
              vec3(0, cos(angle), -sin(angle)),
              vec3(0, sin(angle), cos(angle)));
}

mat3 rotY(float angle) {
  return mat3(vec3(cos(angle), 0, sin(angle)),
            vec3(0, 1, 0),
            vec3(-sin(angle), 0, cos(angle)));
}

mat3 rotZ(float angle) {
  return mat3(vec3(cos(angle), sin(angle), 0),
              vec3(-sin(angle), cos(angle), 0),
              vec3(0, 0, 1));
}

// Component SDFs
float roofSDF(vec3 point) {
  vec3 roofTrans = vec3(0, -0.5, 0);
  float square = squareWave(point.y, 5.0, 0.8);
  float square2 = squareWave((point.z + point.x) / 2.0, 5.0, 0.8);
  float roof = opIntersection(sdTriPrism((point * rotY(PI*3.0/4.0) + roofTrans + vec3(square2/100.0))*vec3(1.0, 1.2, 1.0), vec2(1.3, 1.2)), sdTriPrism((point * rotY(PI*5.0/4.0) + roofTrans + vec3(square/100.0))*vec3(1.0, 1.2, 1.0), vec2(1.3, 1.2)));
  return roof;
}

float houseSDF(vec3 point) {
  vec3 baseTrans = vec3(0, 1.0, 0);
  float base = sdRoundBox(point * rotY(PI*3.0/4.0) + baseTrans, vec3(1.0, 1.0, 1.0), 0.04);

  vec3 doorTrans = vec3(0.6, 1.2, 0);
  float door = sdBox(point * rotY(PI*3.0/4.0) + doorTrans, vec3(0.5, 0.8, 0.4));
  base = opSubtraction(door, base);
  return base;
}

float fieldSDF(vec3 point) {
  vec3 help = vec3(0.0, perlinNoise(point), 0.1);
  float fieldDist = sdBox(point + help / 18.0  + vec3(0.0, 2.5, -6.5), vec3(40.0,1.0,15.0));
  float hole = sdRoundedCylinder(point + vec3(-4.5, 2.5, 4.2), 1.5 + perlinNoise(point) / 15.0, 1.0 + perlinNoise(point) / 10.0, 0.8);
  return opSmoothSubtraction(hole, fieldDist, 0.3);
}

// Noise Function from https://www.shadertoy.com/view/MsB3WR
const mat2 m2 = mat2( 0.60, -0.80, 0.80, 0.60 );
float waterMap( vec2 pos ) {
	vec2 posm = pos * m2;
	vec3 inside = vec3( 8.*posm, u_Time );
	return abs( fbm(inside.x, inside.y)-0.5 )* 0.1;
}

float lakeSDF(vec3 point) {
  // vec3 help = vec3(0.0, perlinNoise(point), 0.1);
  vec3 help = vec3(0.0, pow(waterMap(point.xz), 1.2), 0.1);

  float lakeDist = sdBox(point + help / 50.0 + vec3(-4.5, 2.7, 4.2), vec3(5.0,1.0,5.0));
  return lakeDist;
}

// Noise functions from https://www.shadertoy.com/view/XdsGD7
float snoise( vec2 p ) {
	vec2 f = fract(p);
	p = floor(p);
	float v = p.x+p.y*1000.0;
	vec4 r = vec4(v, v+1.0, v+1000.0, v+1001.0);
	r = fract(100000.0*sin(r*.001));
	f = f*f*(3.0-2.0*f);
	return 2.0*(mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y))-1.0;
}

float terrain( vec2 p, int octaves ) {	
	float h = 0.0; // height
	float w = 0.5; // octave weight
	float m = 0.4; // octave multiplier
	for (int i=0; i<16; i++) {
		if (i<octaves) {
			h += w * snoise((p * m));
		}
		else break;
		w *= 0.5;
		m *= 2.0;
	}
	return h;
}

float mountainSDF(vec3 point) {
  float mountDist =0.0;
  if (point.z > 19.0) {
    float h = terrain(vec2(point.x, point.z), 4);
    h += smoothstep(-0.3, 4.5, h)*2.0; // exaggerate the higher terrain
    h *= smoothstep(-1.5, -0.3, h); // smooth out the lower terrain
    h *= sin(point.x)*sin(point.z);
    vec3 help = vec3(0.0, h, 0.1);
    mountDist = sdBox(point + help  + vec3(0.0, 1.7, -20.0), vec3(40.0,0.5,2.0));
  } else {
    mountDist = sdBox(point  + vec3(0.0, 1.9, -20.0), vec3(40.0,0.5,2.0));
  }

  return mountDist;
}

// Colors and Normals for Components
vec4 roofColor(vec3 p) {
    vec3 darkTile = vec3(0.83, 0.39, 0.31);
    vec3 lightTile = vec3(0.95, 0.61, 0.44);
    float square = squareWave(p.y, 5.0, 0.8);
    float square2 = squareWave((p.z + p.x) / 2.0, 5.0, 0.8);
    float fbm = smoothstep(fract(fbm(p.x, p.y) * 80.0), fract(fbm(p.z, p.y) * 80.0), 0.5);

    vec3 horizontal = mix(darkTile / 2.0, lightTile / 2.0, square + fbm / 5.0);
    vec3 vertical = mix(lightTile / 2.0, darkTile / 2.0, square2);

    return vec4((horizontal + vertical) * 0.8, 1.0);
}

vec3 roofNormal(vec3 p) {
  return normalize(vec3(
      roofSDF(vec3(p.x + epsilon, p.y, p.z)) - roofSDF(vec3(p.x - epsilon, p.y, p.z)),
      roofSDF(vec3(p.x, p.y + epsilon, p.z)) - roofSDF(vec3(p.x, p.y - epsilon, p.z)),
      roofSDF(vec3(p.x, p.y, p.z  + epsilon)) - roofSDF(vec3(p.x, p.y, p.z - epsilon))
  )); 
}

vec4 houseColor(vec3 p) {
  vec3 darkPaint = vec3(0.93, 0.86, 0.73);
  vec3 lightPaint = vec3(0.93, 0.92, 0.85);
  float fbm = smoothstep(fract(fbm(p.x, p.y) * 80.0), fract(fbm(p.z, p.y) * 80.0), 0.5);
  return vec4(mix(darkPaint, lightPaint, fbm), 1.0);
}

vec3 houseNormal(vec3 p) {
  return normalize(vec3(
      houseSDF(vec3(p.x + epsilon, p.y, p.z)) - houseSDF(vec3(p.x - epsilon, p.y, p.z)),
      houseSDF(vec3(p.x, p.y + epsilon, p.z)) - houseSDF(vec3(p.x, p.y - epsilon, p.z)),
      houseSDF(vec3(p.x, p.y, p.z  + epsilon)) - houseSDF(vec3(p.x, p.y, p.z - epsilon))
  ));
}

vec4 fieldColor(vec3 point) {
  return vec4(0.44, 0.89, 0.34, 1.0);
}

vec3 fieldNormal(vec3 p) {
  return normalize(vec3(
      fieldSDF(vec3(p.x + epsilon, p.y, p.z)) - fieldSDF(vec3(p.x - epsilon, p.y, p.z)),
      fieldSDF(vec3(p.x, p.y + epsilon, p.z)) - fieldSDF(vec3(p.x, p.y - epsilon, p.z)),
      fieldSDF(vec3(p.x, p.y, p.z  + epsilon)) - fieldSDF(vec3(p.x, p.y, p.z - epsilon))
  ));
}

vec4 mountainColor(vec3 point) {
  return vec4(0.68, 0.68, 0.68, 1.0);
}

vec3 mountainNormal(vec3 p) {
  return normalize(vec3(
      mountainSDF(vec3(p.x + epsilon, p.y, p.z)) - mountainSDF(vec3(p.x - epsilon, p.y, p.z)),
      mountainSDF(vec3(p.x, p.y + epsilon, p.z)) - mountainSDF(vec3(p.x, p.y - epsilon, p.z)),
      mountainSDF(vec3(p.x, p.y, p.z  + epsilon)) - mountainSDF(vec3(p.x, p.y, p.z - epsilon))
  ));
}

vec4 lakeColor(vec3 point) {
  return vec4(0.09, 0.5, 0.68, 1.0);
}

vec3 lakeNormal(vec3 p) {
  return normalize(vec3(
      lakeSDF(vec3(p.x + epsilon, p.y, p.z)) - lakeSDF(vec3(p.x - epsilon, p.y, p.z)),
      lakeSDF(vec3(p.x, p.y + epsilon, p.z)) - lakeSDF(vec3(p.x, p.y - epsilon, p.z)),
      lakeSDF(vec3(p.x, p.y, p.z  + epsilon)) - lakeSDF(vec3(p.x, p.y, p.z - epsilon))
  ));
}

float sceneSDF(vec3 point, out vec4 color, out vec3 normal, out bool hitWater) {
  float roofDist = roofSDF(point);
  float houseDist = houseSDF(point);
  float fieldDist = fieldSDF(point);
  float mountDist = mountainSDF(point);
  float lakeDist = lakeSDF(point);

  if (roofDist < houseDist && roofDist < fieldDist && roofDist < mountDist && roofDist < lakeDist) {
    color = roofColor(point);
    normal = roofNormal(point);
    return roofDist;
  } else if (houseDist < fieldDist && houseDist < mountDist && houseDist < lakeDist) {
    color = houseColor(point);
    normal = houseNormal(point);
    return houseDist;
  } else if (fieldDist < mountDist && fieldDist < lakeDist) {
    color = fieldColor(point);
    normal = fieldNormal(point);
    return fieldDist;
  } else if (mountDist < lakeDist) {
    color = mountainColor(point);
    normal = mountainNormal(point);
    return mountDist;
  } else {
    color = lakeColor(point);
    normal = lakeNormal(point);
    hitWater = true;
    return lakeDist;
  }
}

float sceneSDFPlain(vec3 point) {
  float roofDist = roofSDF(point);
  float houseDist = houseSDF(point);
  float fieldDist = fieldSDF(point);
  float mountDist = mountainSDF(point);
  float lakeDist = lakeSDF(point);

  if (roofDist < houseDist && roofDist < fieldDist && roofDist < mountDist && roofDist < lakeDist) {
    return roofDist;
  } else if (houseDist < fieldDist && houseDist < mountDist && houseDist < lakeDist) {
    return houseDist;
  } else if (fieldDist < mountDist && fieldDist < lakeDist) {
    return fieldDist;
  } else if (mountDist < lakeDist) {
    return mountDist;
  } else {
    return lakeDist;
  }
}

vec4 skyColor() {
  vec3 skyColor, cloudColor;
  float clouds = fbm(fs_Pos.x + (u_Time / 100.0), fs_Pos.y);
  skyColor = vec3(113.0 / 255.0, 193.0 / 255.0, 252.0 / 255.0);
  cloudColor = vec3(1.0, 1.0, 1.0);
  clouds -= 0.7;
  return vec4(clouds * skyColor * 0.9 + (1.0 - clouds) * cloudColor, 1.0);
}

float rayMarch(vec3 rayDir, out vec4 color, out vec3 normal, out bool hitWater) {
  float depth = 0.0;
  for (int i = 0; i < maxSteps; i++) {
    vec3 norm;
    float dist = sceneSDF(u_Eye + depth * rayDir, color, norm, hitWater);
    if (dist < epsilon) {
        // We're inside the scene surface!
        normal = norm;
        return depth;
    }
    // Move along the ray
    depth += dist;

    if (depth >= maxDist) {
        // Gone too far
        normal = vec3(0.0, 0.0, 0.0);
        return maxDist;
    }
  }
  normal = vec3(0.0, 0.0, 0.0);
  return maxDist;
}

vec3 castRay() {
  vec3 F = normalize(u_Ref - u_Eye);
  vec3 R = normalize(cross(F, u_Up));
  float len = length(u_Ref - u_Eye);
  float aspect = u_Dimensions.x / u_Dimensions.y;
  float fov = radians(75.0);

  vec3 V = u_Up * len * tan(fov / 2.0);
  vec3 H = R * len * aspect * tan(fov / 2.0);
  vec3 p = u_Ref + fs_Pos.x * H + fs_Pos.y * V;
  vec3 ray_Dir = normalize(p - u_Eye);
  return ray_Dir;
}

// IQ Reference: https://iquilezles.org/www/articles/rmshadows/rmshadows.htm
float softShadow(vec3 ro, vec3 rd, float minT, float maxT, float k) {
    float res = 1.0;
    for(float t = minT; t < maxT;) {
        float d = sceneSDFPlain(ro + t * rd);
        if (d < epsilon) {
            return 0.0;
        }
        res = min(res, k * d / t);
        t += d;
    }
    return res;
}

vec4 getColor(vec3 color, vec3 normal, vec3 lightDir) {
  vec4 diffuseColor = vec4(color, 1.0);
  float diffuseTerm = dot(normalize(normal), normalize(lightDir));
  // diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
  float ambientTerm = 0.2;
  float lightIntensity = diffuseTerm + ambientTerm; 
  return vec4(diffuseColor.rgb * lightIntensity, diffuseColor.a);
}

void main() {
  vec3 dir = castRay();
  vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
  vec3 normal = vec3(1.0, 1.0, 1.0);
  bool hitWater = false;
  float depth = rayMarch(dir, color, normal, hitWater);

  // Lights from https://www.shadertoy.com/view/td2GD3
  vec4 lights[3];
  vec3 lightColor[3];
  
  // Light positions with intensity as w-component
  lights[0] = vec4(6.0, 5.0, -7.0, 1.0); // key light
  lights[1] = vec4(4.0, 5.0, -7.0, 1.0); // fill light
  lights[2] = vec4(6.0, 7.0, -7.0, 1.2); // back light
  
  lightColor[0] = vec3(1.0, 240.0 / 255.0, 198.0 / 255.0 );
  lightColor[1] = vec3(207.0 / 255.0, 222.0 / 255.0, 247.0 / 255.0);
  lightColor[2] = vec3(1.0, 240.0 / 255.0, 198.0 / 255.0 );

  if (depth < maxDist) {
    vec4 difColor = color;
    vec3 pos = u_Eye + depth * dir;
    vec3 sum = vec3(0.0f);
    for (int i = 0; i < 3; i++) {
      vec3 lambert = getColor(difColor.xyz, normal, lights[0].xyz).xyz;
      sum += lambert * softShadow(pos, normalize(lights[i].xyz - pos), 0.2, 10.0, 12.0) *lights[i].w * lightColor[i]; 
    }

    out_Col = vec4(sum / 3.0, 1.0);
    vec4 sky = skyColor();
    out_Col += (sky / 10.0);
  } else {
    out_Col = skyColor();
  }
}


