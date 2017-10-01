#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

// вообще, на время компиляции сокращение тела - никак не повлияло. поэтому если понадобится "настоящий" - легко можно вернуть

typedef unsigned int uint;


// int2 functions
////////////////////////////////////////////////////////////////////////////////

// addition
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}


// negate
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}

// addition
inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ int2 operator*(int2 a, int s)
{
    return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ int2 operator*(int s, int2 a)
{
    return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(int2 &a, int s)
{
    a.x *= s; a.y *= s;
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float2 operator/(float s, float2 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}

// length
inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);  // discards w
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}


// 2015-11-14
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}


// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

inline __host__ __device__ void atomicAdd(float3 *a, float3 b)
{ 
    atomicAdd( &(a->x), b.x );
    atomicAdd( &(a->y), b.y );
    atomicAdd( &(a->z), b.z );
}

inline __host__ __device__ void atomicAdd(float2 *a, float2 b)
{ 
    atomicAdd( &(a->x), b.x );
    atomicAdd( &(a->y), b.y );
}

inline __host__ __device__ bool isfinite(float3 a)
{ 
	return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}

inline __host__ __device__ bool isfinite(float2 a)
{ 
	return isfinite(a.x) && isfinite(a.y);
}

inline __device__ float p(float d,float t) {
	float v;
	
	v = 2.0f - abs(d-t)*6.0f;
	v = v<-2.0f ? -2.0f-v : v;
	v = v<1.0f ? v>0.0f ? v : 0.0f: 1.0f;
	
	return v;
};


inline __device__ float3 t2rgb(float t){
	float3 rgb;
	
	t -= floor(t);
	
	rgb.x = p(3.0f/3.0f,t);
	rgb.y = p(1.0f/3.0f,t);
	rgb.z = p(2.0f/3.0f,t);
	
	return rgb;
}

inline __device__ float3 tw2rgbw(float t,float w){
	float3 rgb;
	
	t -= floor(t);
	
	rgb.x = p(3.0f/3.0f,t);
	rgb.y = p(1.0f/3.0f,t);
	rgb.z = p(2.0f/3.0f,t);
	
	rgb.x += (1 - rgb.x) * w;
	rgb.y += (1 - rgb.y) * w;
	rgb.z += (1 - rgb.z) * w;
	
	return rgb;
}

inline __device__ float2 crt2xy( float2 c, float r, float t ){
	float a  = t * 2.0 * 3.14159265359;
	
	c.x     += cos( a ) * r;
	c.y     += sin( a ) * r;
	
	return c;
};

inline __device__ float2 cri2xy( float2 c, float rr, int i ){
	float2 rv = { 0,0 };
	
	if (i == 0){
		rv    = c;
		rv.x += rr;
		return rv;
	};
	
	// константа!
	float Fi = ( sqrtf( 5 ) - 1) / 2;
	
	float r  = sqrtf( i ) * Fi * rr * 2;
	return crt2xy( c, r, Fi * i );
}

inline __device__ float2 rt2xy( float r, float t ){
	float a  = t * 2 * 3.14159265359;
	
	float2 c = { cos( a ) * r, sin( a ) * r };
	
	return c;
};

inline __device__ float2 ri2xy( float rr, int i ){
	float2 rv = { 0,0 };
	
	if (i == 0){
		rv.x += rr;
		return rv;
	};
	
	// константа!
	// float Fi = ( sqrt( 5 ) - 1 ) / 2;
	
	double Fi = 0.6180339887498948482;
	float r   = sqrtf( i ) * Fi * rr * 2;
	
	double a  = Fi * i;
	// комментированием этой строки большой подсолнечник будет разрушен
	// т.к. при передаче в rt2xy  double превратится в float и обрежет точное значение сдвига
	a        -= floor(a);
	
	return rt2xy( r, a );
}

#endif
