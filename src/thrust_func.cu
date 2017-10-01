#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include <thrust/extrema.h>

#include "thrust_func.cuh"

typedef unsigned int uint;

// thrust::remove( ptr, ptr+n, int )

void thrust_inclusiveScan_( uint *ptr1, uint *ptr2, uint *ptr3 ){
	thrust::inclusive_scan(
		thrust::device_ptr<uint>( ptr1 ),
		thrust::device_ptr<uint>( ptr2 ),
		thrust::device_ptr<uint>( ptr3 )
	);
};

void thrust_floatSort_int_(float *ptr1, float *ptr2, uint *ptr3){
	thrust::sort_by_key(
		thrust::device_ptr<float>(ptr1),
		thrust::device_ptr<float>(ptr2),
		thrust::device_ptr<uint >(ptr3)
	);
};

int thrust_remove_int_(int *ptr1, int *ptr2, int value){
	thrust::device_ptr<int> new_end = thrust::remove(
		thrust::device_ptr<int>(ptr1),
		thrust::device_ptr<int>(ptr2),
		value
	);

	return new_end.get() - ptr1;
};


float thrust_reduce_floatSum_(float *ptr1, float *ptr2){
	return thrust::reduce(
		thrust::device_ptr<float>(ptr1),
		thrust::device_ptr<float>(ptr2),
		0.0,
		thrust::plus<float>()
	);
};

posval thrust_reduce_floatMax_(float *ptr1, float *ptr2){
	thrust::device_ptr<float> dev_ptr1 = thrust::device_ptr<float>(ptr1);
	thrust::device_ptr<float> dev_ptr2 = thrust::device_ptr<float>(ptr2);
	thrust::device_ptr<float> max_ptr  = thrust::max_element( dev_ptr1, dev_ptr2 );

	posval rv;

	rv.pos = &max_ptr[0] - &dev_ptr1[0];
	// не понятно, как он считывает данные из памяти видеокарты?
	rv.val = max_ptr[0];

	return rv;
};

posval thrust_reduce_floatMin_(float *ptr1, float *ptr2){
	thrust::device_ptr<float> dev_ptr1 = thrust::device_ptr<float>(ptr1);
	thrust::device_ptr<float> dev_ptr2 = thrust::device_ptr<float>(ptr2);
	thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr1, dev_ptr2);

	posval rv;

	rv.pos = &min_ptr[0] - &dev_ptr1[0];
	// не понятно, как он считывает данные из памяти видеокарты?
	rv.val = min_ptr[0];

	return rv;
};
