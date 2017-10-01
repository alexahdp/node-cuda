typedef unsigned int uint;

struct posval { uint pos; float val; };

void  thrust_inclusiveScan_( uint *ptr1, uint *ptr2, uint *ptr3 );
void  thrust_floatSort_int_(float *ptr1, float *ptr2, uint *ptr3);
int   thrust_remove_int_(int *ptr1, int *ptr2, int value);

float thrust_reduce_floatSum_(float *ptr1, float *ptr2);
posval thrust_reduce_floatMax_(float *ptr1, float *ptr2);
posval thrust_reduce_floatMin_(float *ptr1, float *ptr2);
//void thrust_inclusiveScan_(uint *ptr1, uint *ptr2, uint *ptr3);
