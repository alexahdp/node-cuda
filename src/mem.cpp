#include <cstring>
#include <node_buffer.h>
#include "mem.hpp"
#include <stdio.h>

#include "thrust_func.cuh"


#define DEBUG_MSG 0

typedef unsigned int uint;
//typedef uint GLuint;

using namespace v8;
using namespace NodeCuda;

Persistent<v8::Function> Mem::constructor;

//#include <chrono>
//#include <thread>

//GLFWwindow* Mem::window = NULL;


void mylog(char *txt) {
    FILE *ofp = fopen("../cuda-mem-log.txt", "a");
    fprintf(ofp, "%s\n", txt);
    fclose(ofp);
}

#define USE_GL 0

void Mem::Initialize(Handle<Object> target) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    
  Local<FunctionTemplate> t = FunctionTemplate::New(isolate, Mem::New);

  t->InstanceTemplate()->SetInternalFieldCount(1);
  t->SetClassName(String::NewFromUtf8(isolate,"CudaMem"));

  // Mem objects can only be created by allocation functions
  NODE_SET_METHOD(target, "memAlloc", Mem::Alloc);
  NODE_SET_METHOD(target, "memAllocPitch", Mem::AllocPitch);

  NODE_SET_METHOD(target, "memVBO", Mem::AllocVBO);
  NODE_SET_METHOD(target, "glGetArrayBufferBinding", Mem::glGetArrayBufferBinding);

  
  NODE_SET_METHOD(target, "thrust_inclusiveScan", Mem::thrust_inclusiveScan);
  NODE_SET_METHOD(target, "thrust_reduce_floatSum", Mem::thrust_reduce_floatSum);
  NODE_SET_METHOD(target, "thrust_reduce_floatMax", Mem::thrust_reduce_floatMax);
  NODE_SET_METHOD(target, "thrust_reduce_floatMin", Mem::thrust_reduce_floatMin);
  NODE_SET_METHOD(target, "thrust_floatSort_int", Mem::thrust_floatSort_int);
  NODE_SET_METHOD(target, "thrust_remove_int", Mem::thrust_remove_int);

  //NODE_SET_METHOD(target, "getContext", Mem::getContext);
  //NODE_SET_METHOD(target, "makeContextCurrent", Mem::makeContextCurrent);


  t->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "devicePtr"), Mem::GetDevicePtr);

  NODE_SET_PROTOTYPE_METHOD(t, "free"    , Mem::Free);
  NODE_SET_PROTOTYPE_METHOD(t, "copyHtoD", Mem::CopyHtoD);
  NODE_SET_PROTOTYPE_METHOD(t, "copyDtoH", Mem::CopyDtoH);
  NODE_SET_PROTOTYPE_METHOD(t, "regVBO"  , Mem::RegVBO);
  NODE_SET_PROTOTYPE_METHOD(t, "unregVBO", Mem::UnregVBO);
  
  constructor.Reset(isolate, t->GetFunction());

  char buf[256];
#if USE_GL

   if (!glfwInit())
   {
       sprintf(buf, "\n\n-------------\n ERROR: !glfwInit()"); mylog(buf);
       return;
   }
 
   /* Create a windowed mode window and its OpenGL context */
   glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
   window = glfwCreateWindow(640, 480, "CUDA WIND", NULL, NULL);
   fprintf(stdout, "node_cuda::glfwCreateWindow = 0x%X\n", window);
   if (!window)
   {
       glfwTerminate();
      
       sprintf(buf, "\n\n-------------\n ERROR: !window"); mylog(buf);
       return;
   }
#endif

#if USE_GL 
  //BOOL slok = wglShareLists(host_gl_context, m_context_id);
  //context_inst->make_current();
  fprintf(stdout, "node_cuda::glfwGetWGLContext = 0x%X\n", glfwGetWGLContext(window));
  sprintf(buf, "\n\n-------------\ncuda opengl context 0x%X", glfwGetWGLContext(window)); mylog(buf);

#endif


}

void Mem::New(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

  Mem *pmem = new Mem();
  pmem->Wrap(args.This());
    
  args.GetReturnValue().Set( args.This() );
}


/*
void cuda_vbo_reg(GLuint *vbo, void **cuda, cudaGraphicsResource **cuda_vbo){
    cudaGraphicsGLRegisterBuffer(cuda_vbo, *vbo, cudaGraphicsMapFlagsNone);
    cudaGraphicsMapResources(1, cuda_vbo, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer(cuda, &num_bytes, *cuda_vbo);
};
*/



void cuda_vbo_reg(GLuint vboId2, void **cuda, cudaGraphicsResource **cuda_vbo)
{
    cudaError_t err;
    GLuint vboId = vboId2;
    //cudaGLSetGLDevice(0);
    //fprintf(stdout, "cuda_vbo: %i, vboId: %i, cudaGraphicsMapFlagsNone: %i\n", cuda_vbo, vboId, cudaGraphicsMapFlagsNone);
    //mylog(glGetString(GL_VERSION));
    //mylog((char)vboId );
    
    // alexahdp - только что закомментил!
    
    err = cudaGraphicsGLRegisterBuffer(cuda_vbo, vboId, cudaGraphicsMapFlagsNone);
    
    err = cudaGraphicsMapResources(1, cuda_vbo, 0);
    size_t num_bytes;
    err = cudaGraphicsResourceGetMappedPointer(cuda, &num_bytes, *cuda_vbo);
    
};

void cuda_vbo_unreg(cudaGraphicsResource **cuda_vbo){
    cudaGraphicsUnmapResources(1, cuda_vbo, 0);
    cudaGraphicsUnregisterResource(*cuda_vbo);
};


void Mem::glGetArrayBufferBinding(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    
    //char *argv[] = { "Test" };
    //int   argc = 1;
    //glutInit(&argc, argv);
    
    //glewInit();
    
    //GLuint vbo1;
    //glGenBuffers(1, &vbo1);
    
    //glBindBuffer(GL_ARRAY_BUFFER, vbo1);
    //glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(Space::Ball) * LIGHT->space->Bcmax, 0, GL_DYNAMIC_DRAW_ARB);

    // GLint vbo;
    // glBindBuffer(GL_ARRAY_BUFFER, 1);
    
    //GLint vbo;
    //glGetIntegerv( GL_ARRAY_BUFFER_BINDING, &vbo);
    //
    ////GLuint vbo = reinterpret_cast<GLuint>(vbo_int);
    //
    //args.GetReturnValue().Set( Integer::New(isolate,vbo) );
    
};

/*
void Mem::RegVBO(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());
    
    // Local<Object> buf = args[0]->ToObject();
    // GLuint *vbo = static_cast<GLuint*>(buf->GetIndexedPropertiesExternalArrayData());
    // WebGLBuff
    // unsigned int arg1 = args[0]->Uint32Value();
    // Buffer::Data( buf );
    
    Local<Object> obj = args[0]->ToObject();
    //if (obj->GetIndexedPropertiesExternalArrayDataType() != kExternalFloatArray) return;
    //int len = obj->GetIndexedPropertiesExternalArrayDataLength();
    GLuint *vbo = static_cast<GLuint*>(obj->GetIndexedPropertiesExternalArrayData());
    //GLuint vbo = args[0]->Uint32Value();
    
    // поискать имплементацию v8 webgl CreateBuffer'а, как он читает Float32Array
    
    cuda_vbo_reg(vbo, (void **)&(pmem->m_devicePtr), &(pmem->vbo));
    
    args.GetReturnValue().Set( Undefined( isolate ) );
}
*/


void Mem::RegVBO(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    
    //glfwMakeContextCurrent(window);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());
    GLuint vboId = args[0]->Uint32Value();
    
    //mylog((char)args[0]->Uint32Value());
    
    //fprintf(stdout, "node_cuda::glfwCreateWindow = 0x%X\n", window);
    //fprintf(stdout, "here");
    // зачем по ссылке?
    //cuda_vbo_reg(&vboId, (void **)&(pmem->m_devicePtr), &(pmem->vbo));
    cuda_vbo_reg(vboId, (void **)&(pmem->m_devicePtr), &(pmem->vbo));
    
    args.GetReturnValue().Set( Undefined( isolate ) );
}



void Mem::UnregVBO(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());
    cuda_vbo_unreg(&(pmem->vbo));

    args.GetReturnValue().Set(Undefined(isolate));
}



void Mem::thrust_reduce_floatSum(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem1 = ObjectWrap::Unwrap<Mem>(args[0]->ToObject());
    uint cnt = args[1]->Uint32Value();

    float *ptr1 = (float *)(pmem1->m_devicePtr);
    
    float result = thrust_reduce_floatSum_(ptr1, ptr1 + cnt);
    
    args.GetReturnValue().Set(Number::New(isolate, result ));
};

void Mem::thrust_reduce_floatMax(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem1 = ObjectWrap::Unwrap<Mem>(args[0]->ToObject());
    uint cnt = args[1]->Uint32Value();

    float *ptr1 = (float *)(pmem1->m_devicePtr);

    posval rv = thrust_reduce_floatMax_(ptr1, ptr1 + cnt);
    Handle<Array> ra = Array::New(isolate,2);
    ra->Set(0, Integer::New( isolate, rv.pos ));
    ra->Set(1, Number ::New( isolate, rv.val ));
    args.GetReturnValue().Set( ra );
};

void Mem::thrust_reduce_floatMin(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem1 = ObjectWrap::Unwrap<Mem>(args[0]->ToObject());
    uint cnt = args[1]->Uint32Value();

    float *ptr1 = (float *)(pmem1->m_devicePtr);
    
    posval rv = thrust_reduce_floatMin_(ptr1, ptr1 + cnt);
    Handle<Array> ra = Array::New(isolate, 2);
    ra->Set(0, Integer::New(isolate, rv.pos));
    ra->Set(1, Number::New(isolate, rv.val));
    args.GetReturnValue().Set(ra);
};



void Mem::thrust_inclusiveScan(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);
    
    Mem *pmem1 = ObjectWrap::Unwrap<Mem>( args[0]->ToObject() );
    uint cnt   = args[1]->Uint32Value();
    Mem *pmem2 = ObjectWrap::Unwrap<Mem>( args[2]->ToObject() );
    
    uint *ptr1  = (uint *)(pmem1->m_devicePtr);
    uint *ptr2  = (uint *)(pmem2->m_devicePtr);
    
    thrust_inclusiveScan_( ptr1, ptr1 + cnt, ptr2 );
    
    /*
    thrust::inclusive_scan(
        thrust::device_ptr<uint>( ptr1 ),
        thrust::device_ptr<uint>( ptr1 + cnt ),
        thrust::device_ptr<uint>( ptr2 )
    );
    */
    
    //args.GetReturnValue().Set(Number::New(isolate,error));
}

void Mem::thrust_floatSort_int(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem1 = ObjectWrap::Unwrap<Mem>(args[0]->ToObject());
    uint cnt = args[1]->Uint32Value();
    Mem *pmem2 = ObjectWrap::Unwrap<Mem>(args[2]->ToObject());
    
    float  *ptr1 = (float *)(pmem1->m_devicePtr);
    uint   *ptr2 = (uint  *)(pmem2->m_devicePtr);
    
    thrust_floatSort_int_(ptr1, ptr1 + cnt, ptr2);
}

void Mem::thrust_remove_int(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem1 = ObjectWrap::Unwrap<Mem>(args[0]->ToObject());
    uint cnt = args[1]->Uint32Value();

    int    *ptr1 = (int *)(pmem1->m_devicePtr);
    int    value = args[2]->Uint32Value();

    int    new_end = thrust_remove_int_(ptr1, ptr1 + cnt, value);
    
    args.GetReturnValue().Set(Integer::New(isolate, new_end ));

}


void Mem::AllocVBO(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    v8::Local<v8::Function> cons = v8::Local<v8::Function>::New(isolate, constructor);
    Local<Object> result = cons->NewInstance();

    Mem *pmem = ObjectWrap::Unwrap<Mem>(result);

    args.GetReturnValue().Set(result);
}


void Mem::Alloc(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    //Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
    v8::Local<v8::Function> cons = v8::Local<v8::Function>::New(isolate, constructor);
    Local<Object> result = cons->NewInstance();
    
  Mem *pmem = ObjectWrap::Unwrap<Mem>(result);

  size_t bytesize = args[0]->Uint32Value();

  //fprintf(stdout, "Mem::Alloc %d\n", bytesize);
  
  CUresult error = cuMemAlloc(&(pmem->m_devicePtr), bytesize);

  result->Set(String::NewFromUtf8(isolate,"size"), Integer::NewFromUnsigned(isolate,bytesize));
  result->Set(String::NewFromUtf8(isolate,"error"), Integer::New(isolate,error));

  args.GetReturnValue().Set(result);
}

void Mem::AllocPitch(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    //Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
    
    v8::Local<v8::Function> cons = v8::Local<v8::Function>::New(isolate, constructor);
    Local<Object> result = cons->NewInstance();

  Mem *pmem = ObjectWrap::Unwrap<Mem>(result);

  size_t pPitch;
  unsigned int ElementSizeBytes = args[2]->Uint32Value();
  size_t WidthInBytes = ElementSizeBytes * args[0]->Uint32Value();
  size_t Height = args[1]->Uint32Value();
  CUresult error = cuMemAllocPitch(&(pmem->m_devicePtr), &pPitch, WidthInBytes, Height, ElementSizeBytes);

  result->Set(String::NewFromUtf8(isolate,"size"), Integer::NewFromUnsigned(isolate,pPitch * Height));
  result->Set(String::NewFromUtf8(isolate,"pitch"), Integer::NewFromUnsigned(isolate,pPitch));
  result->Set(String::NewFromUtf8(isolate,"error"), Integer::New(isolate,error));

  args.GetReturnValue().Set(result);
}

void Mem::Free(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());

  CUresult error = cuMemFree(pmem->m_devicePtr);

  args.GetReturnValue().Set(Number::New(isolate,error));
}

void Mem::CopyHtoD(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());

    Local<Float32Array> buff32 = args[0]->ToObject().As<Float32Array>();
  ArrayBuffer::Contents buff32c = buff32->Buffer()->GetContents();

  const size_t buff32_offset = buff32->ByteOffset();
  const size_t buff32_length = buff32->ByteLength();
  char* buff32_data = static_cast<char*>(buff32c.Data()) + buff32_offset;

  //fprintf(stdout, "CopyHtoD: buff32_length = %d\n", buff32_length);
  //fprintf(stdout, "CopyHtoD: buff32_data = %d\n", buff32_data);
 
  char *phost = buff32_data;
  //size_t bytes = Buffer::Length(buf);
  //size_t bytes = args[1]->Int32Value();
  size_t bytes = args[1]->IntegerValue();

  //fprintf(stdout, "CopyHtoD: bytes = %d\n", bytes);
  
  bool async = args.Length() >= 2 && args[1]->IsTrue();

  //fprintf(stdout, "CopyHtoD: async = %d\n", async);
  //fprintf(stdout, "CopyHtoD: pmem->m_devicePtr = 0x%X\n", pmem->m_devicePtr);

  CUresult error;
  if (async) {
    error = cuMemcpyHtoDAsync(pmem->m_devicePtr, phost, bytes, 0);
  } else {
    error = cuMemcpyHtoD(pmem->m_devicePtr, phost, bytes);
  }

  args.GetReturnValue().Set(Number::New(isolate, error));
}

void Mem::CopyDtoH(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(args.This());


  Local<Float32Array> buff32 = args[0]->ToObject().As<Float32Array>();
  ArrayBuffer::Contents buff32c = buff32->Buffer()->GetContents();

  const size_t buff32_offset = buff32->ByteOffset();
  const size_t buff32_length = buff32->ByteLength();
  char* buff32_data = static_cast<char*>(buff32c.Data()) + buff32_offset;

  //fprintf(stdout, "CopyDtoH: buff32_length = %d\n", buff32_length);
  //fprintf(stdout, "CopyDtoH: buff32_data = %d\n", buff32_data);

  char *phost = buff32_data;

  size_t bytes = args[1]->IntegerValue();
  //size_t bytes = Buffer::Length(buf);
  
  bool async = args.Length() >= 3 && args[2]->IsTrue();

  CUresult error;
  if (async) {
    error = cuMemcpyDtoHAsync(phost, pmem->m_devicePtr, bytes, 0);
  } else {
    error = cuMemcpyDtoH(phost, pmem->m_devicePtr, bytes);
  }

  args.GetReturnValue().Set(Number::New(isolate, error));
}

void Mem::GetDevicePtr(Local<String> property, const PropertyCallbackInfo<Value>& info) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Mem *pmem = ObjectWrap::Unwrap<Mem>(info.Holder());
    //Buffer *ptrbuf = Buffer::New(isolate,sizeof(pmem->m_devicePtr));
    //=============#PATCH===========================
    //Handle<Object> ptrbuf = Buffer::New(isolate,sizeof(pmem->m_devicePtr));
    Handle<Object> ptrbuf;
    Buffer::New(isolate, sizeof(pmem->m_devicePtr)).ToLocal(&ptrbuf);
   
	// memcpy(Buffer::Data(ptrbuf->handle_), &pmem->m_devicePtr, sizeof(pmem->m_devicePtr));
	memcpy(Buffer::Data(ptrbuf), &pmem->m_devicePtr, sizeof(pmem->m_devicePtr));
 
  info.GetReturnValue().Set( ptrbuf );
 // info.GetReturnValue().Set(ptrbuf->handle_);
}
