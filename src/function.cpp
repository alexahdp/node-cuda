#include <node_buffer.h>
#include <cstring>
#include <cstdio>
#include "function.hpp"
#include "mem.hpp"

using namespace v8;
using namespace NodeCuda;

v8::Persistent<v8::Function> NodeCuda::Function::constructor;

void NodeCuda::Function::Initialize(Handle<Object> target) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	Local<FunctionTemplate> t = FunctionTemplate::New(isolate, NodeCuda::Function::New);

	t = FunctionTemplate::New(isolate, NodeCuda::Function::New);

	t->InstanceTemplate()->SetInternalFieldCount(1);
  t->SetClassName(String::NewFromUtf8(isolate,"CudaFunction"));

  NODE_SET_PROTOTYPE_METHOD(t, "launchKernel", NodeCuda::Function::LaunchKernel);

  // Function objects can only be created by cuModuleGetFunction
  constructor.Reset(isolate, t->GetFunction());
}

void NodeCuda::Function::New(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);
	
  NodeCuda::Function *pfunction = new NodeCuda::Function();
  pfunction->Wrap(args.This());

  args.GetReturnValue().Set( args.This() );
}

void NodeCuda::Function::LaunchKernel(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	Function *pfunction = ObjectWrap::Unwrap<Function>(args.This());

  Local<Array> gridDim = Local<Array>::Cast(args[0]);
  unsigned int gridDimX = gridDim->Get(0)->Uint32Value();
  unsigned int gridDimY = gridDim->Get(1)->Uint32Value();
  unsigned int gridDimZ = gridDim->Get(2)->Uint32Value();

  Local<Array> blockDim = Local<Array>::Cast(args[1]);
  unsigned int blockDimX = blockDim->Get(0)->Uint32Value();
  unsigned int blockDimY = blockDim->Get(1)->Uint32Value();
  unsigned int blockDimZ = blockDim->Get(2)->Uint32Value();

  Local<Object> buf = args[2]->ToObject();
  char *pbuffer = Buffer::Data(buf);
  size_t bufferSize = Buffer::Length(buf);

  void *cuExtra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, pbuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &bufferSize,
    CU_LAUNCH_PARAM_END
  };

  CUresult error = cuLaunchKernel(pfunction->m_function,
      gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ,
      0, 0, NULL, cuExtra);

  args.GetReturnValue().Set(Number::New(isolate,error));
}

