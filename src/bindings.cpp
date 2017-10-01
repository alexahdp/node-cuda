#include "bindings.hpp"
#include "ctx.hpp"
#include "device.hpp"
#include "function.hpp"
#include "mem.hpp"
#include "module.hpp"

using namespace NodeCuda;
using namespace v8;

void init (Handle<Object> target) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

  // Initiailze the cuda driver api
  cuInit(0);

  // These methods don't need instances
  target->SetAccessor(String::NewFromUtf8(isolate,"driverVersion"), GetDriverVersion);
  target->SetAccessor(String::NewFromUtf8(isolate,"deviceCount"), GetDeviceCount);

  // Initialize driver api bindings
  Ctx::Initialize(target);
  Device::Initialize(target);
  NodeCuda::Function::Initialize(target);
  Mem::Initialize(target);
  Module::Initialize(target);
}

void NodeCuda::GetDriverVersion(Local<String> property, const PropertyCallbackInfo<Value> &info) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	int driverVersion = 0;
  cuDriverGetVersion(&driverVersion);
  info.GetReturnValue().Set(Integer::New(isolate,driverVersion));
}

void NodeCuda::GetDeviceCount(Local<String> property, const PropertyCallbackInfo<Value> &info) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	int count = 0;
  cuDeviceGetCount(&count);
  info.GetReturnValue().Set(Integer::New(isolate,count));
}

NODE_MODULE(cuda, init);
