#include "device.hpp"

using namespace v8;
using namespace NodeCuda;

v8::Persistent<v8::Function> Device::constructor;

void Device::Initialize(Handle<Object> target) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	Local<FunctionTemplate> t = FunctionTemplate::New(isolate, Device::New);
  t->InstanceTemplate()->SetInternalFieldCount(1);
  t->SetClassName(String::NewFromUtf8(isolate,"CudaDevice"));

  t->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate,"name"), Device::GetName);
  t->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate,"totalMem"), Device::GetTotalMem);
  t->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate,"computeCapability"), Device::GetComputeCapability);

	// https://nodejs.org/api/addons.html#addons_wrapping_c_objects
  //NODE_SET_PROTOTYPE_METHOD(t, "Device", PlusOne);
  
  target->Set(String::NewFromUtf8(isolate,"Device"), t->GetFunction());

  constructor.Reset(isolate, t->GetFunction());
}

static Handle<Value> GetName_(CUdevice device) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	char deviceName[256];

  cuDeviceGetName(deviceName, 256, device);
  Local<String> result = String::NewFromUtf8(isolate,deviceName);
  return result;
}

void Device::New(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	if (args.IsConstructCall()) {
		// Invoked as constructor: `new MyObject(...)`
		Device *pdevice = new Device();
		cuDeviceGet(&(pdevice->m_device), args[0]->IntegerValue());
		pdevice->Wrap(args.This());
		args.GetReturnValue().Set(args.This());

	}
	else {
		// Invoked as plain function `MyObject(...)`, turn into construct call.
		const int argc = 1;
		Local<Value> argv[argc] = { args[0] };
		Local<Function> cons = Local<Function>::New(isolate, constructor);
		args.GetReturnValue().Set(cons->NewInstance(argc, argv));
	}

}

void Device::GetComputeCapability(Local<String> property, const PropertyCallbackInfo<Value> &info) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

  Device *pdevice = ObjectWrap::Unwrap<Device>(info.Holder());
  int major = 0, minor = 0;
  cuDeviceComputeCapability(&major, &minor, pdevice->m_device);

  Local<Object> result = Object::New(isolate);
  result->Set(String::NewFromUtf8(isolate,"major"), Integer::New(isolate,major));
  result->Set(String::NewFromUtf8(isolate,"minor"), Integer::New(isolate,minor));
  info.GetReturnValue().Set(result);
}

void Device::GetName(Local<String> property, const PropertyCallbackInfo<Value> &info) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

  Device *pdevice = ObjectWrap::Unwrap<Device>(info.Holder());
  info.GetReturnValue().Set( GetName_(pdevice->m_device) );

}

void Device::GetTotalMem(Local<String> property, const PropertyCallbackInfo<Value> &info) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

  Device *pdevice = ObjectWrap::Unwrap<Device>(info.Holder());
  size_t totalGlobalMem;
  cuDeviceTotalMem(&totalGlobalMem, pdevice->m_device);

  info.GetReturnValue().Set(Number::New(isolate,totalGlobalMem));
}
