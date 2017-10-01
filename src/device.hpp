#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <cuda.h>
#include "bindings.hpp"

namespace NodeCuda {

  class Device : public ObjectWrap {
    public:
      static void Initialize(v8::Handle<v8::Object> target);

    protected:
		static v8::Persistent<v8::Function> constructor;

      static void New(const v8::FunctionCallbackInfo<v8::Value>& args);
	  static void GetComputeCapability(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value> &info);
	  static void GetName(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value> &info);
	  static void GetTotalMem(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value> &info);

      // TODO: cuDeviceGetAttribute
      // TODO: cuDeviceGetProperties

      Device() : ObjectWrap(), m_device(0) {}

      ~Device() {}

    private:
      CUdevice m_device;

      friend class Ctx;
  };

}

#endif
