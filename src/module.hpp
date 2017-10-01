#ifndef MODULE_HPP
#define MODULE_HPP

#include <cuda.h>
//#include <nvrtc.h>

#include "bindings.hpp"

namespace NodeCuda {

  class Module : public ObjectWrap {
    public:
      static void Initialize(v8::Handle<v8::Object> target);
      static void GetFunction(const v8::FunctionCallbackInfo<v8::Value>& args);

    protected:
		static v8::Persistent<v8::Function> constructor;

      static void Load(const v8::FunctionCallbackInfo<v8::Value>& args);
	  static void RuntimeCompile(const v8::FunctionCallbackInfo<v8::Value>& args);

      Module() : ObjectWrap(), m_module(0) {}

      ~Module() {}

    private:
      static void New(const v8::FunctionCallbackInfo<v8::Value>& args);

      CUmodule m_module;
  };

}

#endif
