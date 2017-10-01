#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cuda.h>
#include "bindings.hpp"
#include "module.hpp"

namespace NodeCuda {

  class Function : public ObjectWrap {
    public:
      static void Initialize(v8::Handle<v8::Object> target);

    protected:
		static v8::Persistent<v8::Function> constructor;

      static void LaunchKernel(const v8::FunctionCallbackInfo<v8::Value>& args);

      Function() : ObjectWrap(), m_function(0) {}

      ~Function() {}

    private:
      static void New(const v8::FunctionCallbackInfo<v8::Value>& args);

      CUfunction m_function;

	  friend void Module::GetFunction(const v8::FunctionCallbackInfo<v8::Value>&);
  };

}

#endif
