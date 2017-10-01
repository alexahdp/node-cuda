#ifndef MEM_HPP
#define MEM_HPP

#include <cuda.h>
#include "bindings.hpp"
#include "function.hpp"
//#include "cuda_gl_interop.h"

// #include <X11/Xlibint.h>
// #include <X11/Xlib.h>
// #include <X11/Xproto.h>
//#include <GL/glew.h>

//#include <GLFW/glfw3.h>
//#define GLFW_EXPOSE_NATIVE_WIN32
//#define GLFW_EXPOSE_NATIVE_WGL
//#include <GLFW/glfw3native.h>


#include <GL/freeglut.h>

// #ifdef  _WIN32
//   #include    <GL/wglew.h>
//   #include <GL/wglext.h>
// #else
//     #include    <GL/glxew.h>
// #endif


#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>


namespace NodeCuda {

  class Mem : public ObjectWrap {
    public:
      static void Initialize(v8::Handle<v8::Object> target);

    protected:
        static v8::Persistent<v8::Function> constructor;

      static void GetDevicePtr(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& info);

      static void Alloc(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void AllocPitch(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void Free(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void CopyHtoD(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void CopyDtoH(const v8::FunctionCallbackInfo<v8::Value>& args);
      
      static void thrust_inclusiveScan(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void thrust_reduce_floatSum(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void thrust_reduce_floatMax(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void thrust_reduce_floatMin(const v8::FunctionCallbackInfo<v8::Value>& args);
      
      static void thrust_floatSort_int(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void thrust_remove_int   (const v8::FunctionCallbackInfo<v8::Value>& args);

      static void glGetArrayBufferBinding(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void AllocVBO(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void   RegVBO(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void UnregVBO(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void getContext(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void makeContextCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
      
      Mem() : ObjectWrap(), m_devicePtr(0) {}

      ~Mem() {}

    private:
      //static void New(const FunctionCallbackInfo<Value>& args);
      static void New(const v8::FunctionCallbackInfo<v8::Value>& args);
      
      CUdeviceptr           m_devicePtr;
      struct cudaGraphicsResource *vbo;
      
      friend class NodeCuda::Function;
  };

}

#endif
