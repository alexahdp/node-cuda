#ifndef CTX_HPP
#define CTX_HPP



//#include <stdlib.h>
//#define GL_GLEXT_PROTOTYPES
//#include <windows.h>


//#include <GL/freeglut.h>


#ifdef  _WIN32
    #include    <GL/wglew.h>
#else
    #include    "../inc/GL/glxew.h"
    //#include    <GL/glxew.h>
#endif
//#include <GL/glxew.h>

#include <GL/gl.h>
//#include <GL/wglext.h>

//#include <GL/glut.h>
//#include <GL/glext.h>

#include <cuda.h>

//#define WGL_NV_gpu_affinity
#include <cudaGL.h>
#include <cuda_gl_interop.h>
//#undef WGL_NV_gpu_affinity
#include <uv.h>


#include "bindings.hpp"
#include "device.hpp"

namespace NodeCuda {

  class Ctx : public ObjectWrap {
    public:
      static void Initialize(v8::Handle<v8::Object> target);

    protected:
        static v8::Persistent<v8::Function> constructor;

      static void New(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void Destroy(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void PushCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void PopCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void SetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void GetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void Synchronize(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void GetApiVersion(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value> &info);

      Ctx() : ObjectWrap(), m_context(NULL), m_device(0), sync_in_progress(false) {}

      ~Ctx () {}

    private:
      CUcontext m_context;
      CUdevice m_device;

      bool sync_in_progress;

      uv_work_t work_req_;

      static void Process(uv_work_t* work_req);
      static void After(uv_work_t* work_req, int status);
  };

}

#endif
