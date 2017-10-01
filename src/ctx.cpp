#include "ctx.hpp"
#include "device.hpp"

using namespace NodeCuda;
using namespace v8;

v8::Persistent<v8::Function> Ctx::constructor;

void Ctx::Initialize(Handle<Object> target) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Local<FunctionTemplate> t = FunctionTemplate::New(isolate, Ctx::New);
    t->InstanceTemplate()->SetInternalFieldCount(1);
    t->SetClassName(String::NewFromUtf8(isolate, "CudaCtx"));

    NODE_SET_PROTOTYPE_METHOD(t, "destroy", Ctx::Destroy);
  NODE_SET_PROTOTYPE_METHOD(t, "pushCurrent", Ctx::PushCurrent);
  NODE_SET_PROTOTYPE_METHOD(t, "popCurrent", Ctx::PopCurrent);
  NODE_SET_PROTOTYPE_METHOD(t, "setCurrent", Ctx::SetCurrent);
  NODE_SET_PROTOTYPE_METHOD(t, "getCurrent", Ctx::GetCurrent);
  NODE_SET_PROTOTYPE_METHOD(t, "synchronize", Ctx::Synchronize);
  t->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate,"apiVersion"), Ctx::GetApiVersion);

  target->Set(String::NewFromUtf8(isolate,"Ctx"), t->GetFunction());

  constructor.Reset(isolate, t->GetFunction());
}

void Ctx::New(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

  Ctx *pctx = new Ctx();
  pctx->Wrap(args.This());

  unsigned int flags = args[0]->Uint32Value();
  pctx->m_device = ObjectWrap::Unwrap<Device>(args[1]->ToObject())->m_device;

  cuCtxCreate(&(pctx->m_context), flags, pctx->m_device);

  args.GetReturnValue().Set(args.This());
}

void Ctx::Destroy(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Ctx *pctx = ObjectWrap::Unwrap<Ctx>(args.This());

  CUresult error = cuCtxDestroy(pctx->m_context);
  args.GetReturnValue().Set(Number::New(isolate,error));
}

void Ctx::PushCurrent(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Ctx *pctx = ObjectWrap::Unwrap<Ctx>(args.This());

  CUresult error = cuCtxPushCurrent(pctx->m_context);
  args.GetReturnValue().Set(Number::New(isolate,error));
}

void Ctx::PopCurrent(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Ctx *pctx = ObjectWrap::Unwrap<Ctx>(args.This());

  CUresult error = cuCtxPopCurrent(&(pctx->m_context));
  args.GetReturnValue().Set(Number::New(isolate,error));
}

void Ctx::SetCurrent(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Ctx *pctx = ObjectWrap::Unwrap<Ctx>(args.This());

  CUresult error = cuCtxSetCurrent(pctx->m_context);
  args.GetReturnValue().Set(Number::New(isolate,error));
}

void Ctx::GetCurrent(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Ctx *pctx = ObjectWrap::Unwrap<Ctx>(args.This());

  CUresult error = cuCtxGetCurrent(&(pctx->m_context));
  args.GetReturnValue().Set(Number::New(isolate,error));
}

struct SynchronizeParams {
  Ctx *ctx;
  CUresult error;
 //Persistent<Function> cb;
 Local<Function> cb;
};

void Ctx::Synchronize(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

  if (args.Length() >= 1 && args[0]->IsFunction()) {
    // Asynchronous
    Local<Function> cb = Local<Function>::Cast(args[0]);

    Ctx *ctx = ObjectWrap::Unwrap<Ctx>(args.This());
    if (ctx->sync_in_progress){
        args.GetReturnValue().Set(Number::New(isolate,-1));
        return;
    };

    SynchronizeParams *params = new SynchronizeParams();
    params->ctx = ctx;
    //params->cb = Handle<Function>::New(isolate,cb);
    //params->cb = Local<Function>::New(isolate, cb);
    params->cb = cb; // Local<Function>::New(isolate, cb);
    
    cuCtxPopCurrent(NULL);

    // build up the work request
    uv_work_t* work_req = new uv_work_t;
    work_req->data = params;

    uv_queue_work(uv_default_loop(),
        work_req,
        Process,
        After);
    uv_ref((uv_handle_t*) &work_req);

    ctx->Ref();
    ctx->sync_in_progress = true;

    args.GetReturnValue().Set(Undefined(isolate));

  } else {
    // Synchronous
    CUresult error = cuCtxSynchronize();
    args.GetReturnValue().Set(Number::New(isolate,error));
  }
}

void Ctx::Process(uv_work_t* work_req) {
  SynchronizeParams *params = static_cast<SynchronizeParams*>(work_req->data);

  params->error = cuCtxPushCurrent(params->ctx->m_context);
  if (params->error) return;

  params->error = cuCtxSynchronize();
  if (params->error) return;

  params->error = cuCtxPopCurrent(NULL);
}

void Ctx::After(uv_work_t* work_req, int status) {
  assert(status == 0);
  Isolate* isolate = Isolate::GetCurrent();
  HandleScope scope(isolate);

  SynchronizeParams *params = static_cast<SynchronizeParams*>(work_req->data);

  params->ctx->Unref();
  params->ctx->sync_in_progress = false;

  cuCtxPushCurrent(params->ctx->m_context);

  Local<Value> argv[1];
  argv[0] = Number::New(isolate,params->error);

  TryCatch try_catch;
  
  params->cb->Call(isolate->GetCurrentContext()->Global(), 1, argv);
  
  //params->cb->Call(Context::GetCurrent()->Global(), 1, argv);
  
  if (try_catch.HasCaught()) FatalException(try_catch);

  // 2015-03-08 как должно быть если мы persistent заменили на handle?
  // params->cb->Dispose();

  uv_unref((uv_handle_t*) work_req);
  delete params;
}

void Ctx::GetApiVersion(Local<String> property, const PropertyCallbackInfo<Value> &info) {
    Isolate* isolate = Isolate::GetCurrent();
    HandleScope scope(isolate);

    Ctx *pctx = ObjectWrap::Unwrap<Ctx>(info.Holder());

  unsigned int version;
  CUresult error = cuCtxGetApiVersion(pctx->m_context, &version);

  info.GetReturnValue().Set(Number::New(isolate,version));
}
