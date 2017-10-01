#include "module.hpp"
#include "function.hpp"

#include <fstream>

using namespace v8;
using namespace NodeCuda;

v8::Persistent<v8::Function> Module::constructor;

void Module::Initialize(Handle<Object> target) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	Local<FunctionTemplate> t = FunctionTemplate::New(isolate, Module::New);

	t = FunctionTemplate::New(isolate, Module::New);
	//Local<FunctionTemplate> t = FunctionTemplate::New(isolate, Module::New);
	//constructor_template = Persistent<FunctionTemplate>::New(isolate, t);
	t->InstanceTemplate()->SetInternalFieldCount(1);
	t->SetClassName(String::NewFromUtf8(isolate, "CudaModule"));

  // Module objects can only be created by load functions
  NODE_SET_METHOD(target, "moduleLoad", Module::Load);
  NODE_SET_METHOD(target, "moduleRuntimeCompile", Module::RuntimeCompile);

  NODE_SET_PROTOTYPE_METHOD(t, "getFunction", Module::GetFunction);

  constructor.Reset(isolate, t->GetFunction());
}

void Module::New(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

  Module *pmem = new Module();
  pmem->Wrap(args.This());

  args.GetReturnValue().Set( args.This() );
}

void Module::Load(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	//Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
	v8::Local<v8::Function> cons = v8::Local<v8::Function>::New(isolate, constructor);
	Local<Object> result = cons->NewInstance();

  Module *pmodule = ObjectWrap::Unwrap<Module>(result);

  String::Utf8Value fname(args[0]);
  CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

  result->Set(String::NewFromUtf8(isolate,"fname"), args[0]);
  result->Set(String::NewFromUtf8(isolate,"error"), Integer::New(isolate,error));

  args.GetReturnValue().Set(result);
}

void Module::RuntimeCompile(const FunctionCallbackInfo<Value>& args) {
	/*
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	//Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
	v8::Local<v8::Function> cons = v8::Local<v8::Function>::New(isolate, constructor);
	Local<Object> result = cons->NewInstance();

	Module *pmodule = ObjectWrap::Unwrap<Module>(result);

	String::Utf8Value fname (args[0]);
	String::Utf8Value saxpy (args[1]);
	
	// Create an instance of nvrtcProgram with the SAXPY code string.
	nvrtcProgram prog;
	nvrtcCreateProgram( &prog, *saxpy, *fname, 0, NULL, NULL );

	// Compile the program for compute_20 with fmad disabled.
	const char *opts[] = { "--gpu-architecture=compute_35", "--use_fast_math" };
	
	nvrtcResult compileResult = nvrtcCompileProgram( prog, 2, opts );

	// Obtain compilation log from the program.
	size_t logSize;
	
	nvrtcGetProgramLogSize(prog, &logSize);
	
	char *log = new char[logSize];
	nvrtcGetProgramLog(prog, log);
	result->Set(String::NewFromUtf8(isolate, "log"), String::NewFromUtf8(isolate,log));
	
	delete[] log;

	if (compileResult == NVRTC_SUCCESS) {
		// Obtain PTX from the program.
		size_t ptxSize;
		nvrtcGetPTXSize(prog, &ptxSize);
		char *ptx = new char[ptxSize];
		nvrtcGetPTX(prog, ptx);
		// Destroy the program.
		nvrtcDestroyProgram(&prog);
		
		if ( args.Length() >= 2 ){
			String::Utf8Value dumpto_file(args[2]);
			
			std::ofstream OutFile;
			OutFile.open( *dumpto_file, std::ios::out | std::ios::binary );
			OutFile.write( ptx, ptxSize );
			OutFile.close();
		};
		
		// Load the generated PTX and get a handle to the SAXPY kernel.
		CUresult error = cuModuleLoadDataEx(&(pmodule->m_module), ptx, 0, 0, 0);
		result->Set(String::NewFromUtf8(isolate, "fname"), args[0]);
		result->Set(String::NewFromUtf8(isolate, "error"), Integer::New(isolate, error));
	};

	args.GetReturnValue().Set(result);
	*/
}

void Module::GetFunction(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = Isolate::GetCurrent();
	HandleScope scope(isolate);

	//Local<Object> result = NodeCuda::Function::constructor_template->InstanceTemplate()->NewInstance();
	v8::Local<v8::Function> cons = v8::Local<v8::Function>::New(isolate, NodeCuda::Function::constructor);
	Local<Object> result = cons->NewInstance();

	Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());
  NodeCuda::Function *pfunction = ObjectWrap::Unwrap<NodeCuda::Function>(result);

  String::Utf8Value name(args[0]);
  CUresult error = cuModuleGetFunction(&(pfunction->m_function), pmodule->m_module, *name);

  result->Set(String::NewFromUtf8(isolate,"name"), args[0]);
  result->Set(String::NewFromUtf8(isolate,"error"), Integer::New(isolate,error));

  args.GetReturnValue().Set(result);
}

