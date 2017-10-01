#ifndef BINDINGS_HPP
#define BINDINGS_HPP

#undef True
#undef False
#undef None

#include <v8.h>
#include <node.h>

#include <cuda.h>
#include <cuda_runtime.h>

// 0.12
#include <node_object_wrap.h>

//using namespace v8;
using namespace node;



namespace NodeCuda {

	static void GetDriverVersion(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value> &info);
	static void GetDeviceCount(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value> &info);

}

#endif
