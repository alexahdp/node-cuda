//module.exports = require('./cuda.node');

module.exports = require('./build/Release/cuda.node');

// от Вити
// module.exports = util.inherits( function(){ } , require('./build/Release/cuda.node') );

var _t       = use('t');
var md5      = require('./md5');
var fs       = require('fs');

// пока так
module.exports.ctx = new module.exports.Ctx(0, module.exports.Device(0));


var CCO = module.exports.Device(0).computeCapability;
var CC  = CCO.major + '' + CCO.minor;
if ( CC > 35  ) CC = 35;
if ( CC == 21 ) CC = 20;


var cachedir =  __dirname + '/cache';
if (! fs.existsSync(cachedir) ) fs.mkdirSync( cachedir );

var CachePath = cachedir + '/cache_'+CC+'/';
if (! fs.existsSync(CachePath) ) fs.mkdirSync( CachePath );

var TempPath = cachedir + '/cache_src_'+CC+'/';
if (! fs.existsSync(TempPath) ) fs.mkdirSync( TempPath );

var CachePath2 = cachedir + '/cp_cache_'+CC+'/';
if (! fs.existsSync(CachePath2) ) fs.mkdirSync( CachePath2 );

var TempPath2 = cachedir + '/cp_cache_src_'+CC+'/';
if (! fs.existsSync(TempPath2) ) fs.mkdirSync( TempPath2 );

//var cutil_math = fs.readFileSync(__dirname + '/h/cutil_math.h', {encoding: 'utf8'});


var inc_cutil_math = `#include <${__dirname}/h/my_cutil_math.h>\n`;

// Type names follow the W3C typed array specs, not NodeJS's Buffer library

var typeByteSize = {
  "Uint8": 1,
  "Uint16": 2,
  "Uint32": 4,
  "Int8": 1,
  "Int16": 2,
  "Int32": 4,
  "Float32": 4,
  "Float64": 8,
  "DevicePtr": 4
};

var typeAlignment = {
  "Uint8": 1,
  "Uint16": 2,
  "Uint32": 4,
  "Int8": 1,
  "Int16": 2,
  "Int32": 4,
  "Float32": 4,
  "Float64": 8,
  "DevicePtr": 4
};

var typeBufferFunc = {
  "Uint8": "UInt8",
  "Uint16": "UInt16LE",
  "Uint32": "UInt32LE",
  "Int8": "Int8",
  "Int16": "Int16LE",
  "Int32": "Int32LE",
  "Float32": "FloatLE",
  "Float64": "DoubleLE"
}

var elKernelTypes = {
	"uint"  : ["Int32"],
	"int"   : ["Int32"],
	"int2"  : ["Int32","Int32"],
	"int3"  : ["Int32","Int32","Int32"],
	"int4"  : ["Int32","Int32","Int32","Int32"],
	"float" : ["Float32"],
	"float2": ["Float32","Float32"],
	"float3": ["Float32","Float32","Float32"],
	"float4": ["Float32","Float32","Float32","Float32"],
};

function alignUp(offset, alignment) {
  return (((offset) + (alignment) - 1) & ~((alignment) - 1));
}

module.exports.prepareArguments = function (args) {
  var paramBufferSize = 0;

  for (var i in args) {
	//console.info( i, args[i].type, args[i].value );
	
	var type_ = Array.isArray( args[i].type ) ? args[i].type : [ args[i].type ];
	
	paramBufferSize = alignUp(paramBufferSize, typeAlignment[type_[0]] * type_.length );
	type_.forEach(type => {
		// paramBufferSize = alignUp(paramBufferSize, typeAlignment[type]);
		
		//console.info(type);
		
		if (typeof(typeByteSize[type]) != "number")
		  throw "Invalid type given";
		
		paramBufferSize += typeByteSize[type];
	});
  }

  var paramBuffer = new Buffer(paramBufferSize);

  var offset = 0;
  for (var i in args) {
    var type_ = Array.isArray( args[i].type ) ? args[i].type : [ args[i].type ];
	
	offset = alignUp(offset, typeAlignment[type_[0]] * type_.length );
	
    type_.forEach( (type,j) => {
		var v = Array.isArray(args[i].value) ? args[i].value[j] : args[i].value;
		
		
		if (type == "DevicePtr") {
		  args[i].value.copy(paramBuffer, offset);
		} else {
		  paramBuffer["write" + typeBufferFunc[type]](v, offset);
		}
		
		offset += typeByteSize[type];
	});
  }
  
  return paramBuffer;
}

module.exports.launch = function () {
  var func = arguments[0];
  var gridDim = arguments[1];
  var blockDim = arguments[2];
  var args = arguments[3];

  args = module.exports.prepareArguments(args);

  return func.launchKernel(gridDim, blockDim, args);
}


var iKernels = module.exports.iKernels = { };

var WARN = { };

module.exports.iKernel = function( id_, n, params, code, prefix, callpoint ){
	/*
	if (!WARN[id]){
		WARN[id] = 1;
		console.info(id,n,params,code,prefix);
	};
	*/
	
	// должны ли callpoint и id быть разными вещами? (продумать обратную совместимость)
	
	var funcName = id_ || 'callpoint';
	var id       = id_ ||  callpoint ;
	
	if ( !iKernels[id] ){
		if (!prefix) prefix = '';
		
		var paramNames = Object.keys( params ).sort();
		var code       = prefix + `
			extern "C" {
				__global__ void gpuFunc_` + funcName + '( ' + paramNames.join(', ') + ') {' + `
					int i = blockIdx.x * blockDim.x + threadIdx.x;
					if (i < n) {
						` + code + `;
					};
				}
			}
		` + '\n';
		
		if ( code.match(/__iKernelStructs__/) ){
			var iKernelStructs = paramNames.join(',');
			var iKernelVars    = paramNames.map ( it => it.replace(/^[^ ]+ \*?/,'') ).join(',');
			
			code = code.replace( /__iKernelStructs__/g,iKernelStructs);
			code = code.replace( /__iKernelVars__/g   ,iKernelVars   );
		};
		
		var cuModule;
		
		function compile( src,out ){
			_t.bench( function(){
				console.info( require('child_process').execSync(
					[
						'nvcc --ptx',
						// '--std c++11',
						'--gpu-architecture compute_' + CC,
						'--gpu-code sm_' + CC,
						// '--optimize 2',
						'-maxrregcount=0  --machine 32 --compile  --use_fast_math -Xptxas -v,-abi=no  -use_fast_math ',
						'-o ' + src,
						out
					].join(' '), {encoding: 'utf8'}) );
			}, "GPU[" + id + "] compile time: ");
		};
		
		
		if ( callpoint ){
			// новая система кеширования
			
			var cacheFile  = md5( code ) + '.ptx';
			var sourceFile = md5( code ) + '.cu';
			
			// добавить md5 эвристику на копирование папки
			
			if ( fs.existsSync( CachePath2 + cacheFile ) ){
				cuModule = module.exports.moduleLoad( CachePath2 + cacheFile );
			}else{
				fs.writeFileSync(TempPath2 + sourceFile, code, {encoding: 'utf8'});
				
				compile(
					CachePath2 + cacheFile,
					TempPath2  + sourceFile
				);
				
				// в принципе никогда не удаляем кеш-код cuda.
				// это позволяет по md5 мгновенно получить любую эквивалентную версию кода
				// также путь к файлу НЕ содержит callpoint идентификатора, он используется только для runtime кеша
				
				cuModule = module.exports.moduleLoad( CachePath2 + cacheFile );
			};
		}else{
			// старая система кеширования
			
			var idpath = id + '-' + md5( global.VOID_MAIN_FILEPATH ).substring(0,8);
			
			var cacheFile  = idpath + '.' + md5( code ) + '.ptx';
			var sourceFile = idpath + '.' + md5( code ) + '.cu';
			
			// добавить md5 эвристику на копирование папки
			
			if ( fs.existsSync( CachePath + cacheFile ) ){
				cuModule = module.exports.moduleLoad( CachePath + cacheFile );
			}else{
				fs.writeFileSync(TempPath + sourceFile, code, {encoding: 'utf8'});
				
				compile(
					CachePath + cacheFile,
					TempPath  + sourceFile
				);
				
				//cuModule = module.exports.moduleRuntimeCompile( id + ".cu", code, CachePath + cacheFile );
				
				fs.readdirSync( CachePath ).forEach( function( fileName ){
					if ( fileName.split('.')[0] == idpath && fileName != cacheFile ){
						fs.unlinkSync( CachePath + fileName );
					};
				} );
				
				cuModule = module.exports.moduleLoad( CachePath + cacheFile );
			};
		};
		
		if (cuModule.log){
			console.info( code );
			console.error("GPU[" + id + "]: ", cuModule.log);
			throw Error();
		};
		
		iKernels[id] = {
			prog      : cuModule.getFunction( 'gpuFunc_' + funcName ),
			paramNames: paramNames,
		};
	};
	
	if (n > 0){
		var p = 256;
		var q = 1;
		
		var error = module.exports.launch(
			iKernels[id].prog,
			[Math.floor((n + (p-1)) / p), 1, 1],
			[p, q, 1],
			iKernels[id].paramNames.map(function(k){
				var v = params[k];
				return { type : v[0], value: v[1] };
			})
		);
		
		if (error){
			console.error("Launched kernel:", error);
			throw Error();
		};
	};
};

var elKernels = module.exports.elKernels = { };

var INIT = 0;

module.exports.elKernel = function( first, opt ){
	var cu = this;
	
	if (!opt){
		opt = first;
	}else{
		opt.count = first;
	};
	
	
	//var caller = callerId.getData();
	//var file = md5.digest_s( caller.filePath );
	//var line = caller.line;
	//var id = opt.id ? opt.id + '_' + file : file + '_' + line;
	
	var id = opt.id + opt.callpoint;
	
	// elements прекомпилируется хардкодно
	if ( !elKernels[id] ){
		var elk = elKernels[id] = {};
		
		elk.prefix  = opt.prefix  || '';
		elk.postfix = opt.postfix || '';
		
		elk.kernel = opt.kernel;
		elk.params = { };
		
		elk.struct = Object.keys( opt.elements ).sort().map(function( elName ){
			// 2016-03-27 - фича передачи в кернел только упомянутых arrs'ов
			var el_      = opt.elements[ elName ];
			var el       = Array.isArray( el_ ) ? el_[0] : el_;
			var usedArrs = Array.isArray( el_ ) ? el_[1] : Object.keys( el.arrs );
			
			elk.params[ 'int ' + elName + 'count' ] = [ 'Int32', el.count ];
			
			return usedArrs.map( function( arrName ){
				var arr   = el.arrs[ arrName ];
				
				if ( arr.struct.split(' ').length > 1 ){
					var tName  = 'T' + elName + arrName;
					var vName  =       elName + arrName;
					var struct = 'struct ' + tName + ' { ' + arr.struct + '; };\n';
				}else{
					var tName  = arr.struct;
					var vName  =       elName + arrName;
					var struct = '\n';
				};
				
				// 2015-12-15 на девятый месяц зоркий глаз заметил, что кернел нельзя вызвать,
				// передав ему другой экземпляр такого же element'а (на примере sst -> morphPlot() )
				
				elk.params[ tName + ' *' + vName ] = [ 'DevicePtr', arr._devicePtr() ];
				//elk.params[ tName + ' *' + vName ] = [ 'DevicePtr', arr.devicePtr ];
				
				return struct;
			} ).join('');
		}).join('');
	};
	
	var elk = elKernels[id];
	
	
	// пока самый тупой способ уйти от закешированности!
	// 2015-12-15 на девятый месяц зоркий глаз заметил, что кернел нельзя вызвать,
	// передав ему другой экземпляр такого же element'а (на примере sst -> morphPlot() )
	
	Object.keys( opt.elements ).forEach(function( elName ){
		// 2016-03-27 - фича передачи в кернел только упомянутых arrs'ов
		var el_      = opt.elements[ elName ];
		var el       = Array.isArray( el_ ) ? el_[0] : el_;
		var usedArrs = Array.isArray( el_ ) ? el_[1] : Object.keys( el.arrs );
		
		// писец, и это тоже было закешировано! и там же баг всплыл!
		
		// 2016-03-27 по идее, нам в scopeMode не нужны эти readOnly count'ы
		if (!opt.scopeMode) {
			elk.params[ 'int ' + elName + 'count' ] = [ 'Int32', el.count ];
		};
		
		return usedArrs.forEach( function( arrName ){
			var arr   = el.arrs[ arrName ];
			
			if ( arr.struct.split(' ').length > 1 ){
				var tName  = 'T' + elName + arrName;
				var vName  =       elName + arrName;
				//var struct = 'struct ' + tName + ' { ' + arr.struct + '; };\n';
			}else{
				var tName  = arr.struct;
				var vName  =       elName + arrName;
				//var struct = '\n';
			};
			
			// 2015-12-15 на девятый месяц зоркий глаз заметил, что кернел нельзя вызвать,
			// передав ему другой экземпляр такого же element'а (на примере sst -> morphPlot() )
			
			elk.params[ tName + ' *' + vName ] = [ 'DevicePtr', arr._devicePtr() ];
			//elk.params[ tName + ' *' + vName ] = [ 'DevicePtr', arr.devicePtr ];
		} );
	});
	//
	
	
	
	if (opt.args){
		var args = opt.args;
		
		// можно оптимизировать запуск, подготавливая структуры заранее
		Object.keys( args ).sort().forEach( function( argK ){
			var argV = opt.args[argK];
			
			// при scopeMode - Array в качестве значения параметра используются дл float234 значений, а не для передачи указателей вручную
			if ( !opt.scopeMode && Array.isArray(argV) ){
				elk.params[argK] = argV;
			}else{
				var type = elKernelTypes[ (argK.split(' '))[0] ];
				elk.params[argK] = [type, argV];
			};
		});
	};
	
	// каждый раз разный
	elk.params['int n'] = [ "Int32", opt.count ];
	
	return cu.iKernel( opt.id, opt.count, elk.params, elk.kernel, inc_cutil_math + elk.prefix + "\n" + elk.struct + "\n" + elk.postfix + "\n", opt.callpoint );
};


GLOBAL.Cu = function( count, args, kernel ){
	var where = (new Error()).stack.split('\n')[2];
	var run   = {
		// 2015-12-12 пока оставил тупой md5 от точки вызова
		// to do: сделать двухуровневый кеш
		// один   кеширует текущий стек вызова (максимально полно)
		// другой на каждый старт нового стека ищет по md5 уже скомпиленный код в глобальной кеш-директории?
		// либо создавать директории в кеш на каждый исполняемый .void-файл?
		callpoint: md5( where ),
		count    :  count,
		args     : { },
		elements : { },
		kernel   : kernel,
	};
	
	Object.keys( args ).forEach( key => {
		var type = ( key.indexOf(' ') != -1 ) ? 'args': 'elements';
		run[type][key] = args[key];
	} );
	
	if ( count == 0 && count !== 0 ){
		run.count = elements[count].count;
	};
	
	return module.exports.elKernel( run );
};


