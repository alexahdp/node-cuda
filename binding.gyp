{
  "targets": [
    {
      "target_name": "cuda",
      "sources": [
        "src/bindings.cpp",
        "src/ctx.cpp",
        "src/device.cpp",
        "src/function.cpp",
        "src/mem.cpp",
        "src/module.cpp",
        "src/thrust_func.cu"
        ],
        
      'rules': [{
        'extension': 'cu',
        'inputs': ['<(RULE_INPUT_PATH)'],
        'outputs':[ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).<(obj)'],
        'conditions': [
          [
            'OS=="win"',
            {'rule_name': 'cuda on windows',
             'message': "compile cuda file on windows",
             'process_outputs_as_sources': 0,
             'action': ['nvcc -c <(_inputs) -o  <(_outputs)'],
            },
            {
              'rule_name': 'cuda on linux',
              'message': "compile cuda file on linux",
              'process_outputs_as_sources': 1,
              'action': [
                'nvcc',
                '-Xcompiler',
                '-fpic',
                '-c',
                '--machine', '32',
                '-lcuda',
                '-lcudart',
                '<@(_inputs)',
                '-o','<@(_outputs)'
              ],
            }
          ]
        ]
      }],
      
      'conditions': [
        [ 'OS=="mac"', {
          'libraries': ['-framework CUDA'],
          'include_dirs': ['/usr/local/include'],
          'library_dirs': ['/usr/local/lib']
        }],
        [ 'OS=="linux"',
          {'variables': {'obj': 'o'}},
          {
          'libraries': ['-lcuda', '-lcudart', '-lfreeimage','-lGLEW','-lGL'],
          'include_dirs': [
            "./inc",
            "./src",
            "./deps/include",
            "/usr/local/cuda-7.0/include/",
            "/usr/include/GL",
            "/usr/local/cuda/include",
            #"/usr/lib/i386-linux-gnu",
            #"/usr/lib32/nvidia-352",
            #"/usr/local/cuda-7.0/samples/common/inc",
          ],
          'library_dirs': [
            "./inc",
            "/usr/local/cuda-7.0/lib",
            "/usr/lib/i386-linux-gnu",
            "/usr/lib32/nvidia-352",
            #"./src",
            #"/usr/local/nvidia/lib",
            #"/usr/local/cuda/include",
            #"/usr/include/linux",
            #"/usr/local/cuda-7.0/samples/common/inc"
          ]
        }],
        [ 'OS=="win"', {
          'conditions': [
            ['target_arch=="x64"',
              {
                'variables': { 'arch': 'x64' }
              }, {
                'variables': { 'arch': 'Win32' }
              }
            ],
          ],
          'variables': {
            'cuda_root%': '$(CUDA_PATH)'
          },
          'libraries': [
            '-l<(cuda_root)/lib/<(arch)/cuda.lib',
			'-l<(cuda_root)/lib/<(arch)/cudart.lib',
			'../lib/32/freeglut.lib',
			'../lib/32/glew32.lib',
          ],
          "include_dirs": [
            "<(cuda_root)/include",
			
            "inc",
          ],
        }, {
          "include_dirs": [
            "/usr/local/cuda/include"
          ],
        }]
      ]
    }
  ]
}
