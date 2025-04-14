from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_maxpool_cuda',  # Name of the package install
    ext_modules=[
        CUDAExtension(
            # This name below MUST match TORCH_EXTENSION_NAME in the cpp file
            # and the name used in the Python import statement
            name='custom_maxpool_cuda',
            sources=[
                'kernel_binding.cpp', # Your C++ binding code
                '2d_pooling.cu',   # Your CUDA kernel code
            ],
            # Optional extra compile args if needed (e.g., for C++ standard, optimization)
            # extra_compile_args={'cxx': ['-std=c++17'],
            #                     'nvcc': ['-O3']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension # Command to build the C++/CUDA extension
    }
)