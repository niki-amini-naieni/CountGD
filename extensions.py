# extensions.py
import os
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
import glob


def get_extensions():
    # extensions_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.dirname(os.path.abspath("models/GroundingDINO/ops/src"))
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available():
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError("Cuda is not available")

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules
