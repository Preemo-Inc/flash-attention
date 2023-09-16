# pyright: strict

from typing import Any, Dict
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # type: ignore[reportUnknownVariableType]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": [
                CUDAExtension(
                    # Flash Attention v1's rotary_emb kernels
                    "rotary_emb",
                    [
                        "csrc/rotary/rotary.cpp",
                        "csrc/rotary/rotary_cuda.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-g", "-march=native", "-funroll-loops"],
                        "nvcc": ["-O3", "--use_fast_math", "--expt-extended-lambda"],
                    },
                ),
                CUDAExtension(
                    # Flash Attention v1's dropout_layer_norm kernels
                    name="dropout_layer_norm",
                    sources=[
                        "csrc/layer_norm/ln_api.cpp",
                        "csrc/layer_norm/ln_fwd_256.cu",
                        "csrc/layer_norm/ln_bwd_256.cu",
                        "csrc/layer_norm/ln_fwd_512.cu",
                        "csrc/layer_norm/ln_bwd_512.cu",
                        "csrc/layer_norm/ln_fwd_768.cu",
                        "csrc/layer_norm/ln_bwd_768.cu",
                        "csrc/layer_norm/ln_fwd_1024.cu",
                        "csrc/layer_norm/ln_bwd_1024.cu",
                        "csrc/layer_norm/ln_fwd_1280.cu",
                        "csrc/layer_norm/ln_bwd_1280.cu",
                        "csrc/layer_norm/ln_fwd_1536.cu",
                        "csrc/layer_norm/ln_bwd_1536.cu",
                        "csrc/layer_norm/ln_fwd_2048.cu",
                        "csrc/layer_norm/ln_bwd_2048.cu",
                        "csrc/layer_norm/ln_fwd_2560.cu",
                        "csrc/layer_norm/ln_bwd_2560.cu",
                        "csrc/layer_norm/ln_fwd_3072.cu",
                        "csrc/layer_norm/ln_bwd_3072.cu",
                        "csrc/layer_norm/ln_fwd_4096.cu",
                        "csrc/layer_norm/ln_bwd_4096.cu",
                        "csrc/layer_norm/ln_fwd_5120.cu",
                        "csrc/layer_norm/ln_bwd_5120.cu",
                        "csrc/layer_norm/ln_fwd_6144.cu",
                        "csrc/layer_norm/ln_bwd_6144.cu",
                        "csrc/layer_norm/ln_fwd_7168.cu",
                        "csrc/layer_norm/ln_bwd_7168.cu",
                        "csrc/layer_norm/ln_fwd_8192.cu",
                        "csrc/layer_norm/ln_bwd_8192.cu",
                        "csrc/layer_norm/ln_parallel_fwd_256.cu",
                        "csrc/layer_norm/ln_parallel_bwd_256.cu",
                        "csrc/layer_norm/ln_parallel_fwd_512.cu",
                        "csrc/layer_norm/ln_parallel_bwd_512.cu",
                        "csrc/layer_norm/ln_parallel_fwd_768.cu",
                        "csrc/layer_norm/ln_parallel_bwd_768.cu",
                        "csrc/layer_norm/ln_parallel_fwd_1024.cu",
                        "csrc/layer_norm/ln_parallel_bwd_1024.cu",
                        "csrc/layer_norm/ln_parallel_fwd_1280.cu",
                        "csrc/layer_norm/ln_parallel_bwd_1280.cu",
                        "csrc/layer_norm/ln_parallel_fwd_1536.cu",
                        "csrc/layer_norm/ln_parallel_bwd_1536.cu",
                        "csrc/layer_norm/ln_parallel_fwd_2048.cu",
                        "csrc/layer_norm/ln_parallel_bwd_2048.cu",
                        "csrc/layer_norm/ln_parallel_fwd_2560.cu",
                        "csrc/layer_norm/ln_parallel_bwd_2560.cu",
                        "csrc/layer_norm/ln_parallel_fwd_3072.cu",
                        "csrc/layer_norm/ln_parallel_bwd_3072.cu",
                        "csrc/layer_norm/ln_parallel_fwd_4096.cu",
                        "csrc/layer_norm/ln_parallel_bwd_4096.cu",
                        "csrc/layer_norm/ln_parallel_fwd_5120.cu",
                        "csrc/layer_norm/ln_parallel_bwd_5120.cu",
                        "csrc/layer_norm/ln_parallel_fwd_6144.cu",
                        "csrc/layer_norm/ln_parallel_bwd_6144.cu",
                        "csrc/layer_norm/ln_parallel_fwd_7168.cu",
                        "csrc/layer_norm/ln_parallel_bwd_7168.cu",
                        "csrc/layer_norm/ln_parallel_fwd_8192.cu",
                        "csrc/layer_norm/ln_parallel_bwd_8192.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-O3"],
                        "nvcc": [
                            "-O3",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                        ],
                    },
                    include_dirs=["csrc/layer_norm"],
                ),
            ],
            "cmdclass": {"build_ext": BuildExtension},
        }
    )