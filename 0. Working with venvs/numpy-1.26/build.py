from pathlib import Path

import numpy
from setuptools import Extension, setup


HERE = Path(__file__).parent

setup(
    name="numpy-hello",
    version="0.1.0",
    ext_modules=[
        Extension(
            "numpy_hello",
            sources=[str(HERE / "numpy_hello.c")],
            include_dirs=[numpy.get_include()],
        )
    ],
    script_args=["build_ext", "--inplace"],
)
