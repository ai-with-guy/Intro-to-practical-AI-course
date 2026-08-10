from pathlib import Path

import numpy
from setuptools import Extension, setup


HERE = Path(__file__).parent

setup(
    name="numpy-20-hello",
    version="0.1.0",
    py_modules=[],
    ext_modules=[
        Extension(
            "numpy_20_hello",
            sources=[str(HERE / "numpy_20_hello.c")],
            include_dirs=[numpy.get_include()],
        )
    ],
    script_args=["build_ext", "--inplace"],
)
