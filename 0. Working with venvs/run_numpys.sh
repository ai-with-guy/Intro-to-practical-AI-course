#!/bin/sh

uv venv .venv-numpy-1.26
source .venv-numpy-1.26/bin/activate
uv pip install "numpy==1.26.4" "setuptools==75.3.0"
python build_numpy_126.py
python -c "import numpy_126_hello; numpy_126_hello.hello()"
deactivate

uv venv .venv-numpy-2.0
source.venv-numpy-2.0/bin/activate
uv pip install "numpy==2.0.2" "setuptools==75.3.0"
python build_numpy_20.py
python -c "import numpy_20_hello; numpy_20_hello.hello()"
deactivate