#!/bin/sh
set -eu

cd "$(dirname "$0")"

uv venv --no-project --python 3.12 --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python build.py
.venv/bin/python -c "import numpy_hello; numpy_hello.hello()"

echo "Good job! You built and ran the NumPy 2.x extension locally."
