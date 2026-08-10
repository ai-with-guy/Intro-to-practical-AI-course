#!/bin/sh
set -eu

.venv/bin/python -c "import numpy_20_hello; numpy_20_hello.hello()"
echo "Good job! The Docker environment can run the native extension."
