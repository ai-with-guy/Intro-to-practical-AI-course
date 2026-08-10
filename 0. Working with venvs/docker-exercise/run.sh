#!/bin/sh
set -eu

.venv/bin/python -c "import numpy_hello; numpy_hello.hello()"
echo "Good job! The Docker environment can run the native extension."
