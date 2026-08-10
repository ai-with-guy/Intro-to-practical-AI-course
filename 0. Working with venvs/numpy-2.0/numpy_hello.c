#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#if NPY_ABI_VERSION < 0x02000000
#error "Wrong NumPy ABI: build this project with NumPy 2.x"
#endif

static PyObject *hello(PyObject *self, PyObject *args) {
    PySys_WriteStdout("Good job! You built and imported the NumPy 2.x extension.\n");
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"hello", hello, METH_NOARGS, "Print a success message."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "numpy_hello",
    "A tiny extension built against the NumPy 2.x C ABI.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_numpy_hello(void) {
    import_array();
    return PyModule_Create(&module);
}
