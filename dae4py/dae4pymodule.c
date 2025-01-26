#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "numpy/arrayobject.h"

#include "dassl/dassl.h"
#include "pside/pside.h"
#include "radau/radau.h"

static PyMethodDef methods[] = {
    {"dassl", (PyCFunction)dassl, METH_VARARGS | METH_KEYWORDS, dassl_doc},
    {"pside", (PyCFunction)pside, METH_VARARGS | METH_KEYWORDS, pside_doc},
    {"radau", (PyCFunction)radau, METH_VARARGS | METH_KEYWORDS, radau_doc},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "dae4py",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_dae4py(void)
{
    import_array();
    return PyModule_Create(&module);
}