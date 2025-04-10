#ifndef COMMON_H
#define COMMON_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "numpy/arrayobject.h"

static PyObject* linspace(double start, double stop, int num) {
    // check for valid number of points
    if (num <= 0) {
        PyErr_SetString(PyExc_ValueError, "linspace: Number of points must be greater than 0");
        return NULL;
    }

    // calculate the step-size
    double step = (num > 1) ? (stop - start) / (num - 1) : 0.0;

    // create a NumPy array of doubles
    npy_intp dims[1] = {num};  // dimension of the array
    PyObject* array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);  // 1D array of type double
    if (!array) {
        PyErr_SetString(PyExc_RuntimeError, "linspace: Failed to create NumPy array");
        return NULL;
    }

    // Fill the array with linearly spaced values
    double* data = (double*)PyArray_DATA((PyArrayObject*)array);
    for (int i = 0; i < num; i++) {
        data[i] = start + i * step;
    }

    return array;
}

// create _RichResult from given dictionary
static PyObject* rich_result(PyObject* dict) {
    PyObject* util_module = PyImport_ImportModule("scipy._lib._util");
    if (util_module == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyImport_ImportModule('scipy._lib._util') failed");
    }

    PyObject* rich_result_class = PyObject_GetAttrString(util_module, "_RichResult");
    Py_XDECREF(util_module);
    if (rich_result_class == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyObject_GetAttrString(util_module, '_RichResult') failed");
    }

    PyObject* args = PyTuple_Pack(1, dict);
    PyObject* result = PyObject_CallObject(rich_result_class, args);
    Py_XDECREF(args);
    Py_XDECREF(rich_result_class);
    Py_XDECREF(dict);

    return result;
}

#endif // COMMON_H
