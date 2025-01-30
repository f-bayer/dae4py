#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "numpy/arrayobject.h"

#include "dassl/dassl.h"
#include "pside/pside.h"
#include "radau/radau.h"

PyDoc_STRVAR(doc,
"Function signature: solve(f, t_span, y0, yp0, rtol=1e-6, atol=1e-3, J=None, t_eval=None)\n"
"\n"
"Solve a DAE system f(t, y, y') = 0.\n"
"\n"
"Parameters\n"
"----------\n"
"f : callable\n"
"    A Python function that defines the DAE system. The function \n"
"    must have the signature `f(t, y, yp)`, where `t` is the\n"
"    current time, `y` is the state vector, and `yp` is the \n"
"    derivative of the state vector.\n"
"\n"
"t_span : array-like\n"
"    A 2-element list or array defining the time interval `[t_start, t_end]`\n"
"    over which to integrate the system.\n"
"\n"
"y0 : array-like\n"
"    The initial conditions for the state vector `y` at the start of the integration.\n"
"\n"
"yp0 : array-like\n"
"    The initial conditions for the derivative of the state vector `yp` at the start of the integration.\n"
"\n"
"rtol, atol: float (optional)\n"
"    The used relative and absolute tolerances. Default values: rtol=1e-6, atol=1e-3.\n"
"\n"
"t_eval: array-like (optional)\n"
"      The requested evaluation points. If not given, 500 equidistance points in t_span are chosen.\n"
"\n"
"Returns\n"
"-------\n"
"result : dict\n"
"    A dictionary containing the results of the integration. The dictionary has the following keys:\n"
"    - 'success': Was the integration successful?\n"
"    - 'order': List of used integration order.\n"
"    - 't': List of time points.\n"
"    - 'y': List of stage values corresponding to t.\n"
"    - 'yp': List of stage derivatives corresponding to t.\n"
"    - 'nsteps': Total number of steps.\n"
"    - 'nf': Number of function evaluations.\n"
"    - 'njac': Number of jacobian evaluations.\n"
"    - 'nrejerror': Number of error tests failures.\n"
"    - 'nrejnewton': Number of convergence tests failures.\n"
"\n"
"Examples\n"
"--------\n"
"def f(t, y, yp):\n"
"    # Example: A simple harmonic oscillator\n"
"    return np.array([\n"
"        yp[0] - y[1],\n"
"        yp[1] + y[0],\n"
"    )]\n"
"result = integrate(f, [0, 10], [1.0, 0.0], [0.0, -1.0])\n"
"print(result)\n"
"print(result['t'])\n"
"print(result['y'])\n"
"print(result['yp'])"
);

static PyMethodDef methods[] = {
    {"dassl", (PyCFunction)dassl, METH_VARARGS | METH_KEYWORDS, doc},
    {"pside", (PyCFunction)pside, METH_VARARGS | METH_KEYWORDS, doc},
    {"radau5", (PyCFunction)radau5, METH_VARARGS | METH_KEYWORDS, doc},
    {"radau", (PyCFunction)radau, METH_VARARGS | METH_KEYWORDS, doc},
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