#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "numpy/arrayobject.h"

#ifdef HAVE_BLAS_ILP64
#define F_INT npy_int64
#define F_INT_NPY NPY_INT64
#else
#define F_INT int
#define F_INT_NPY NPY_INT
#endif

typedef struct _radau_globals {
    PyObject *python_function;
    PyObject *t_sol;
    PyObject *y_sol;
    PyObject *yp_sol;
} radau_params;

static radau_params global_radau_params = {NULL, NULL, NULL, NULL};

#if defined(UPPERCASE_FORTRAN)
    #if defined(NO_APPEND_FORTRAN)
        /* nothing to do here */
    #else
        #define RADAU5  RADAU5_
        #define RADAU  RADAU_
    #endif
#else
    #if defined(NO_APPEND_FORTRAN)
        #define RADAU5  radau5
        #define RADAU  radau
    #else
        #define RADAU5  radau5_
        #define RADAU  radau_
    #endif
#endif

typedef void radau_f_t(F_INT *neq, double *t, double *y, 
                       double *f, double *rpar, F_INT *ipar);
typedef void radau_jac_t(double *t, double *y, double *ydot, 
                         double *J, double* cj, 
                         double *rpar, F_INT *ipar);
typedef void radau_mas_t(F_INT *neq, double *am, F_INT *lmas,
                         double *rpar, F_INT *ipar);

typedef void radau_solout_t(F_INT *nr, double *told, double *t, double *y, 
                            double *contr, F_INT *lrc, F_INT *neqn,
                            double *rpar, F_INT *ipar, F_INT *itrn);

// function signature of radau calls
typedef void radau_t(F_INT *neq, radau_f_t *f, double *t, 
                     double *y, double *tout, double *h, 
                     double* rtol, double *atol, F_INT* itol, 
                     radau_jac_t *jac, F_INT *ijac /*should be boolean*/, 
                     F_INT *mljac /*should be boolean*/, F_INT *mujac, 
                     radau_mas_t *mas, F_INT *imas /*should be boolean*/, 
                     F_INT *mlmas /*should be boolean*/, F_INT *mumas, 
                     radau_solout_t *solout, F_INT *iout,
                     double *rwork, F_INT *lwork, F_INT *iwork, F_INT *liwork, 
                     double *rpar, F_INT *ipar, F_INT *idid);

radau_t RADAU;
radau_t RADAU5;

// f(t, y) = [
//  u' = v
//  0 = f(t, u, v)
// ]
void radau_f(F_INT *neqn, double *t, double *y, 
            double *f, double *rpar, F_INT *ipar)
{
    // python objects
    PyObject *u_obj = NULL;
    PyObject *v_obj = NULL;
    PyObject *result = NULL;
    PyObject *arglist = NULL;
    PyArrayObject *u_array = NULL;
    PyArrayObject *v_array = NULL;
    PyArrayObject *result_array = NULL;

    // dimension of implicit differential equation since y = (u, v)
    F_INT n = (*neqn) / 2;
    npy_intp dims[1];
    dims[0] = n;

    // decompose y = (u, v)
    double *u = y;
    double *v = y + n;

    /* Build numpy arrays from u and v. */
    u_obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u);
    if (u_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u) failed.");
        goto fail;
    }
    v_obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, v);
    if (v_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, v) failed.");
        goto fail;
    }

    /* Build argument list. */
    arglist = Py_BuildValue(
        "dOO",
        *t,
        u_obj,
        v_obj
    );
    if (arglist == NULL) {
        PyErr_SetString(PyExc_ValueError, "Py_BuildValue failed.");
        goto fail;
    }

    /* Call the Python function. */
    result = PyObject_CallObject(global_radau_params.python_function, arglist);
    if (result == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyObject_CallObject(global_radau_params.python_function, arglist) failed.");
        goto fail;
    }

    /* Build numpy array from u, v and result. */
    u_array = (PyArrayObject *) PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0);
    if (u_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0) failed.");
        goto fail;
    }
    v_array = (PyArrayObject *) PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0);
    if (v_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0) failed.");
        goto fail;
    }
    result_array = (PyArrayObject *) PyArray_ContiguousFromObject(result, NPY_DOUBLE, 0, 0);
    if (result_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(result, NPY_DOUBLE, 0, 0) failed.");
        goto fail;
    }

    /* Copy data from the result array to C array. */
    // u' = v
    // 0 = f(t, u, v)
    memcpy(f, PyArray_DATA(v_array), PyArray_NBYTES(v_array));
    memcpy(f + n, PyArray_DATA(result_array), PyArray_NBYTES(result_array));
    // double *res = (double *) PyArray_DATA(result_array);
    // for (int i = 0; i < n; i++) {
    //     f[i] = v[i];
    //     f[i + n] = res[i];
    // }

    fail:
        Py_XDECREF(u_obj);
        Py_XDECREF(v_obj);
        Py_XDECREF(result);
        Py_XDECREF(arglist);
        Py_XDECREF(v_array);
        Py_XDECREF(u_array);
        Py_XDECREF(result_array);
        return;
}

void radau_jac(F_INT *neqn, double *t, double *y, double *dfy, 
               F_INT *ldfym, double *rpar, F_INT *ipar){}

void radau_mas(F_INT *neqn, double *am, F_INT *lmas,
               double *rpar, F_INT *ipar) {
    int n = (*neqn) / 2;

    // banded
    for (int j = 0; j < (*neqn); j++) {
        if (j < n) {
            am[j] = 1.0;
        } else {
            am[j] = 0.0;
        }
    }
}

void radau_solout(F_INT *nr, double *told, double *t, double *y, 
                  double *contr, F_INT *lrc, F_INT *neqn,
                  double *rpar, F_INT *ipar, F_INT *itrn) {

    PyObject *u_obj = NULL;
    PyObject *v_obj = NULL;
    PyArrayObject *u_array = NULL;
    PyArrayObject *v_array = NULL;

    // dimension of implicit differential equation since y = (u, v)
    F_INT n = (*neqn) / 2;
    npy_intp dims[1];
    dims[0] = n;

    // decompose y = (u, v)
    double *u = y;
    double *v = y + n;

    /* Build numpy arrays from u and v. */
    u_obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u);
    if (u_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u) failed.");
        goto fail;
    }
    v_obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, v);
    if (v_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, v) failed.");
        goto fail;
    }
    u_array = (PyArrayObject *) PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0);
    if (u_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0) failed.");
        goto fail;
    }
    v_array = (PyArrayObject *) PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0);
    if (v_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0) failed.");
        goto fail;
    }

    PyList_Append(global_radau_params.t_sol, PyFloat_FromDouble(*t));
    PyList_Append(global_radau_params.y_sol, PyArray_NewCopy(u_array, NPY_ANYORDER));
    PyList_Append(global_radau_params.yp_sol, PyArray_NewCopy(v_array, NPY_ANYORDER));

    fail:
        Py_XDECREF(u_obj);
        Py_XDECREF(v_obj);
        Py_XDECREF(v_array);
        Py_XDECREF(u_array);
        return;
}

// TODO:
// - add dense output for solout
// - add possibility for index 2 and 3 varibles to iwork
// - allow for more options in iwork
static PyObject* radau_call(PyObject *self, PyObject *args, PyObject *kwargs, radau_t *radau_)
{
    PyObject *f_obj = NULL;
    PyObject *J_obj = Py_None;
    PyObject *t_span_obj = NULL;
    PyObject *u_obj = NULL;
    PyObject *v_obj = NULL;
    PyArrayObject *u_array = NULL;
    PyArrayObject *v_array = NULL;

    double rtol = 1.0e-3;
    double atol = 1.0e-6;
    double h = 1e-3;
    double t, t1;
    double *u, *v, *y;

    int success;

    int n;
    int neqn;
    int iout = 1; // use solout

    int ijac;
    int mljac;
    int mujac;
    
    int imas;
    int mlmas;
    int mumas;

    int itol = 0; // scalar tolerances

    int lrwork;
    int liwork;
    double *rwork;
    int *iwork;

    double *rpar;
    int *ipar;
    int idid;

    // parse inputs
    static char *kwlist[] = {"f", "t_span", "y0", "yp0", // mandatory arguments
                             "rtol", "atol", "J", NULL}; // optional arguments and NULL termination
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|ddOOO", kwlist, 
                                     &f_obj, &t_span_obj, &u_obj, &v_obj, // positional arguments
                                     &rtol, &atol, &J_obj)) // optional arguments
        return NULL;

    // check if function and Jacobians (if present) are callable
    if (!PyCallable_Check(f_obj)) {
        PyErr_SetString(PyExc_ValueError, "`f` must be a callable function.");
    }
    if (J_obj != Py_None) {
        if (!PyCallable_Check(J_obj)) {
            PyErr_SetString(PyExc_ValueError, "`J` must be a callable function.");
        }
        ijac = 1;
        PyErr_SetString(PyExc_ValueError, "User-defined Jacobian `J` is not implemented yet.");
    } else {
        ijac = 0; 
    }

    // unpack t_span tuple
    PyArg_ParseTuple(t_span_obj, "dd", &t, &t1);
    if (!(t1 > t)) {
        PyErr_SetString(PyExc_ValueError, "`t1` must larger than `t0`.");
    }

    // initial conditions
    u_array = (PyArrayObject *) PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0);
    if (u_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0) failed");
        goto fail;
    }
    if (PyArray_NDIM(u_array) > 1) {
        PyErr_SetString(PyExc_ValueError, "Initial condition y0 must be one-dimensional.");
        goto fail;
    }
    u = (double *) PyArray_DATA(u_array);
    n = PyArray_Size((PyObject *) u_array);
    neqn = 2 * n;

    v_array = (PyArrayObject *) PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0);
    if (v_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0) failed");
        goto fail;
    }
    if (PyArray_NDIM(v_array) > 1) {
        PyErr_SetString(PyExc_ValueError, "Initial condition yp0 must be one-dimensional.");
        goto fail;
    }
    v = (double *) PyArray_DATA(v_array);
    if (!(n == PyArray_Size((PyObject *) v_array))) {
        PyErr_SetString(PyExc_ValueError, "Size of y0 and yp0 have to coincide.");
        goto fail;
    }

    // allocate state array and fill with initial conditions
    y = malloc(neqn * sizeof(double));
    memcpy(y, PyArray_DATA(u_array), PyArray_NBYTES(u_array));
    memcpy(y + n, PyArray_DATA(v_array), PyArray_NBYTES(v_array));

    // decompose y = (u, v)
    u = y;
    v = y + n;

    /* Build numpy arrays from u and v. */
    npy_intp dims[1];
    dims[0] = n;

    u_obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u);
    if (u_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u) failed.");
        goto fail;
    }
    v_obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, v);
    if (v_obj == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, v) failed.");
        goto fail;
    }

    u_array = (PyArrayObject *) PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0);
    if (u_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(u_obj, NPY_DOUBLE, 0, 0) failed");
        goto fail;
    }
    v_array = (PyArrayObject *) PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0);
    if (v_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "PyArray_ContiguousFromObject(v_obj, NPY_DOUBLE, 0, 0) failed");
        goto fail;
    }

    // initialize iwork and rwork
    lrwork = 20 + neqn * (neqn + 1 + 7 * neqn + 3 * 7 + 3);
    liwork = 3 * neqn + 20;

    rwork = malloc(lrwork * sizeof(double));
    iwork = malloc(liwork * sizeof(int));

    for (int i=0; i<liwork; i++) {
        iwork[i] = 0;
    }
    for (int i=0; i<lrwork; i++) {
        rwork[i] = 0.0;
    }

    // second order system
    iwork[8] = n;
    iwork[9] = n;

    // full jacobian with finite differences
    ijac = 0;
    mljac = neqn;
    mujac = neqn;
    
    // banded user-defined jacobian
    imas = 1;
    mlmas = 1;
    mumas = 1;

    // set global parameters
    global_radau_params.python_function = f_obj;

    // store solution in python list and start with initial values
    global_radau_params.t_sol = PyList_New(0);
    global_radau_params.y_sol = PyList_New(0);
    global_radau_params.yp_sol = PyList_New(0);
    PyList_Append(global_radau_params.t_sol, PyFloat_FromDouble(t));
    PyList_Append(global_radau_params.y_sol, PyArray_NewCopy(u_array, NPY_ANYORDER));
    PyList_Append(global_radau_params.yp_sol, PyArray_NewCopy(v_array, NPY_ANYORDER));

    // call radau solver
    radau_(&neqn, radau_f, &t, y, &t1, &h, 
           &rtol, &atol, &itol, 
           radau_jac, &ijac, &mljac, &mujac,
           radau_mas, &imas, & mlmas, &mumas,
           radau_solout, &iout, 
           rwork, &lrwork, iwork, &liwork, 
           rpar, ipar, &idid);

    success = idid > 0;

    // cleanup
    free(rwork);
    free(iwork);
    free(y);
    Py_XDECREF(f_obj);
    Py_XDECREF(J_obj);
    Py_XDECREF(t_span_obj);
    Py_XDECREF(u_obj);
    Py_XDECREF(v_obj);
    Py_XDECREF(u_array);
    Py_XDECREF(v_array);
    
    return Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:i,s:i,s:i,s:i,s:i}",
        "success", success ? Py_True : Py_False,
        "t", PyArray_Return(PyArray_FromAny(
                                global_radau_params.t_sol, // Input object
                                NULL,                      // Desired data type (None means let NumPy decide)
                                0,                         // Minimum number of dimensions
                                0,                         // Maximum number of dimensions
                                NPY_ARRAY_DEFAULT,         // Flags
                                NULL)                      // Array description (NULL means default)
                            ),
        "y", PyArray_Return(PyArray_FromAny(
                                global_radau_params.y_sol, // Input object
                                NULL,                      // Desired data type (None means let NumPy decide)
                                0,                         // Minimum number of dimensions
                                0,                         // Maximum number of dimensions
                                NPY_ARRAY_DEFAULT,         // Flags
                                NULL)                      // Array description (NULL means default)
                            ),
        "yp", PyArray_Return(PyArray_FromAny(
                                global_radau_params.yp_sol, // Input object
                                NULL,                       // Desired data type (None means let NumPy decide)
                                0,                          // Minimum number of dimensions
                                0,                          // Maximum number of dimensions
                                NPY_ARRAY_DEFAULT,          // Flags
                                NULL)                       // Array description (NULL means default)
                            ),
        "nf", iwork[13],
        "njac", iwork[14],
        "nsteps", iwork[15],
        "naccpt", iwork[16],
        "nrejerror", iwork[17],
        "nlu", iwork[18],
        "nsol", iwork[19]
    );

    fail:
        free(rwork);
        free(iwork);
        free(y);
        Py_XDECREF(f_obj);
        Py_XDECREF(J_obj);
        Py_XDECREF(t_span_obj);
        Py_XDECREF(u_obj);
        Py_XDECREF(v_obj);
        Py_XDECREF(u_array);
        Py_XDECREF(v_array);
        return NULL;
}

static PyObject* radau(PyObject *self, PyObject *args, PyObject *kwargs) {
    return radau_call(self, args, kwargs, RADAU);
}

static PyObject* radau5(PyObject *self, PyObject *args, PyObject *kwargs) {
    return radau_call(self, args, kwargs, RADAU5);
}
