#include <functional>
#include <memory>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "lap.h"

static char module_docstring[] =
    "This module wraps LAPJV - Jonker-Volgenant linear sum assignment algorithm.";
static char lapjv_docstring[] =
    "Solves the linear sum assignment problem.";

static PyObject *py_lapjv(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"lapjv", reinterpret_cast<PyCFunction>(py_lapjv),
   METH_VARARGS | METH_KEYWORDS, lapjv_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_lapjv(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "lapjv",             /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  return m;
}
}

template <typename O>
using pyobj_parent = std::unique_ptr<O, std::function<void(O*)>>;

template <typename O>
class _pyobj : public pyobj_parent<O> {
 public:
  _pyobj() : pyobj_parent<O>(
      nullptr, [](O *p){ if (p) Py_DECREF(p); }) {}
  explicit _pyobj(PyObject *ptr) : pyobj_parent<O>(
      reinterpret_cast<O *>(ptr), [](O *p){ if(p) Py_DECREF(p); }) {}
  void reset(PyObject *p) noexcept {
    pyobj_parent<O>::reset(reinterpret_cast<O*>(p));
  }
};

using pyobj = _pyobj<PyObject>;
using pyarray = _pyobj<PyArrayObject>;

static PyObject *py_lapjv(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *cost_matrix_obj;
  int verbose = 0;
  int force_doubles = 0;
  static const char *kwlist[] = {
      "cost_matrix", "verbose", "force_doubles", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "O|pb", const_cast<char**>(kwlist),
      &cost_matrix_obj, &verbose, &force_doubles)) {
    return NULL;
  }
  pyarray cost_matrix_array;
  bool float32 = true;
  cost_matrix_array.reset(PyArray_FROM_OTF(
      cost_matrix_obj, NPY_FLOAT32,
      NPY_ARRAY_IN_ARRAY | (force_doubles? 0 : NPY_ARRAY_FORCECAST)));
  if (!cost_matrix_array) {
    PyErr_Clear();
    float32 = false;
    cost_matrix_array.reset(PyArray_FROM_OTF(
        cost_matrix_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    if (!cost_matrix_array) {
      PyErr_SetString(PyExc_ValueError, "\"cost_matrix\" must be a numpy array "
                                        "of float32 or float64 dtype");
      return NULL;
    }
  }
  auto ndims = PyArray_NDIM(cost_matrix_array.get());
  if (ndims != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\" must be a square 2D numpy array");
    return NULL;
  }
  auto dims = PyArray_DIMS(cost_matrix_array.get());
  if (dims[0] != dims[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\" must be a square 2D numpy array");
    return NULL;
  }
  int dim = dims[0];
  if (dim <= 0) {
    PyErr_SetString(PyExc_ValueError,
                    "\"cost_matrix\"'s shape is invalid or too large");
    return NULL;
  }
  auto cost_matrix = PyArray_DATA(cost_matrix_array.get());
  npy_intp ret_dims[] = {dim, 0};
  pyarray row_ind_array(PyArray_SimpleNew(1, ret_dims, NPY_INT));
  pyarray col_ind_array(PyArray_SimpleNew(1, ret_dims, NPY_INT));
  auto row_ind = reinterpret_cast<int*>(PyArray_DATA(row_ind_array.get()));
  auto col_ind = reinterpret_cast<int*>(PyArray_DATA(col_ind_array.get()));
  pyarray u_array(PyArray_SimpleNew(
      1, ret_dims, float32? NPY_FLOAT32 : NPY_FLOAT64));
  pyarray v_array(PyArray_SimpleNew(
      1, ret_dims, float32? NPY_FLOAT32 : NPY_FLOAT64));
  double lapcost;
  if (float32) {
    auto u = reinterpret_cast<float*>(PyArray_DATA(u_array.get()));
    auto v = reinterpret_cast<float*>(PyArray_DATA(v_array.get()));
    Py_BEGIN_ALLOW_THREADS
    lapcost = lap(dim, reinterpret_cast<float*>(cost_matrix), verbose,
                  row_ind, col_ind, u, v);
    Py_END_ALLOW_THREADS
  } else {
    auto u = reinterpret_cast<double*>(PyArray_DATA(u_array.get()));
    auto v = reinterpret_cast<double*>(PyArray_DATA(v_array.get()));
    Py_BEGIN_ALLOW_THREADS
    lapcost = lap(dim, reinterpret_cast<double*>(cost_matrix), verbose,
                  row_ind, col_ind, u, v);
    Py_END_ALLOW_THREADS
  }
  return Py_BuildValue("(OO(dOO))",
                       row_ind_array.get(), col_ind_array.get(), lapcost,
                       u_array.get(), v_array.get());
}
