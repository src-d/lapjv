.. lapjv documentation master file, created by
   sphinx-quickstart on Mon Jun  5 16:52:34 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lapjv's documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. reference:

*************
API Reference
*************


:mod:`lapjv` --- Linear Assignment Problem solver using Jonker-Volgenant algorithm
==================================================================================

This project is the rewrite of `pyLAPJV <https://github.com/hrldcpr/pyLAPJV>`_ which
supports Python 3 and updates the core code. The performance is twice as high as
the original thanks to the optimization of the augmenting row reduction phase
using Intel AVX intrinsics. It is a native Python 3 module and does
not work with Python 2.x, stick to pyLAPJV otherwise.

`Blog post. <https://blog.sourced.tech/post/lapjv/>`_

`Linear assignment problem <https://en.wikipedia.org/wiki/Assignment_problem>`_
is the bijection between two sets with equal cardinality which optimizes the sum
of the individual mapping costs taken from the fixed cost matrix. It naturally
arises e.g. when we want to fit `t-SNE <https://lvdmaaten.github.io/tsne/>`_ results
into a rectangular regular grid.
See this awesome notebook for the details about why LAP matters:
`CloudToGrid <https://github.com/kylemcdonald/CloudToGrid/blob/master/CloudToGrid.ipynb>`_.

Jonker-Volgenant algorithm is described in the paper:

R. Jonker and A. Volgenant, "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems," _Computing_, vol. 38, pp. 325-340, 1987.

This paper is not publicly available though a brief description exists on
`sciencedirect.com <http://www.sciencedirect.com/science/article/pii/S0166218X99001729>`_.
JV is faster in than the `Hungarian algorithm <https://en.wikipedia.org/wiki/Hungarian_algorithm>`_ in practice,
though the complexity is the same - O(n\ :sup:`3`).

The C++ source of the algorithm comes from http://www.magiclogic.com/assignment.html
It has been reworked and partially optimized with OpenMP 4.0 SIMD.

.. function:: lapjv(cost_matrix, verbose=0, force_doubles=False)

   Solves the LAP.

   :module: lapjv
   :param numpy.ndarray cost_matrix: must be a square 2D numpy array of either :class:`numpy.float32` or :class:`numpy.float64` :code:`dtype`.
   :param int verbose: sets the logging level: 0 means complete silence, 1 is INFO and 2 is DEBUG.
   :param bool force_doubles: forces the double precision in the actual problem solution when the :code:`dtype` is :code:`float64`.
   :return: (row assignments, column assignments, (solution cost, u array, v array)), all arrays of 1D shape - :code:`cost_matrix`'s size.
   :rtype: (numpy.ndarray, numpy.ndarray, (float, numpy.ndarray, numpy.ndarray))
   :raises TypeError: some argument has a wrong type.
   :raises ValueError: cost_matrix has an invalid shape or :code:`dtype`.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
