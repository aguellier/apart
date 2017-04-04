'''
This package provides the basis for carrying out many simulations, gathering
statistics about them, and utlimately measure the performances and provacy of
the protocol implementation in the :mod:`~apart` package.

In particular, the module :mod:`~measures.common_measures` provides the function
:func:`~measures.common_measures.generic_measure`, which allows to run a set of
network simulations under parameters chosen by the user, and to automatically
compute and aggregate statistics on these runs. 

.. seealso::
    See the examples measure modules that make use of the
    :func:`measures.common_measures.generic_measure` function. These may be
    found in the the `examples/` folder, or in the *Example* topic of the
    documentation.
'''