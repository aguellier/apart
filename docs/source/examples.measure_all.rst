======================================================
``measure_all`` --- Perform all three measures at once
======================================================

.. automodule:: measure_all


Source Code
-----------

Refer to the source code for more detail. It is depicted below, and can be
downloaded :download:`here <../../examples/measure_all.py>`.

The code is outlined as follows:

* Definitions of constants:

        * The name of the measure defined in this module
          (:attr:`MEASURE_TITLE`).
        * The parameters relevant for this measure module
          (:attr:`MEASURE_PARAMS`), *i.e.* network and  simulation parameters
          (see :class:`~apart.simulation.SimulationParams` and
          :class:`~apart.core.network.NetworkParams`). In this module, all the
          parameters of the simulation and of the network can be tuned and their
          impact measured simultaneously.
        * Note that no :attr:`MEASURE_STATS` constant is defined. It is
          implicitly assumed to be the union of the three modules'
          :attr:`MEASURE_STATS` constant.

* The function :func:`measure_all`, which sets the simulation and network
  parameters accordingly to their values in each of the three modules. In
  particular, each :attr:`SimulationParams.log_` boolean is set to a disjunction
  of its value defined in each of the three module. Then, the function 
  :func:`measures.common_measures.generic_measure` is called.
  performs checks on the simulation and network parameters.
* The :func:`get_sim_params` function, returning an instance of
  :obj:`~apart.simulation.SimulationParams` with custom parameter values.
* The :func:`compute_metrics_one_network_run` function, which is the callback
  function to provide as ``metric_computer_callback`` to
  :func:`measures.common_measures.generic_measure`. Here,this function is
  designed to call the corresponding callback from each of the three modules,
  and merges their output. 
* A *main* part, allowing to perform measures and/or plot previously acquired
  measure results.

.. note::
        Contrarily to other example modules, this one does not define *plotting*
        functions. Instead, it directly calls the relevant plotting functions
        from the said modules (see the end of the *main* part of the code).


.. literalinclude:: ../../examples/measure_all.py
