=======================================
``measure_privacy`` --- Measure Privacy
=======================================

.. automodule:: measure_privacy


Source Code
-----------

Refer to the source code for more detail. It is depicted below, and can be
downloaded :download:`here <../../examples/measure_privacy.py>`.

The code is outlined as follows:

* Definitions of constants:

        * The metrics (statistics names) that the module computes
          (:attr:`MEASURE_STATS`).

          .. note::
                The value for each constant listed in :attr:`MEASURE_STATS` is a
                string (with spelling mistakes, I know). However, it can be
                anything as long as it is serialisable data, and ensures
                uniqueness of values. We chose strings for it makes its easy to
                identify metrics during debug, or when printing results out on
                stdout.

        * The name of the measure defined in this module
          (:attr:`MEASURE_TITLE`).
        * The parameters relevant for this measure module
          (:attr:`MEASURE_PARAMS`), *i.e.* network and  simulation
          parameters (see :class:`~apart.simulation.SimulationParams` and
          :class:`~apart.core.network.NetworkParams`).

* The function :func:`measure_general_efficiency`), which basically calls
  :func:`measures.common_measures.generic_measure`. Prior to this calls, it
  performs checks on the simulation and network parameters.
* The :func:`get_sim_params` function, returning an instance of
  :obj:`~apart.simulation.SimulationParams` with custom parameter values.
* The :func:`compute_metrics_one_network_run` function, which is the callback
  function to provide as ``metric_computer_callback`` to 
  :func:`measures.common_measures.generic_measure`.
* Then, this module defines only *one* plotting function:
  :func:`plot_RECEIVER_ANONYMITY_vs_theoretical`. Contrarily to plotting
  functions in the other example measure modules, the latter function is however
  very complex, and broken down over several module-private functions. Indeed,
  in this module, the only information gathered during the network simulations,
  as per the definition of :func:`compute_metrics_one_network_run`, are the
  description of routes (*i.e.* a list of routes, as given by
  :attr:`~measures.advanced_stats_helper.AdvancedNetworkStatsHelper.AdvancedNetworkStatsHelperSingleton.complete_routes_descriptions`
  from the :mod:`measures.advanced_stats_helper` module. It is then the role of
  function :func:`plot_RECEIVER_ANONYMITY_vs_theoretical` to process these
  routes descriptions in order to deduce the receiver anonymity. Note that the
  function accepts corruptions ratios as argument (with the
  ``results_augmentation`` a parameter).
* A *main* part, allowing to perform measures and/or plot previously acquired
  measure results.



.. literalinclude:: ../../examples/measure_privacy.py
