=============================================================
``measure_route_prop`` --- Study the route proposal mechanism
=============================================================

.. automodule:: measure_route_prop


Source Code
-----------

Refer to the source code for more detail. It is depicted below, and can be
downloaded :download:`here <../../examples/measure_route_prop.py>`.

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

* The function :func:`measure_route_prop`), which basically calls
  :func:`measures.common_measures.generic_measure`. Prior to this calls, it
  performs checks on the simulation and network parameters.
* The :func:`get_sim_params` function, returning an instance of
  :obj:`~apart.simulation.SimulationParams` with custom parameter values.
* The :func:`compute_metrics_one_network_run` function, which is the callback
  function to provide as ``metric_computer_callback`` to 
  :func:`measures.common_measures.generic_measure`.
* One plotting function for each metric computed by the module. Each such
  function is named by prepending `plot_` to the metric name, is decorated
  with :func:`measures.plotting.plotfunc`, and basically consists in calling
  :func:`measures.plotting.plot_simple_metric` or
  :func:`measures.plotting.plot_histogram`.
  
  * The functions :func:`plot_plot_HIST_ACCEPT_REFUSE_ROUTES` and
    :func:`plot_HIST_ROUTES_LENGTH` are not decorated, because they are
    histograms, and not *simple metrics*.
  * The function :func:`plot_ROUTES_JACCARD_DISTANCE` pertains to a more complex
    metric, and necessitates custom processing of results. It directly calls
    :func:`measures.plotting.do_plot_simple_metric`.

* A *main* part, allowing to perform measures and/or plot previously acquired
  measure results.



.. literalinclude:: ../../examples/measure_route_prop.py
