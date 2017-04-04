# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html



'''
Example module to makes simulations and measures on the APART protocol. This
module allows to run all three other example measures simultaneously.

That is, it runs network simulations, and computes all the statistics of modules
:mod:`~measure_general_efficiency`, :mod:`~measure_privacy`, and
:mod:`~measure_route_prop`. 
'''

from collections import OrderedDict, defaultdict
import functools
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

from apart.core.network import Network, NetworkParams
from apart.simulation import  SimulationParams
from common.custom_logging import * 
from common.utilities import print_recursive_structure
import measure_general_efficiency as measure_general_efficiency
import measure_privacy as measure_privacy
import measure_route_prop as measure_route_prop
from measures.file_handling import load_measures_results
from measures.common_measures import MeasureException, generic_measure, \
    preprocess_measure_input_params, \
    make_params_combinations, merge_measures_outputs_with_prefix, \
    merge_measures_outputs


MEASURE_TITLE = "ALL"

_MEASURE_PARAMS_DEFAULT = OrderedDict([('nb_nodes', (int, 30)),
                                ('corruption_ratio', (float, 0.33)),
                                ('ocom_sessions', ((tuple, dict, defaultdict), (5, 10))),
                                ('rtprop_policy_max_routes', (int, 3)),
                                ('rtprop_policy_p_reaccept', (float, 0.7)),
                                ('rtprop_policy_p_replace', (float, 0.5)),
                                ('batching_t_interval', (int,  1*60*1000)),
                                ('batching_nmin', (int, 2)),
                                ('batching_f', (float, 0.8)),
                                ('dummypol_fdum', (float, 0.8)),
                                ('dummypol_deltar', (int, 2))
                                ])

MEASURE_PARAMS = list(_MEASURE_PARAMS_DEFAULT.keys())

_MODULES = [measure_route_prop, measure_general_efficiency, measure_privacy]
        

# Default folder where to save the 
_DEFAULT_SAVING_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Measures/'))
if not os.path.exists(_DEFAULT_SAVING_FOLDER):
    os.makedirs(_DEFAULT_SAVING_FOLDER, mode=0o775)


def measure_all(save_results_to=None, save_network_states=True, restrict_to_params=None, overwrite_existing=False, force_recomputation=False, **measure_params):
    """Calls :func:`measures.common_measures.generic_measure` after some pre-processing.
    
    Here, the processing mainly consists in combining all the simulation and
    network parameters of all three other example modules
    :mod:`~measure_general_efficiency`, :mod:`~measure_privacy`, and
    :mod:`~measure_route_prop`.
    """
    if save_results_to is None:
        save_results_to = os.path.join(_DEFAULT_SAVING_FOLDER, "ALL")
    
    # Make sure all parameters are well included
    all_params_iterator = itertools.chain(*(mod.MEASURE_PARAMS for mod in _MODULES))
    
    missing_params = [p for p in all_params_iterator if p not in MEASURE_PARAMS]
    if missing_params:
        raise MeasureException()
     
    preprocess_measure_input_params(_MEASURE_PARAMS_DEFAULT, measure_params)
    
    # Custom check for the 'ocom_session' parameter
    for i in range(len(measure_params['ocom_sessions'])):
        if isinstance(measure_params['ocom_sessions'][i], tuple):
            measure_params['ocom_sessions'][i] = defaultdict(functools.partial(lambda x: x, measure_params['ocom_sessions'][i]))
        if isinstance(measure_params['ocom_sessions'][i], defaultdict):
            measure_params['ocom_sessions'][i][0] = measure_params['ocom_sessions'][i][0]
 
    
    
    nb_iters= measure_params.pop('nb_iters')
    nb_iters_same_topo_graph= measure_params.pop('nb_iters_same_topo_graph')
    net_params_combinations = make_params_combinations(nb_iters, nb_iters_same_topo_graph, MEASURE_PARAMS, measure_params)
    
    # Get the sim_params of each module
    measures_sim_params = [mod.get_sim_params() for mod in _MODULES]
    
    sim_params = SimulationParams()
    sim_params.logging_level = logging.WARNING  # Display only Warnings or errors
    sim_params.print_nodes_tables = False  # In general, no one will be there to see the stdout or graphs, so don't draw them
    sim_params.draw_topology_graph = False
    sim_params.draw_routes = None
    sim_params.time_of_simulation = 0 # Go until the end of the simulation
    sim_params.automatic_oriented_comm_phase = True 
    sim_params.oriented_communication_sessions = None
    
    # Set each simulation parameter to a disjunction of its value in the three
    # modules (measure_route_prop, measure_general_efficiency, measure_privay)
    sim_params.log_ocom_latency = any(sp.log_ocom_latency for sp in measures_sim_params)
    sim_params.log_end_topo_diss = any(sp.log_end_topo_diss for sp in measures_sim_params)
    sim_params.log_end_ocom_phase = any(sp.log_end_ocom_phase for sp in measures_sim_params)
    sim_params.log_and_store_all_real_msgs = any(sp.log_and_store_all_real_msgs for sp in measures_sim_params)
    sim_params.log_dummy_link_msgs = any(sp.log_dummy_link_msgs for sp in measures_sim_params)
    sim_params.log_real_link_msgs = any(sp.log_real_link_msgs for sp in measures_sim_params)
    sim_params.log_real_e2e_msgs = any(sp.log_real_e2e_msgs for sp in measures_sim_params)
    sim_params.log_histogram_real_msgs_per_round = any(sp.log_histogram_real_msgs_per_round for sp in measures_sim_params)
    sim_params.log_traffic_rates_equilibrium = any(sp.log_traffic_rates_equilibrium for sp in measures_sim_params)
    sim_params.log_real_msgs_waiting_time = any(sp.log_real_msgs_waiting_time for sp in measures_sim_params)
    sim_params.log_histogram_real_msgs_waiting_time = any(sp.log_histogram_real_msgs_waiting_time for sp in measures_sim_params)
    sim_params.log_e2e_dummies = any(sp.log_e2e_dummies for sp in measures_sim_params)
    sim_params.log_frequency_batch_intervention = any(sp.log_frequency_batch_intervention for sp in measures_sim_params)
    sim_params.log_route_props = any(sp.log_route_props for sp in measures_sim_params)
    sim_params.log_sent_link_msgs_per_round = any(sp.log_sent_link_msgs_per_round for sp in measures_sim_params)
    sim_params.log_rt_props_latency = any(sp.log_rt_props_latency for sp in measures_sim_params)
    sim_params.log_ocom_routes = any(sp.log_ocom_routes for sp in measures_sim_params)


    return generic_measure(MEASURE_TITLE, save_results_to, 
                           net_params_combinations, 
                           metrics_computer_callback=compute_metrics_one_network_run,
                           sim_params=sim_params,
#                            net_params=net_params,
                           save_network_states=save_network_states,
                           restrict_to_params=restrict_to_params,
                           overwrite_existing=overwrite_existing,
                           force_recomputation=force_recomputation)

def compute_metrics_one_network_run(aggregated_metrics, net_mngr):
    """Callback function for this measure module.
    
    Computes the statistics on the network state at the end of a run, and
    aggregates the statistics over several network runs.
    
    Arguments:
        aggregated_metrics
        net_mngr(:obj:`apart.core.networtk.NetworkManager`): the network 
            manager, containing in particular the network state after the run. 
    
    Returns:
        dict, dict: The resulting statistics

        The first dict contains the *aggregated statistics. That is, d[k] = v,
        where k is the metric measure_name, and v is *aggregated* metric value over all
        iterations (of a particular parameters combination) that happened so far

        The second dict contains the statistics for the one network unr. That
        is, dict: d[k] = v, where k is the metric measure_name, and v is the metric
        value. This value can be an int, a float, a dict, etc.
    """ 
    
    # This function merely calls the callback of each module (measure_route_prop, measure_general_efficiency, measure_privay)
    
    
    mod_aggregated_metrics = {} 
    mod_this_iteration_metrics = {}
    
    for mod in _MODULES:
        mod_aggregated_metrics[mod], mod_this_iteration_metrics[mod] = mod.compute_metrics_one_network_run(aggregated_metrics, net_mngr)
        
        
    aggregated_metrics = {}
    this_iteration_metrics = {}
    for mod in _MODULES:
        for k, v in mod_aggregated_metrics[mod].items():
            aggregated_metrics[k] = v
        for k, v in mod_this_iteration_metrics[mod].items():
            if k in mod_this_iteration_metrics:
                raise MeasureException("Conflict in aggregation of results of all measure modules: "
                                       "statistic named '{}' seems to be present in at least two modules".format(k))
            this_iteration_metrics[k] = v

    
    return aggregated_metrics, this_iteration_metrics



if __name__ == '__main__':
    folder = os.path.join(_DEFAULT_SAVING_FOLDER, 'measure_'+MEASURE_TITLE)
    
    do_measures = True
    do_plot = True
    
    if do_measures:
        measure_all(nb_nodes=[7, 10, 15],
                    nb_iters=1,
                    nb_iters_same_topo_graph=2,
                    save_results_to=folder,
                    overwrite_existing=True)
    
    if do_plot:
        results = load_measures_results(folder, MEASURE_TITLE+".pickle")
        folder_graphs = os.path.join(folder, "graphs")
        
        for mod in [measure_privacy, measure_general_efficiency, measure_route_prop]:
            print("\n"+("-"*80)+"\n\nPlotting graphs for module {}".format(mod.MEASURE_TITLE))
            folder_graphs = os.path.join(folder, "graphs_"+mod.MEASURE_TITLE)
            if mod is measure_privacy:
                print("\nPlotting stat {}.{}".format(mod.MEASURE_TITLE, "RECEIVER_ANONYMITY"))
                mod.plot_RECEIVER_ANONYMITY_vs_theoretical(results, save_to=folder_graphs, overwrite_existing=True)
            else:
                for stat in mod.MEASURE_STATS:
                    print("\nPlotting stat {}.{}".format(mod.MEASURE_TITLE, stat))
                    mod.__dict__["plot"+stat](results, save_to=folder_graphs, overwrite_existing=True)

    
    
   