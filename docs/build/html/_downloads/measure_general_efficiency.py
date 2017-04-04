# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
Example module to makes simulations and measures on the APART protocol. This
module provides a base to measure the efficiency of the protocol.

More specifically, this module measures the following metrics (non exhaustive list):

* Time taken for topology dissemination
* Latency in communications
* Number of rounds each real message is delayed in a message pool, as per the message re-ordering mechanism
* The equilibrium and values of traffic rates constraints.
          
All these metrics are computed with a combination of different parameters range. Namely,

* Number of nodes 
* Dummy policy and controlled traffic rates parameters : delta r and fraction of pools filled with a dummy message at each round
* Message re-ordering parameters : nmin and f
    
The module contains functions for running the measures, and plotting graphs from
network runs saved on disk.
'''

from collections import OrderedDict, defaultdict
import functools
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

from common.custom_logging import * 
from apart.core.network import Network, NetworkParams
from apart.simulation import  SimulationParams
from common.utilities import print_recursive_structure, range1
from measures.advanced_stats_helper import AdvancedNetworkStatsHelper
from measures.file_handling import load_measures_results
from measures.common_measures import MeasureException, generic_measure, \
    preprocess_measure_input_params, \
    make_params_combinations, merge_measures_outputs_with_prefix
from measures.plotting import plotfunc, plot_simple_metric, \
    plot_histogram
from measures.network_statistics import RunningStat


_TOPO_DISS_TIME = "Number of rounds for topology dissemination (general efficiency)"
_OCOM_PHASE_TIME = "Number of rounds needed for the oriented communication phase, in total"
_OCOM_MSG_TIME = "Number of rounds needed for an oriented communication, brought down to the time for 1 message"
_OCOM_MSG_TIME_WITHOUT_INIT = "Number of rounds needed for an oriented communication, brought down to the time for 1 message, without ocom init"
_HIST_WAITING_TIME_REAL_MSGS = "Number of rounds a real message stays in pools in average"
_MEAN_WAITING_TIME_REAL_MSGS = "Number of rounds a real message stays in pools in average, but not a histogram"
_RATIO_REAL_OVER_DUMMY_MSGS = "Ratio of number of real messages sent over number of dummies sent (both link messages), on the whole network"
_EQUILIBRIUM_PER_ROUND = "Traffic rate equilibrium per round"
_RATIO_E2E_DUMMY = "Ratio: number of times one particular node resorts to adding e2e dummies, over the total number of rounds"
_RATIO_DUMMY_BCAST = "Ratio: number of times one particular node resorts to the defaut dummy broadcast, over the total number of rounds"
_HIST_NB_LINK_MSGS_SENT_PER_ROUND = "A prob distrib of the number of link messages a node sends per round (reals and dummies confunded)"


MEASURE_STATS = ['_TOPO_DISS_TIME',
                '_OCOM_PHASE_TIME',
                '_OCOM_MSG_TIME',
                '_OCOM_MSG_TIME_WITHOUT_INIT',
                '_HIST_WAITING_TIME_REAL_MSGS',
                '_MEAN_WAITING_TIME_REAL_MSGS',
                '_RATIO_REAL_OVER_DUMMY_MSGS',
                '_EQUILIBRIUM_PER_ROUND',
                '_RATIO_E2E_DUMMY',
                '_RATIO_DUMMY_BCAST',
                '_HIST_NB_LINK_MSGS_SENT_PER_ROUND']


# Title of the measures performed in this module
MEASURE_TITLE = 'GeneralEfficiency'


# Network parameters that vary for the measures uin this module
_MEASURE_PARAMS_DEFAULT = OrderedDict([('nb_nodes', (int, 30)), 
                                       ('ocom_sessions', ((tuple, dict, defaultdict), (5, 10))),
                                       ('batching_t_interval', (int,  1*60*1000)),
                                       ('batching_nmin', (int, 2)),
                                       ('batching_f', (float, 0.8)),
                                       ('dummypol_fdum', (float, 0.8)),
                                       ('dummypol_deltar', (int, 2))
                                       ])
MEASURE_PARAMS = list(_MEASURE_PARAMS_DEFAULT.keys())


# Default folder where to save the results and graphs
_DEFAULT_SAVING_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Measures/'))
if not os.path.exists(_DEFAULT_SAVING_FOLDER):
    os.makedirs(_DEFAULT_SAVING_FOLDER, mode=0o775)


def measure_general_efficiency(save_results_to=None, save_network_states=True, restrict_to_params=None, overwrite_existing=False, force_recomputation=False, **measure_params):
    """Calls :func:`measures.common_measures.generic_measure` after some pre-processing."""
    
    if save_results_to is None:
        save_results_to = os.path.join(_DEFAULT_SAVING_FOLDER, MEASURE_TITLE)
    
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
    
    sim_params = get_sim_params()
    
    net_params = NetworkParams(dummypol_deltar=2, 
                               rtprop_policy_max_routes = 3,
                               rtprop_policy_p_reaccept = 0.7,
                               rtprop_policy_p_replace = 0.5,
                               batching_t_interval = 1*60*1000,
                               batching_nmin = 2,
                               batching_f = 0.8,
                               dummypol_fdum = 0.8)

    return generic_measure(MEASURE_TITLE, save_results_to, 
                           net_params_combinations, 
                           metrics_computer_callback=compute_metrics_one_network_run,
                           sim_params=sim_params,
                           net_params=net_params,
                           save_network_states=save_network_states,
                           restrict_to_params=restrict_to_params,
                           overwrite_existing=overwrite_existing,
                           force_recomputation=force_recomputation)

def get_sim_params():
    sim_params = SimulationParams()
    sim_params.logging_level = logging.WARNING  # Display only Warnings or errors
    sim_params.print_nodes_tables = False  # In general, no one will be there to see the stdout or graphs, so don't draw them
    sim_params.draw_topology_graph = False
    sim_params.draw_routes = None
    sim_params.time_of_simulation = 0 # Go until the end of the simulation
    sim_params.automatic_oriented_comm_phase = True # Oriented communication will be measured, so yes !
    sim_params.oriented_communication_sessions = None
    
    # Put all the logging to False by default
    sim_params.log_ocom_latency = True
    sim_params.log_end_topo_diss = True
    sim_params.log_end_ocom_phase = True
    sim_params.log_and_store_all_real_msgs = False
    sim_params.log_dummy_link_msgs = True
    sim_params.log_real_link_msgs = True      
    sim_params.log_real_e2e_msgs = False          
    sim_params.log_histogram_real_msgs_per_round = False
    sim_params.log_traffic_rates_equilibrium = True
    sim_params.log_real_msgs_waiting_time = False
    sim_params.log_histogram_real_msgs_waiting_time = True
    sim_params.log_e2e_dummies = False
    sim_params.log_frequency_batch_intervention = True
    sim_params.log_sent_link_msgs_per_round = True
    sim_params.log_route_props = False
    sim_params.log_rt_props_latency = False
    sim_params.log_ocom_routes = False
    
    return sim_params

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


    if aggregated_metrics is None:
        aggregated_metrics = {}
        aggregated_metrics[_TOPO_DISS_TIME] = RunningStat()
        aggregated_metrics[_OCOM_PHASE_TIME] = RunningStat()
        aggregated_metrics[_OCOM_MSG_TIME] = RunningStat()
        aggregated_metrics[_OCOM_MSG_TIME_WITHOUT_INIT] = RunningStat()
        aggregated_metrics[_HIST_WAITING_TIME_REAL_MSGS] = defaultdict(lambda: RunningStat())
        aggregated_metrics[_RATIO_REAL_OVER_DUMMY_MSGS] = RunningStat()
        aggregated_metrics[_RATIO_E2E_DUMMY] = RunningStat()
        aggregated_metrics[_EQUILIBRIUM_PER_ROUND] = {'equilibrium': RunningStat(),
                                                      'lbound': RunningStat(),
                                                      'ubound': RunningStat()}
        aggregated_metrics[_RATIO_DUMMY_BCAST] = RunningStat()
        aggregated_metrics[_HIST_NB_LINK_MSGS_SENT_PER_ROUND] = defaultdict(lambda: RunningStat())
        
    # Not needed here
    # stats_helper = AdvancedNetworkStatsHelper(net_mngr)
    
    this_iteration_metrics = {}
    
    #Time for topo diss to complete in number of rounds
    this_iteration_metrics[_TOPO_DISS_TIME] = net_mngr.net_stats.end_topo_diss
    aggregated_metrics[_TOPO_DISS_TIME].push_value(this_iteration_metrics[_TOPO_DISS_TIME])
    
    # Time for ocom to complete
    this_iteration_metrics[_OCOM_PHASE_TIME] = net_mngr.net_stats.end_ocom_phase-net_mngr.net_stats.end_topo_diss 
    aggregated_metrics[_OCOM_PHASE_TIME].push_value(this_iteration_metrics[_OCOM_PHASE_TIME])
    
    # Time needed for 1 ocom payload message to be delivered, in average
    nb_payload_msgs_in_each_ocom = net_mngr.sim_params.oriented_communication_sessions[0][1]
    this_iteration_metrics[_OCOM_MSG_TIME] = net_mngr.net_stats.ocom_latency.mean/nb_payload_msgs_in_each_ocom
    aggregated_metrics[_OCOM_MSG_TIME].push_value(this_iteration_metrics[_OCOM_MSG_TIME])
    
    # The same, but without taking ocom init into account
    this_iteration_metrics[_OCOM_MSG_TIME_WITHOUT_INIT] = net_mngr.net_stats.ocom_latency_after_init.mean/nb_payload_msgs_in_each_ocom
    aggregated_metrics[_OCOM_MSG_TIME_WITHOUT_INIT].push_value(this_iteration_metrics[_OCOM_MSG_TIME_WITHOUT_INIT])
    
    # The time a real message spends waiting in pools before being forwarded/sent
    this_iteration_metrics[_HIST_WAITING_TIME_REAL_MSGS] = net_mngr.net_stats.histogram_real_msg_waiting_time
    for waiting_time, n in this_iteration_metrics[_HIST_WAITING_TIME_REAL_MSGS].items():
        aggregated_metrics[_HIST_WAITING_TIME_REAL_MSGS][waiting_time].push_value(n)
    
    # The number of real messages compared to dummy ones
    nb_sent_real_link_msgs = sum(v.total for v in net_mngr.net_stats.nb_sent_real_msgs_per_node.values())
    nb_sent_dummy_link_msgs = sum(v.total for v in net_mngr.net_stats.nb_sent_dummy_link_msgs_per_node.values())
    this_iteration_metrics[_RATIO_REAL_OVER_DUMMY_MSGS] = nb_sent_real_link_msgs/nb_sent_dummy_link_msgs
    aggregated_metrics[_RATIO_REAL_OVER_DUMMY_MSGS].push_value(this_iteration_metrics[_RATIO_REAL_OVER_DUMMY_MSGS])
    
    # The traffic rate equilibrium
    this_iteration_metrics[_EQUILIBRIUM_PER_ROUND] = {'equilibrium': net_mngr.net_stats.traffic_rate_status['equilibrium'].mean,
                                                       'lbound': net_mngr.net_stats.traffic_rate_status['lbound'].mean,
                                                       'ubound': net_mngr.net_stats.traffic_rate_status['ubound'].mean} 
    for k, v in this_iteration_metrics[_EQUILIBRIUM_PER_ROUND].items():
        aggregated_metrics[_EQUILIBRIUM_PER_ROUND][k].push_value(v)
    
    # The number of sent e2e dummy per rounds
    this_iteration_metrics[_RATIO_E2E_DUMMY] = net_mngr.net_stats.nb_batch_intervention_add_e2e_dummy.mean
    aggregated_metrics[_RATIO_E2E_DUMMY].push_value(this_iteration_metrics[_RATIO_E2E_DUMMY])
    
    
    # The number of sent e2e dummy per rounds
    this_iteration_metrics[_RATIO_DUMMY_BCAST] = net_mngr.net_stats.nb_batch_intervention_default_dummy_bcast.mean
    aggregated_metrics[_RATIO_DUMMY_BCAST].push_value(this_iteration_metrics[_RATIO_DUMMY_BCAST])
     
    # The number of link messages sent per neighbor at each round (real and dummies confounded)
    this_iteration_metrics[_HIST_NB_LINK_MSGS_SENT_PER_ROUND] = net_mngr.net_stats.histogram_nb_sent_link_msgs_per_round_per_neighbor
    for nb_msgs, occurences in this_iteration_metrics[_HIST_NB_LINK_MSGS_SENT_PER_ROUND].items():
        aggregated_metrics[_HIST_NB_LINK_MSGS_SENT_PER_ROUND][nb_msgs].push_value(occurences)
     
    return aggregated_metrics, this_iteration_metrics



# #################################
# Plotting functions ############## 
# #################################



@plotfunc(MEASURE_PARAMS)    
def plot_TOPO_DISS_TIME(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'topo_diss_latency/')
    y_stat = (_TOPO_DISS_TIME, 'topo.diss. latency', 'topo_diss_latency')
    
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)


@plotfunc(MEASURE_PARAMS)
def plot_OCOM_PHASE_TIME(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ocom_phase_latency/')
    y_stat = (_OCOM_PHASE_TIME, 'ocom. phase latency', 'ocom_phase_latency')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)


@plotfunc(MEASURE_PARAMS)
def plot_OCOM_MSG_TIME(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ocom_one_msg_latency_with_init/')
    y_stat = (_OCOM_MSG_TIME, 'ocom. one msg latency w/ init', 'ocom_one_msg_latency_with_init')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

@plotfunc(MEASURE_PARAMS)
def plot_OCOM_MSG_TIME_WITHOUT_INIT(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ocom_one_msg_latency_no_init/')
    y_stat = (_OCOM_MSG_TIME_WITHOUT_INIT, 'ocom. one msg latency w/o init', 'ocom_one_msg_latency_no_init')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

@plotfunc(MEASURE_PARAMS)
def plot_RATIO_REAL_OVER_DUMMY_MSGS(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ratio_reals_over_dummies/')
    y_stat = (_RATIO_REAL_OVER_DUMMY_MSGS, '# reals / # dummies', 'ratio_reals_over_dummies')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

@plotfunc(MEASURE_PARAMS)
def plot_RATIO_E2E_DUMMY(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ratio_round_with_e2edum/')
    y_stat = (_RATIO_E2E_DUMMY, '# rounds w/ e2edum / # rounds', 'ratio_round_with_e2edum')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

@plotfunc(MEASURE_PARAMS)
def plot_RATIO_DUMMY_BCAST(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ratio_round_with_dumbcast/')
    y_stat = (_RATIO_DUMMY_BCAST, '# rounds w/ dum. bcast. / # rounds', 'ratio_round_with_dumbcast')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)


def plot_HIST_WAITING_TIME_REAL_MSGS(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'real_msgs_waiting_time_hist/')
    y_stat = (_HIST_WAITING_TIME_REAL_MSGS, '%', 'real_msgs_waiting_time_hist')
    
    plot_histogram(experiments_results, MEASURE_PARAMS, y_stat=y_stat, max_bars=21, print_yerr=True, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)
    
    

@plotfunc(MEASURE_PARAMS)
def plot_MEAN_WAITING_TIME_REAL_MSGS(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'real_msgs_waiting_time_mean/')
    y_stat = (_MEAN_WAITING_TIME_REAL_MSGS, '# rounds', 'real_msgs_waiting_time_mean')
    
    # The man waiting time of real messages is not computed during simulation.
    # But it can be deduced from the histogram of real message waiting time.
    # This data is processed here, thus computing the mean waiting time of real
    # messages
    for r in experiments_results:
        if _MEAN_WAITING_TIME_REAL_MSGS in r['results']['overall']:
            break
        r['results']['overall'][_MEAN_WAITING_TIME_REAL_MSGS] = RunningStat()
        for n_rounds, n_occurences in r['results']['overall'][_HIST_WAITING_TIME_REAL_MSGS].items():
            for _ in range(round(n_occurences.total)): r['results']['overall'][_MEAN_WAITING_TIME_REAL_MSGS].push_value(n_rounds)
#         r['results']['overall'][_MEAN_WAITING_TIME_REAL_MSGS] /= sum(n.total for n in r['results']['overall'][_HIST_WAITING_TIME_REAL_MSGS].values())
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)
    
    
    

@plotfunc(MEASURE_PARAMS)
def plot_EQUILIBRIUM_PER_ROUND(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'equilibirum_per_round/')
    y_stat = (_EQUILIBRIUM_PER_ROUND, 'avg. equilibrium of each node at each round', 'equilibirum_per_round')
    
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, several_curves=['lbound', 'equilibrium', 'ubound'], restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

def plot_HIST_NB_LINK_MSGS_SENT_PER_ROUND(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'link_msgs_per_round_per_neighbor/')
    y_stat = (_HIST_NB_LINK_MSGS_SENT_PER_ROUND, '%', 'link_msgs_per_round_per_neighbor')
    
    plot_histogram(experiments_results, MEASURE_PARAMS, y_stat=y_stat, print_yerr=True, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)




if __name__ == '__main__':
    folder = os.path.join(_DEFAULT_SAVING_FOLDER, 'measure_'+MEASURE_TITLE)
    
    do_measures = True
    do_plot = True
    
    if do_measures:
        measure_general_efficiency(nb_nodes=10,
                                    nb_iters=1,
                                    nb_iters_same_topo_graph=2,
                                    batching_nmin=[5],
                                    batching_f=[0.5],
                                    dummypol_fdum=[0.8],
                                    dummypol_deltar=[8],
                                    save_results_to=folder,
                                    overwrite_existing=True)
    
    if do_plot:
        results = load_measures_results(folder, MEASURE_TITLE+".pickle")
        folder_graphs = os.path.join(folder, "graphs")
        
        print("\nPlotting graphs...")
        for stat in  MEASURE_STATS:
            print("Plotting metric {}.{}".format(MEASURE_TITLE, stat))
            sys.modules[__name__].__dict__["plot"+stat](results, x_param=('nb_nodes', '# nodes'), save_to=folder_graphs, overwrite_existing=True)

    
    