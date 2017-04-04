# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html



'''
Example module to makes simulations and measures on the APART protocol. This
module runs network simulation and makes measures on the route proposal
mechanisms.

More specifically, this module measures the following metrics (non exhaustive list):

* Number of routes between pairs of nodes
* Ratio route length over length of corresponding shortest path
* Probability distribution over the route length (all pairs of node taken together)
  (for better privacy, this distrib should have high entropy)
* Similarity measure (jackard) between the sets of routes constructed between runs 
  of the network *over the same topology graph* (similarity should be low, for better
  privacy)
          
All these metrics are computed with a combination of different parameters range. Namely,

* Number of nodes 
* Rt prop policy, max routes 
* Rt prop policy, prob of reaccepting 
* Rt prop policy, prob of replacing a route 
* Rt prop policy, prob of re proposing 
    
The module contains functions for running the measures, and plotting graphs from
network runs saved on disk.
'''

from collections import defaultdict, OrderedDict, Counter
import copy
import functools
import itertools
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

from common.custom_logging import *
from apart.core.network import NetworkParams, NetworkManager, Network
from apart.core.protocol_constants import RtPolicyReason
from apart.simulation import SimulationParams
from common.utilities import print_recursive_structure, make_hashable
from measures.advanced_stats_helper import AdvancedNetworkStatsHelper
from measures.common_measures import MeasureException, generic_measure, \
    merge_measures_outputs, load_measures_results, \
    merge_measures_outputs_with_prefix, make_params_combinations,\
    preprocess_measure_input_params
from measures.plotting import plotfunc, plot_simple_metric, \
    plot_histogram, process_plot_function_inputs, do_plot_simple_metric
from measures.network_statistics import RegularStat, RunningStat
import dill as pickle
import matplotlib.pyplot as plt


_RATIO_ACCEPTED_PROPOSED_ROUTES = "#accepted/#refused routes"
"""Nb accepted over nb refused routes. Averaged over all nodes of a network, and averaged over several network runs. Stdev represents the stdev between network runs""" 
_HIST_ACCEPT_REFUSE_ROUTES = "Histogram of route refused/accepted routes"
_TOPO_DISS_TIME = "Number of rounds for topology dissemination (routepropmechanism)"
_LATENCY_RT_PROPS = "Number of rounds to complete a route proposal"
_NB_ROUTES_BETWEEN_PAIRS = "Number of routes between pairs of nodes"
_RATIO_ROUTES_LENGTH_SHORTEST_PATH = "Ratio of the routes length versus the their corresponding shortest path"
_HIST_ROUTES_LENGTH = "Histogram of routes lengths"
_ROUTES_JACCARD_DISTANCE = "Similarity measure of routes for different runs on the same topology graph"

_stat_to_proper_human_readable = {_RATIO_ACCEPTED_PROPOSED_ROUTES: "Ratio of number of accepted routes overnumber of proposed ones",
                                  _HIST_ACCEPT_REFUSE_ROUTES: "Histogram of reasons nodes accepted/refused routes",
                                  _TOPO_DISS_TIME: "Number of rounds for topology dissemination to complete",
                                  _LATENCY_RT_PROPS: "Number of rounds to complete a route proposal",
                                  _NB_ROUTES_BETWEEN_PAIRS: "Number of routes between pairs of nodes",
                                  _RATIO_ROUTES_LENGTH_SHORTEST_PATH: "Ratio of the routes length versus the their corresponding shortest path",
                                  _HIST_ROUTES_LENGTH: "Histogram of routes lengths",
                                  _ROUTES_JACCARD_DISTANCE: "Similarity measure of routes for different runs on the same topology graph"
                                  }

MEASURE_STATS = ['_HIST_ROUTES_LENGTH', 
                  '_HIST_ACCEPT_REFUSE_ROUTES', 
                  '_LATENCY_RT_PROPS', 
                  '_NB_ROUTES_BETWEEN_PAIRS', 
                  '_ROUTES_JACCARD_DISTANCE', 
                  '_RATIO_ROUTES_LENGTH_SHORTEST_PATH', 
                  '_TOPO_DISS_TIME', 
                  '_RATIO_ACCEPTED_PROPOSED_ROUTES']

# Title of the measures performed in this module
MEASURE_TITLE = 'RouteProposalMechanism'

# Network parameters that vary for the measures uin this module
_MEASURE_PARAMS_DEFAULT = OrderedDict([('nb_nodes', (int, 30)), 
                                       ('rtprop_policy_max_routes', (int, 3)),
                                       ('rtprop_policy_p_reaccept', (float, 0.7)),
                                       ('rtprop_policy_p_replace', (float, 0.5))
                                       ])
MEASURE_PARAMS = list(_MEASURE_PARAMS_DEFAULT.keys())

# Default folder where to save the 
_DEFAULT_SAVING_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Measures/'))
if not os.path.exists(_DEFAULT_SAVING_FOLDER):
    os.makedirs(_DEFAULT_SAVING_FOLDER, mode=0o775)

def measure_route_prop_mechanism(save_results_to=None, save_network_states=True, restrict_to_params=None, overwrite_existing=False, force_recomputation=False, **measure_params):
    """Calls :func:`measures.common_measures.generic_measure` after some pre-processing."""
    if save_results_to is None:
        save_results_to = os.path.join(_DEFAULT_SAVING_FOLDER, MEASURE_TITLE)
        
    
    preprocess_measure_input_params(_MEASURE_PARAMS_DEFAULT, measure_params)
    
    nb_iters = measure_params.pop('nb_iters')
    nb_iters_same_topo_graph= measure_params.pop('nb_iters_same_topo_graph')
    net_params_combinations = make_params_combinations(nb_iters, nb_iters_same_topo_graph, MEASURE_PARAMS,  measure_params)

    sim_params = get_sim_params()
    
    net_params = NetworkParams(dummypol_deltar=2, 
                               rtprop_policy_max_routes = 3,
                               rtprop_policy_p_reaccept = 0.7,
                               rtprop_policy_p_replace = 0.5,
                               batching_t_interval = 1*60*1000,
                               batching_nmin = 2,
                               batching_f = 0.5,
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
    sim_params.automatic_oriented_comm_phase = False # We only measure topo diss, so no oriented communications are neede
    sim_params.oriented_communication_sessions = None # Let the communications be random
    
    # Put all the logging to False by default
    sim_params.log_ocom_latency = False
    sim_params.log_end_topo_diss = True
    sim_params.log_end_ocom_phase = False
    sim_params.log_and_store_all_real_msgs = False
    sim_params.log_dummy_link_msgs = False
    sim_params.log_real_link_msgs = False   
    sim_params.log_real_e2e_msgs = False             
    sim_params.log_histogram_real_msgs_per_round = False
    sim_params.log_traffic_rates_equilibrium = False
    sim_params.log_real_msgs_waiting_time = False
    sim_params.log_histogram_real_msgs_waiting_time = False
    sim_params.log_e2e_dummies = False
    sim_params.log_frequency_batch_intervention = False
    sim_params.log_sent_link_msgs_per_round = False
    sim_params.log_route_props = True
    sim_params.log_rt_props_latency = True
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
        # This is the first iteration of a parameters combination. 
        aggregated_metrics = {}
        aggregated_metrics[_RATIO_ACCEPTED_PROPOSED_ROUTES] = RunningStat()
        aggregated_metrics[_HIST_ACCEPT_REFUSE_ROUTES] = dict((reason, RunningStat()) for reason in RtPolicyReason)
        aggregated_metrics[_TOPO_DISS_TIME] = RunningStat()
        aggregated_metrics[_LATENCY_RT_PROPS] = {'mean': RunningStat(), 'min': RunningStat(), 'max': RunningStat()}
        aggregated_metrics[_NB_ROUTES_BETWEEN_PAIRS] = {'mean': RunningStat(), 'min': RunningStat(), 'max': RunningStat()}
        aggregated_metrics[_RATIO_ROUTES_LENGTH_SHORTEST_PATH] = RunningStat()
        aggregated_metrics[_HIST_ROUTES_LENGTH] = defaultdict(lambda: RunningStat())
        aggregated_metrics[_ROUTES_JACCARD_DISTANCE] = None
        
    
    stats_helper = AdvancedNetworkStatsHelper(net_mngr)
    
    this_iteration_metrics = {}
    
    # Ratio of accepted over proposed routes
    total_accepted_rtprops = stats_helper.nb_rt_props['accepted'].total
    total_proposed_rtprops = stats_helper.nb_rt_props['proposed'].total
    
    this_iteration_metrics[_RATIO_ACCEPTED_PROPOSED_ROUTES] = total_accepted_rtprops/total_proposed_rtprops
    aggregated_metrics[_RATIO_ACCEPTED_PROPOSED_ROUTES].push_value(this_iteration_metrics[_RATIO_ACCEPTED_PROPOSED_ROUTES]) 
     

    # Histogram of accepted/refused routes, with the reason to accept/refuse
    this_iteration_metrics[_HIST_ACCEPT_REFUSE_ROUTES] = {}
    for reason in RtPolicyReason:
        this_iteration_metrics[_HIST_ACCEPT_REFUSE_ROUTES][reason] = stats_helper.nb_rt_props[reason].total
        aggregated_metrics[_HIST_ACCEPT_REFUSE_ROUTES][reason].push_value(stats_helper.nb_rt_props[reason].total)
    
    #Time for topo diss to complete in number of rounds
    this_iteration_metrics[_TOPO_DISS_TIME] = net_mngr.net_stats.end_topo_diss
    aggregated_metrics[_TOPO_DISS_TIME].push_value(net_mngr.net_stats.end_topo_diss)
    
    # Time taken by route proposal (in average, min, and max) in number of rounds,
    rt_props_latency = net_mngr.net_stats.rt_props_latency
    this_iteration_metrics[_LATENCY_RT_PROPS] = {'mean': rt_props_latency.mean,
                                  'min': rt_props_latency.min,
                                  'max': rt_props_latency.max}
    aggregated_metrics[_LATENCY_RT_PROPS]['mean'].push_value(rt_props_latency.mean)
    aggregated_metrics[_LATENCY_RT_PROPS]['min'].push_value(rt_props_latency.min)
    aggregated_metrics[_LATENCY_RT_PROPS]['max'].push_value(rt_props_latency.max)
    
    # Number of routes between pairs of nodes
    nb_routes_btw_pairs = stats_helper.nb_routes_btw_pairs
    this_iteration_metrics[_NB_ROUTES_BETWEEN_PAIRS] = {'mean': nb_routes_btw_pairs.mean,
                              'min': nb_routes_btw_pairs.min,
                              'max': nb_routes_btw_pairs.max}
    aggregated_metrics[_NB_ROUTES_BETWEEN_PAIRS]['mean'].push_value(nb_routes_btw_pairs.mean)
    aggregated_metrics[_NB_ROUTES_BETWEEN_PAIRS]['min'].push_value(nb_routes_btw_pairs.min)
    aggregated_metrics[_NB_ROUTES_BETWEEN_PAIRS]['max'].push_value(nb_routes_btw_pairs.max)

    # Ratio route length over length of corresponding shortest path
    this_iteration_metrics[_RATIO_ROUTES_LENGTH_SHORTEST_PATH] = stats_helper.routes_length_vs_shortest_path().mean
    aggregated_metrics[_RATIO_ROUTES_LENGTH_SHORTEST_PATH].push_value(this_iteration_metrics[_RATIO_ROUTES_LENGTH_SHORTEST_PATH])
        
    #Probability distribution over the route length (all pairs of node taken together)
    this_iteration_metrics[_HIST_ROUTES_LENGTH] = stats_helper.histogram_routes_length()
    for l, nb in this_iteration_metrics[_HIST_ROUTES_LENGTH].items():
        aggregated_metrics[_HIST_ROUTES_LENGTH][l].push_value(nb)
    
    # Similarity measure (jackard) between the sets of routes constructed
    # between runs of the network *over the same topology graph*
    # This special metric asks that topo graph be the same over several runs. 
    
    # First iteration and second iteration
    this_iteration_metrics[_ROUTES_JACCARD_DISTANCE] = stats_helper.complete_routes_descriptions
    
    
    if aggregated_metrics[_ROUTES_JACCARD_DISTANCE] is None:
        # First iteration
        aggregated_metrics[_ROUTES_JACCARD_DISTANCE] = {'jaccard_distances': [],
                                                        'nb_nodes': net_mngr.network.nb_nodes,
                                                        'graph': net_mngr.network.topology_graph,
                                                        'routes_first_iteration': this_iteration_metrics[_ROUTES_JACCARD_DISTANCE]}
    else:
        # Second iteration only : the actual computation
        # TODO : checks for debug, to remove
        if 'nb_nodes' not in aggregated_metrics[_ROUTES_JACCARD_DISTANCE]:
            logging.error("Error in measure of jaccard distance of routes: more than two iterations for the same graph")
            return aggregated_metrics, this_iteration_metrics
        if net_mngr.network.nb_nodes != aggregated_metrics[_ROUTES_JACCARD_DISTANCE]['nb_nodes']:
            logging.error("Error in measure of jaccard distance of routes: number of nodes do not match between the first and second iteration")
            return aggregated_metrics, this_iteration_metrics
        G1 = aggregated_metrics[_ROUTES_JACCARD_DISTANCE]['graph']
        G2 = net_mngr.network.topology_graph
        if set(G1.nodes()) != set(G2.nodes()) or set(G1.edges()) != set(G2.edges()):
            logging.error("Error in measure of jaccard distance of routes: topology graphs do not match between the first and second iteration")
            return aggregated_metrics, this_iteration_metrics
       
        routes_first_run = aggregated_metrics[_ROUTES_JACCARD_DISTANCE]['routes_first_iteration']
        routes_this_run = this_iteration_metrics[_ROUTES_JACCARD_DISTANCE]
        
        counter1 = Counter(map(lambda route: tuple(n[0] for n in route), itertools.chain(*(routes_first_run.values()))))
        counter2 = Counter(map(lambda route: tuple(n[0] for n in route), itertools.chain(*(routes_this_run.values()))))
        
        union_of_routes = counter1 | counter2
        inter_of_routes = counter1 & counter2
        
        len_union_of_routes = sum(union_of_routes.values()) 
        len_inter_of_routes = sum(inter_of_routes.values())
        
        jaccard_distance = (len_union_of_routes-len_inter_of_routes)/len_union_of_routes
        
        aggregated_metrics[_ROUTES_JACCARD_DISTANCE]['jaccard_distances'].append(jaccard_distance)
        

    return aggregated_metrics, this_iteration_metrics



# #################################
# Plotting functions ############## 
# #################################


@plotfunc(MEASURE_PARAMS)
def plot_RATIO_ACCEPTED_PROPOSED_ROUTES(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ratio_accepted_proposed_routes/')
    y_stat = (_RATIO_ACCEPTED_PROPOSED_ROUTES, '#accepted/#proposed', 'ratio_accepted_proposed_routes')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)
            
@plotfunc(MEASURE_PARAMS)    
def plot_TOPO_DISS_TIME(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'topo_diss_latency/')
    y_stat = (_TOPO_DISS_TIME, 'topo.diss. latency', 'topo_diss_latency')
    
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

@plotfunc(MEASURE_PARAMS)
def plot_RATIO_ROUTES_LENGTH_SHORTEST_PATH(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'ratio_routes_to_short_path/')
    y_stat = (_RATIO_ROUTES_LENGTH_SHORTEST_PATH, '|routes|/|shortest routes|', 'ratio_routes_to_short_path')
     
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

@plotfunc(MEASURE_PARAMS)  
def plot_NB_ROUTES_BETWEEN_PAIRS(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'routes_btw_pairs/')
    y_stat = (_NB_ROUTES_BETWEEN_PAIRS, '# routes btw pairs of node', 'routes_btw_pairs')
     

    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, several_curves=['mean', 'min', 'max'], restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)
    
@plotfunc(MEASURE_PARAMS)
def plot_LATENCY_RT_PROPS(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'latency_rt_prop/')
    y_stat = (_LATENCY_RT_PROPS, 'avg. latency of a rt prop', 'rt_prop_latency')
  
    plot_simple_metric(experiments_results, MEASURE_PARAMS, y_stat=y_stat, x_param=x_param, several_curves=['mean', 'min', 'max'], restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)


def plot_HIST_ACCEPT_REFUSE_ROUTES(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'hist_accept_refuse_routes/')
    y_stat = (_HIST_ACCEPT_REFUSE_ROUTES, '%', 'hist_accept_refuse_routes')
     
    plot_histogram(experiments_results, MEASURE_PARAMS, y_stat=y_stat, label_rotation=90, xticks=[r.name for r in sorted(RtPolicyReason)], print_yerr=True, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)
     
def plot_HIST_ROUTES_LENGTH(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    if save_to is not None:
        save_to = os.path.join(save_to, 'hist_routes_lengths/')
    y_stat = (_HIST_ROUTES_LENGTH, '%', 'hist_routes_lengths')
    
    plot_histogram(experiments_results, MEASURE_PARAMS, y_stat=y_stat, print_yerr=True, restrict_to_other_params=restrict_to_other_params, save_to=save_to, overwrite_existing=overwrite_existing)

    
    
@plotfunc(MEASURE_PARAMS)
def plot_ROUTES_JACCARD_DISTANCE(experiments_results, x_param=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    """Custom plotting function.
    
    The jaccard distance (or *route dissimilarity*) is a more complex metric,
    and can not be simply plotted with the plot_simple_metric or plot_histogram
    function. 
    """
    if save_to is not None:
        save_to = os.path.join(save_to, 'routes_disimilarity/')
    y_stat = (_ROUTES_JACCARD_DISTANCE, 'D_jaccard btw 2 runs', 'routes_dissimilarity')
    
    
    x_param, y_stat, restrict_to_other_params = process_plot_function_inputs(x_param, y_stat, restrict_to_other_params)
    x_param, x_param_label, x_param_file_name = x_param
    y_stat, y_stat_label, y_stat_file_name = y_stat
    
    if save_to is not None:
        save_to = os.path.join(save_to, 'vs_{}'.format(x_param_file_name))
    
    
    graph_routes_distance_vs = {'x_label': x_param_label, 'y_label': y_stat_label, 
                                  'linestyle': '-'}
    all_possible_graphs_values = defaultdict(lambda: defaultdict(lambda: RunningStat()))
    for experiment in experiments_results:
        # Keep the experiment results only if they fit into the asked other parameters
        skip = False
        for p, v_list in restrict_to_other_params.items():
            if  experiment['measure_params'][p] not in v_list:
                skip = True
                break
        
        if skip:
            continue
        
        x_value = experiment['measure_params'][x_param]
        other_measure_params = list(sorted(((p, v) for p, v in experiment['measure_params'].items() if p != x_param and p in MEASURE_PARAMS), 
                                            key=lambda x: MEASURE_PARAMS.index(x[0])))
        other_measure_params.append(('nb_iters', experiment['measure_params']['nb_iters']))
        other_measure_params = make_hashable(other_measure_params)
        
        # DEBUG : compatibility with previous way of storing stats
        if isinstance( experiment['results']['overall'][y_stat], dict):
            for jaccard_distance in experiment['results']['overall'][y_stat]['jaccard_distances']:
                all_possible_graphs_values[other_measure_params][x_value].push_value(jaccard_distance)
        else:
            all_possible_graphs_values[other_measure_params][x_value].push_value(experiment['results']['overall'][y_stat])
    
    # Each graph must show a statistic (in y axis) according to a network
    # parameter (in x axis), where **only** this network parameter varies. Thus,
    # we make one graph per set of other network parameters
    all_possible_graphs_values = OrderedDict(sorted(all_possible_graphs_values.items(), key=lambda x: x[0]))
    i = 0
    for other_measure_params, graph_values in all_possible_graphs_values.items():
        sorted_zipped_data = sorted(((k, v.mean, v.stdev) for k, v in graph_values.items())
                                    , key=lambda x: x[0])
    
        x_values, y_values, stdevs = map(list, zip(*sorted_zipped_data))
        
        graph_routes_distance_vs['title'] = ",\n".join(map(lambda v: str(v[0])+": "+str(v[1]), other_measure_params))
        if save_to:
            graph_routes_distance_vs['file_name'] = "{}_vs_{}".format(y_stat_file_name, x_param_file_name).lower() + "_"+str(i)
            i+=1
        do_plot_simple_metric(x_values, y_values, stdevs, graph_routes_distance_vs, save_to=save_to, overwrite_existing=overwrite_existing)
        

if __name__ == '__main__':
    folder = os.path.join(_DEFAULT_SAVING_FOLDER, 'measure_'+MEASURE_TITLE)
    
    do_measures = True
    do_plot = True
    
    if do_measures:
        measure_route_prop_mechanism(nb_nodes=10,
                                    nb_iters=2,
                                    nb_iters_same_topo_graph=2,
                                    rtprop_policy_max_routes=3,
                                    rtprop_policy_p_reaccept=0.7,
                                    rtprop_policy_p_replace=0.25,
                                    save_results_to=folder,
                                    overwrite_existing=True)
    
    if do_plot:
        results = load_measures_results(folder, MEASURE_TITLE+".pickle")
        folder_graphs = os.path.join(folder, "graphs")
        
        print("\nPlotting graphs...")
        for stat in MEASURE_STATS:
            print("Plotting metric {}.{}".format(MEASURE_TITLE, stat))
            sys.modules[__name__].__dict__["plot"+stat](results, x_param=('nb_nodes', '# nodes'), save_to=folder_graphs, overwrite_existing=True)
            