# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
Example module to makes simulations and measures on the APART protocol. This
module runs network simulation and makes measures on the privacy provided by the
protocol.

More specifically, this module measures the receiver anonymity. This metric is
computed in function of the number of nodes in the network, and of the
corruption ratio.
'''

from collections import defaultdict, OrderedDict, Counter
import copy
import functools
import itertools
import math
import numpy
import os
import random
from scipy.optimize import curve_fit
import statistics
import sys


sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))


from apart.core.network import NetworkParams, NetworkManager, Network, \
    SimulationException
from apart.core.protocol_constants import RtPolicyReason
from apart.core.tables import RoutingTable
from apart.simulation import SimulationParams
from common.custom_logging import *
from common.utilities import print_recursive_structure, range1, \
    make_hashable, comb
import dill as pickle
import matplotlib.pyplot as plt
from measures.advanced_stats_helper import AdvancedNetworkStatsHelper
from measures.common_measures import MeasureException, generic_measure, \
    save_measures_results, load_measures_results, \
    preprocess_measure_input_params, make_params_combinations, \
    merge_measures_outputs_with_prefix, merge_measures_outputs
from measures.file_handling import save_graphs_simulation_results
from measures.network_statistics import RegularStat, RunningStat
from measures.plotting import do_plot_simple_metric,process_plot_function_inputs
import networkx as nx




_RECEIVER_ANONYMITY = "Overall receiver anonymity"
_RECEIVER_ANONYMITY_HIST_NORMALISED_ENTROPY = "receiver anonymity: normalised entropy"
_RECEIVER_ANONYMITY_HIST_PROB_MOST_LIKELY = "receiver anonymity: probability of most likely receiver in the prob distrib"
_RECEIVER_ANONYMITY_FREQ_GOOD_GUESS = "receiver anonymity: number of times or frequency at which the adversary would guess right"

_HIST_SIMPLE_ROUTE_LENGTH = "Dict[k] = v, for k = route length,v = nb routes with this length"
_CORRUPTION_STATES = "Corrupted and honest nodes in the network"
_TOPOLOGY_GRAPH = "Topology graph of the network"
_FULL_OCOM_ROUTES = "All ocom routes in a network run"


MEASURE_STATS = ['_RECEIVER_ANONYMITY']

# Title of the measures performed in this module
MEASURE_TITLE = 'Privacy'

# Network parameters that vary for the measures uin this module
_MEASURE_PARAMS_DEFAULT = OrderedDict([('nb_nodes', (int, 30)), 
                                       ('corruption_ratio', (float, 0.33))
                                       ])
MEASURE_PARAMS = list(_MEASURE_PARAMS_DEFAULT.keys())

# Default folder where to save the 
_DEFAULT_SAVING_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Measures/'))
if not os.path.exists(_DEFAULT_SAVING_FOLDER):
    os.makedirs(_DEFAULT_SAVING_FOLDER, mode=0o775)

def measure_privacy(save_results_to=None, save_network_states=True, restrict_to_params=None, overwrite_existing=False, force_recomputation=False, **measure_params):
    
    if save_results_to is None:
        save_results_to = os.path.join(_DEFAULT_SAVING_FOLDER, MEASURE_TITLE)
    
    preprocess_measure_input_params(_MEASURE_PARAMS_DEFAULT, measure_params)
    
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

def recompute_measure_privacy(from_measures_folder, save_results_to=None, overwrite_existing=False):
    if save_results_to is None:
        save_results_to = os.path.join(_DEFAULT_SAVING_FOLDER, MEASURE_TITLE)
        
    return recompute_generic_measure(from_measures_folder, MEASURE_TITLE, 
                                     save_results_to=save_results_to, 
                                     metrics_computer_callback=compute_metrics_one_network_run,
                                     overwrite_existing=overwrite_existing)

def get_sim_params():
    sim_params = SimulationParams()
    sim_params.logging_level = logging.WARNING  # Display only Warnings or errors
    sim_params.print_nodes_tables = False  # In general, no one will be there to see the stdout or graphs, so don't draw them
    sim_params.draw_topology_graph = False
    sim_params.draw_routes = None
    sim_params.time_of_simulation = 0 # Go until the end of the simulation
    sim_params.automatic_oriented_comm_phase = True # We only measure topo diss, so no oriented communications are neede
    sim_params.oriented_communication_sessions = defaultdict(lambda: (5, 1)) # Each node sends 1 message to 10 different receivers
    
    # Put all the logging to False by default
    sim_params.log_ocom_latency = False
    sim_params.log_end_topo_diss = False
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
    sim_params.log_route_props = False
    sim_params.log_sent_link_msgs_per_round = False
    sim_params.log_rt_props_latency = False
    sim_params.log_ocom_routes = True
    
    return sim_params
def compute_metrics_one_network_run(aggregated_metrics, net_mngr):
    """Callback function for this measure module.
    
    Computes the statistics on the network state at the end of a run, and aggregates the statistics over several network runs.
    
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
        aggregated_metrics[_HIST_SIMPLE_ROUTE_LENGTH] = defaultdict(lambda: 0)
        aggregated_metrics[_TOPOLOGY_GRAPH] = net_mngr.network.topology_graph

    
    stats_helper = AdvancedNetworkStatsHelper(net_mngr)
    this_iteration_metrics = {}
    
    # This function call may take a long time to return
    routes = stats_helper.complete_routes_descriptions
     
    
    # Update the histogram of routes lengths (for this topology graph, which
    # should be the same across all iterations
    for route in itertools.chain(*(routes.values())):
        aggregated_metrics[_HIST_SIMPLE_ROUTE_LENGTH][len(route)-1] += 1
    
    
    # Store the corruption states and the full ocom routes for this iteration
    this_iteration_metrics[_CORRUPTION_STATES] = dict((n.id, n.is_corrupted) for n in net_mngr.network.nodes)
    this_iteration_metrics[_FULL_OCOM_ROUTES] = []
    for ocom_route in net_mngr.net_stats.ocom_routes.values():
        end_sender = ocom_route['end_sender']
        end_rcvr = ocom_route['end_rcvr']
        helper = ocom_route['helper']
        end_sender_first_hop_node = ocom_route['first_hop_end_sender']['node']
        end_sender_first_hop_cid = ocom_route['first_hop_end_sender']['cid']
        helper_first_hop_node = ocom_route['first_hop_helper']['node']
        helper_first_hop_cid = ocom_route['first_hop_helper']['cid']
        
        # Find he exact (full) ocom route
        route_first_leg = next((route for route in routes[(end_sender, helper)] 
                                if route[0] == (end_sender, end_sender_first_hop_cid) and route[1][0] == end_sender_first_hop_node))
 
        full_route = [(n, net_mngr.network.nodes[n].is_corrupted) for n, _ in route_first_leg]
        full_route[-1] = (full_route[-1][0], full_route[-1][1], 'helper')
         
        if helper != end_rcvr:
            route_second_leg = next((route for route in routes[(helper, end_rcvr)] 
                                    if route[0] == (helper, helper_first_hop_cid) and route[1][0] == helper_first_hop_node))
             
            full_route += [(n, net_mngr.network.nodes[n].is_corrupted) for n, _ in route_second_leg[1:]]
        
        # Alternatively, to be more compact, we could store: end_sender,
        # end_receiver, nb_hops_corrupted, last_corrupted_node_considered
        this_iteration_metrics[_FULL_OCOM_ROUTES].append(full_route)
    
    return aggregated_metrics, this_iteration_metrics




# #################################
# Plotting functions ############## 
# #################################

# The private module functions (those beginning with "_" below) are all there to
# help process results from simulations, and compute the receiver aonymity.

def _gen_p_distribs_routes_lengths(hist_simple_route_length, nb_nodes):
    # Transform the histogram into a probability distribution
    normaliser = sum(hist_simple_route_length.values())
    p_distrib_length_simple_route = OrderedDict(sorted((l, n/normaliser) for l, n in hist_simple_route_length.items()))
    max_hops_one_route = max(p_distrib_length_simple_route.keys())
          
    # Deduce the compound probability distribution on a full route (two legs)
    max_hops_full_route = 2*max_hops_one_route
    p_distrib_first_leg = copy.deepcopy(p_distrib_length_simple_route)
    p_distrib_second_leg = OrderedDict()
    p_distrib_second_leg[0] = (1/nb_nodes)/(1+1/nb_nodes)
    p_distrib_second_leg.update((l, p/(1+1/nb_nodes)) for l, p in p_distrib_length_simple_route.items())
    assert abs(sum(p_distrib_second_leg.values()) - 1.0) <= 0.000000001, "Second leg p distrib is not a prob distrib: sum is {}\n{}".format(sum(p_distrib_second_leg.values()), p_distrib_second_leg)
    p_distrib_full_route = OrderedDict()
    for l in range1(1, max_hops_full_route):
        p_distrib_full_route[l] = 0
        for l2 in range1(1, max_hops_full_route):
            p_distrib_full_route[l] += p_distrib_first_leg.get(l2,0)*p_distrib_second_leg.get(l-l2, 0)

    assert abs(sum(p_distrib_full_route.values()) - 1.0) <= 0.00000001, "Compound p distrib is not a prob distrib: sum is {}\n{}".format(sum(p_distrib_full_route.values()), p_distrib_full_route)
        
    return p_distrib_full_route

def _gen_p_distrib_receivers_adv_view(corruption_states, topology_graph, p_distrib_length_ocom_routes, full_route, first_honest_node_after_collusion, nb_hops_corrupted, max_remaining_hop_count_adv_view, potential_receivers, longest_shortest_path):
    # Deduce the probability distribution on remaining route length of the adversary          
    p_distrib_remaining_hops_adv_view = {}
    normaliser = sum((p_distrib_length_ocom_routes[l+nb_hops_corrupted] for l in range1(0, max_remaining_hop_count_adv_view)))
    for l in range1(0, max_remaining_hop_count_adv_view):
        p_distrib_remaining_hops_adv_view[l] = p_distrib_length_ocom_routes[l+nb_hops_corrupted]/normaliser
        
    assert abs(sum(p_distrib_remaining_hops_adv_view.values()) - 1.0) <= 0.00000001, "Adversary's distrib on hop count is not a prob distrib: sum is {}\n{}".format(sum(p_distrib_remaining_hops_adv_view.values()), p_distrib_remaining_hops_adv_view)


    # Compute the longest shortest path in the network
    cutoff = min(max_remaining_hop_count_adv_view, longest_shortest_path-1)
    
    
    receivers_at_n_hops = defaultdict(set)
    
    # The only receiver possible at 0 hop is the node next to the last corrupted
    # node
    receivers_at_n_hops[0].add(first_honest_node_after_collusion)
        
    # For all hops longer than the longest shortest path, we can be sure that
    # the anonymity set is maximal
    for l in range1(longest_shortest_path, max_remaining_hop_count_adv_view):
        receivers_at_n_hops[l] = set(potential_receivers)
    
    # To exhaustively test all other hops, use the functions provided by
    # networkx, that forces us to iterate over receivers (rather than route
    # lengths)
    for receiver in potential_receivers:
        all_paths = nx.all_simple_paths(topology_graph, 
                                        source=first_honest_node_after_collusion, target=receiver, 
                                        cutoff=cutoff)
        for path in all_paths:
            receivers_at_n_hops[len(path)-1].add(receiver)
    
    # Normalise the distrib on route lengths, if there is a number of hop at
    # which no receiver is possible
    normaliser = sum(p for l, p in p_distrib_remaining_hops_adv_view.items() if len(receivers_at_n_hops[l]) > 0)
    p_distrib_remaining_hops_adv_view = dict((l, p/normaliser) for l, p in p_distrib_remaining_hops_adv_view.items())
    

    # Finally, compute the prob distrib
    p_distrib_receivers_adversary_view = defaultdict(lambda: 0)
    for l in range1(0, max_remaining_hop_count_adv_view):
        p_this_length = p_distrib_remaining_hops_adv_view[l]
        for receiver in receivers_at_n_hops[l]:
            p_distrib_receivers_adversary_view[receiver] += p_this_length/len(receivers_at_n_hops[l])
    
    
    # Again, a normalisation : because some hops may have no possible
    # honest receiver (i.e. all receiver at e;g. 1 hop are corrupted)
#     normaliser = sum(p_distrib_receivers_adversary_view.values())
#     p_distrib_receivers_adversary_view = dict((r, p/normaliser) for r, p in p_distrib_receivers_adversary_view.items())
      
    assert abs(sum(p_distrib_receivers_adversary_view.values()) - 1.0) <= 0.00000001, "Adversary's distrib on receiver is not a prob distrib: sum is {}\n{}".format(sum(p_distrib_receivers_adversary_view.values()), p_distrib_receivers_adversary_view)
    
    
    return p_distrib_receivers_adversary_view, p_distrib_remaining_hops_adv_view, receivers_at_n_hops



def _simple_collusion_get_collusion_info(full_route):
    # Simple collusion (as opposed to end-to-end collusion) : finds the largest
    # collusion of consecutive corrutped nodes on the route
    
    collusions = [] 
    current_collusion_start_node = None
    for i, n in enumerate(full_route):
        if n[1]:
            if current_collusion_start_node:
                collusions[-1].update({'nb_hops': collusions[-1]['nb_hops']+1, 'last': (i, n[0])})
            else:
                current_collusion_start_node = (i, n[0])
                collusions.append({'first': (i, n[0]), 'nb_hops': 0, 'last': (i, n[0])})
        else:
            current_collusion_start_node = None
    
    largest_collusion = max(collusions, key=lambda x: x['nb_hops'])
    
    assert largest_collusion['last'][0] < len(full_route)-1
    first_honest_node_after_collusion = full_route[largest_collusion['last'][0]+1][0]
    
    # Deduce the non potential receiver, i.e. there can be only one, in all cases
    non_potential_receivers = set()
    second_leg_start = next(i for i in range(len(full_route)) if len(full_route[i])> 2 and full_route[i][2] == 'helper')
    non_potential_receiver_index = largest_collusion['first'][0]-1
    if non_potential_receiver_index >= 0 and non_potential_receiver_index >= second_leg_start:
        non_potential_receivers.add(full_route[non_potential_receiver_index][0])    
    
    
    
     
    # Find the maximum number of remaining hops, according to the adversary's view
    nb_hops_corrupted = (largest_collusion['nb_hops']+2)
    if largest_collusion['first'][0] == 0:
        nb_hops_corrupted -= 1 
     
#     last_corrupted_node_considered = largest_collusion[2]
#     first_honest_node_after_collusion = full_route_nodes_only[full_route_nodes_only.index(last_corrupted_node_considered)+1]
    
    return nb_hops_corrupted, first_honest_node_after_collusion, non_potential_receivers

def _end_to_end_collusion_get_collusion_info(full_route):
    # End-to-end collusion (as opposed to simple collusions): take the first and
    # last corrupted nodes on the route, and consider all the portion of route
    # in-between as being corrupted
    
    full_route_nodes_only = [n[0] for n in full_route]
    
    # Find the first and last corrupted node on the route
    first_honest_node_after_collusion = None
    end_corrupted_nodes_on_route = {'first': None, 'last': None, 'nb_hops': -1}
    for i, n in enumerate(full_route):
        if n[1]:
            if end_corrupted_nodes_on_route['first'] is None:
                end_corrupted_nodes_on_route['first'] = n[0]
            end_corrupted_nodes_on_route['last'] = n[0]
            end_corrupted_nodes_on_route['nb_hops'] = full_route_nodes_only.index(n[0])-full_route_nodes_only.index(end_corrupted_nodes_on_route['first'])
            if i < len(full_route)-1:
                first_honest_node_after_collusion = full_route[i+1][0] 
    
    # Deduce the impossible receiver: the ones that share an edge with a node in
    # the collusion (except the last one) on the *second leg* of the route
    non_potential_receivers = set()
    second_leg_start = next(i for i in range(len(full_route)) if len(full_route[i])> 2 and full_route[i][2] == 'helper')
    for i in range(second_leg_start, len(full_route)): 
        n = full_route[i][0]
        c = full_route[i][1]
        if not c:
            if (i+1 < len(full_route) and full_route[i+1][1]
                or i-1 >= 0 and full_route[i-1][1] and i != len(full_route)-1):
                non_potential_receivers.add(n)
    non_potential_receivers.discard(first_honest_node_after_collusion) # Just to be sure           
                      
    # Compute the maximum remaining number of hops
    nb_hops_corrupted = end_corrupted_nodes_on_route['nb_hops']+2
    if end_corrupted_nodes_on_route['first'] == full_route[0][0]:
        nb_hops_corrupted -= 1 
    
#     last_corrupted_node_considered = end_corrupted_nodes_on_route['last']
    
    return nb_hops_corrupted, first_honest_node_after_collusion, non_potential_receivers

def _compute_receiver_anonymity(nb_nodes, corruption_ratio, topo_graph, hist_simple_route_length, iteration_results, use_simple_collusion=False):
    # Get the prob distrib on routes length, several network runs taken into account
    p_distrib_length_ocom_routes = _gen_p_distribs_routes_lengths(hist_simple_route_length, nb_nodes)
    max_hops_full_route = max(l for l, p in p_distrib_length_ocom_routes.items() if p > 0)
    
    
    all_shortest_paths_lengths = nx.shortest_path_length(topo_graph)
    longest_shortest_path = max(itertools.chain(*([l for l in d.values()] for d in all_shortest_paths_lengths.values())))

    # Iterate over each network run (i.e. over each iteration) that use the
    # same topo graph. Create histograms, that merge values independently
    # from their network run
    hist_normalised_entropies = []
    hist_normalised_min_entropies = []
    hist_most_likely_rcvr_prob = []
    nb_good_guesses = 0
    nb_routes_total = 0
    chances_of_good_guess_alternative = []
    maximal_anonymity = []
    for i, iteration_result in enumerate(iteration_results):
        # Change of plans : we do not use the corruption state of the network,
        # but generate a new corruption state here. That allows the re-use of 
        # the same simulation results, and has no impact on the simulation 
        # itself (results stay fully valid, because corruption state have 
        # no impact on node behavior)
        base_full_ocom_routes = iteration_result[_FULL_OCOM_ROUTES]
#         base_corruption_states = iteration_result[_CORRUPTION_STATES]
        
        for _ in range(100):
            try:
                corrupted_nodes_indexes = Network.gen_corruption_states(nb_nodes, corruption_ratio, topo_graph)
                break
            except SimulationException:
                pass
        else:
            return None
        corruption_states = dict((n, n in corrupted_nodes_indexes) for n in range(nb_nodes))
        full_ocom_routes = [[(n[0], corruption_states[n[0]], 'helper') if len(n) > 2 and n[2] == 'helper' else  (n[0], corruption_states[n[0]]) for n in one_full_ocom_route] for one_full_ocom_route in base_full_ocom_routes]
        
    
        total_honest_receivers = sum(not c for c in corruption_states.values())
#         maximal_entropy = math.log(total_honest_receivers, 2)

        # Iterate over each ocom route, and fill up the histograms
        for full_route in full_ocom_routes:
            # TODO temporary : if receiver corrupted, abort
            if full_route[-1][1]:
                continue
              
            # If the route has no corrupted nodes, the entropy is maximal !
            if not any((n[1] for n in full_route)):
                # If there are no corrupted node on the route, it is the best case scenario !
                hist_normalised_entropies.append(1.0)
                hist_most_likely_rcvr_prob.append(1/total_honest_receivers)
                nb_good_guesses += 0
                nb_routes_total += 1
                chances_of_good_guess_alternative.append(0)
                maximal_anonymity.append(0)
                continue
              
              
            if use_simple_collusion:
                nb_hops_corrupted, first_honest_node_after_collusion, non_potential_receivers = _simple_collusion_get_collusion_info(full_route)
            else:
                nb_hops_corrupted, first_honest_node_after_collusion, non_potential_receivers = _end_to_end_collusion_get_collusion_info(full_route)
            
            max_remaining_hop_count_adv_view = max_hops_full_route-nb_hops_corrupted
            potential_receivers = [n for n, c in corruption_states.items() if not c and n not in non_potential_receivers]
            
              
            p_distrib_receivers_adv_view, _, _ = _gen_p_distrib_receivers_adv_view(corruption_states, topo_graph,
                                                                             p_distrib_length_ocom_routes, 
                                                                             full_route, first_honest_node_after_collusion, 
                                                                             nb_hops_corrupted, max_remaining_hop_count_adv_view,
                                                                             potential_receivers, longest_shortest_path)
            
            
            
            diff = set(p_distrib_receivers_adv_view.keys()) - set(potential_receivers)
            if diff:
                print("Error : potential receivers and p distrib do not feature the same receivers")
                print(diff)
                print("Potentials : ", potential_receivers)
                print("p distrib : ", list(p_distrib_receivers_adv_view.keys()))
                print(full_route)
                exit()
            
            # Deduce entropy, maximal prob, and most likely receiver
            entropy = -functools.reduce(lambda total, p: total+(p*math.log(p, 2)), p_distrib_receivers_adv_view.values(), 0)
            most_likely_receiver_prob = max(p_distrib_receivers_adv_view.values())
            min_entropy = -math.log(most_likely_receiver_prob, 2)
            if len(potential_receivers) == 1:
                normalised_entropy = 0
                normalised_min_entropy = 0
            else:
                max_entropy = (-math.log(1/total_honest_receivers, 2))
                normalised_entropy = entropy/max_entropy
                normalised_min_entropy = min_entropy/max_entropy
            
            
            
            most_likely_receivers = [n for n, p in p_distrib_receivers_adv_view.items() if abs(p - most_likely_receiver_prob) <= 0.0000001] 
#             
            hist_normalised_entropies.append(normalised_entropy)
            hist_normalised_min_entropies.append(normalised_min_entropy)
            hist_most_likely_rcvr_prob.append(most_likely_receiver_prob)
            good_guess = (random.choice(most_likely_receivers) == full_route[-1][0])
            nb_good_guesses += good_guess
            nb_routes_total += 1
            if full_route[-1][0] not in most_likely_receivers:
                chances_of_good_guess_alternative.append(0)
            else:
                chances_of_good_guess_alternative.append(1/len(most_likely_receivers))
            maximal_anonymity.append(1/total_honest_receivers)
            
            assert normalised_entropy < 1.0
            assert most_likely_receiver_prob >= (1/len(potential_receivers))-0.00000001, ("max prob = {}, optimal prob = {}, \n"
                                                                                          "maxprob*len(potentials) = {}, len(potentials) = {}, len(prodistrib) = {}\n"
                                                                                          "difference is : {}\n"
                                                                                          "full route is {} and potentials are {}\n{}".format(most_likely_receiver_prob, 1/len(potential_receivers), 
                                                                                                                                              most_likely_receiver_prob*len(potential_receivers),
                                                                                                                                              len(potential_receivers),
                                                                                                                                              len(p_distrib_receivers_adv_view),
                                                                                                                                              set(potential_receivers) ^ set(p_distrib_receivers_adv_view.keys()),
                                                                                                                                              full_route,
                                                                                                                                              potential_receivers,
                                                                                                                                             "\n\t".join(["{} : {}".format(r, p) for r, p in p_distrib_receivers_adv_view.items()])))
            
            
            
            
        
    frequence_good_guesses = nb_good_guesses/nb_routes_total
    return hist_normalised_entropies, hist_normalised_min_entropies, frequence_good_guesses, chances_of_good_guess_alternative, maximal_anonymity

            

def plot_RECEIVER_ANONYMITY_vs_theoretical(experiments_results, restrict_to_other_params={}, save_to=None, overwrite_existing=False, use_simple_collusion=True, results_augmentation=None, use_saved_poltting_results=False):
    """Plots the receiver anonymity, along with the theoretical bound on receiver anonymity obtained from the formal proofs of the protocol"""
    
    if save_to is not None:
        save_to = os.path.join(save_to, 'receiver_anonymity/')
    x_param = ('corruption_ratio', 'corruption ratio')
    y_stat = (_RECEIVER_ANONYMITY, 'prob. of breaking RA', 'receiver_anonymity')
    
    
    x_param, y_stat, restrict_to_other_params = process_plot_function_inputs(x_param, y_stat, restrict_to_other_params)
    x_param, x_param_label, x_param_file_name = x_param
    y_stat, y_stat_label, y_stat_file_name = y_stat
    
    # Load the param combinations for which the graphs were already computed.
    if not use_saved_poltting_results:
        all_possible_graphs = defaultdict(lambda: defaultdict(RunningStat))
        all_possible_graphs_alternative = defaultdict(lambda: {'chances_good_guess': defaultdict(list), 'maximal_anon': defaultdict(list)})
        for i, experiment in enumerate(experiments_results):
            # Keep the experiment results only if they fit into the asked other parameters
            skip = False
            for p, v_list in restrict_to_other_params.items():
                if  experiment['measure_params'][p] not in v_list:
                    skip = True
                    break        
            if skip:
                continue
            
            
            other_measure_params = list(sorted(((p, v) for p, v in experiment['measure_params'].items() if p != x_param and p in MEASURE_PARAMS), key=lambda x: MEASURE_PARAMS.index(x[0])))
            
            other_measure_params.append(('nb_iters', experiment['measure_params']['nb_iters']))
            collusion_type = 'simple' if use_simple_collusion else 'endtoend'
            other_measure_params.append(('collusion', collusion_type))
            if experiment['net_params'].rtprop_policy_max_hop_count:
                lmax = experiment['net_params'].rtprop_policy_max_hop_count
            else:
                all_shortest_paths_lengths = nx.shortest_path_length(experiment['results']['overall'][_TOPOLOGY_GRAPH])
                lmax = 1+max(itertools.chain(*([l for l in d.values()] for d in all_shortest_paths_lengths.values())))
            other_measure_params.append(('rtprop_policy_max_hop_count', lmax))
            other_measure_params = make_hashable(other_measure_params)
            
            
            if results_augmentation is None:
                results_augmentation = [experiment['measure_params'][x_param]] 
            # Then results_augmentation contains a list of corruption ratio values, that we "simulate" on the acquired network state
            for x_value in results_augmentation:
                x_value = make_hashable(x_value)
                
                print("\tTreating network runs {}/{} with corruption ratio {} and params {}".format(i, len(experiments_results), x_value, other_measure_params))
                
                # Do the actual computation of sender/receiver privacy 
                aux = _compute_receiver_anonymity(experiment['measure_params']['nb_nodes'], 
                                                                           x_value, 
                                                                           experiment['results']['overall'][_TOPOLOGY_GRAPH], 
                                                                           experiment['results']['overall'][_HIST_SIMPLE_ROUTE_LENGTH], 
                                                                           experiment['results']['by_ite'],
                                                                           use_simple_collusion=use_simple_collusion)
                if aux is None:
                    continue
                _, _, frequence_good_guesses, chances_of_good_guess_alternative, maximal_anonymity = aux 
                
                
                all_possible_graphs[other_measure_params][x_value].push_value(frequence_good_guesses)
                all_possible_graphs_alternative[other_measure_params]['chances_good_guess'][x_value].append(chances_of_good_guess_alternative)
                all_possible_graphs_alternative[other_measure_params]['maximal_anon'][x_value].append(maximal_anonymity)
            print()
                    
            
        
        # Each graph must show a statistic (in y axis) according to a network
        # parameter (in x axis), where **only** this network parameter varies. Thus,
        # we make one graph per set of other network parameters
        all_possible_graphs = OrderedDict(sorted(all_possible_graphs.items(), key=lambda x: x[0]))
        all_possible_graphs_alternative = OrderedDict(sorted(all_possible_graphs_alternative.items(), key=lambda x: x[0]))

    
        with open(os.path.join('/tmp', 'allpossiblegaphsvstheoretical.pickle'), 'wb') as f:
            pickle.dump(all_possible_graphs, f)
        with open(os.path.join('/tmp/', 'allpossiblegaphsvstheoretical_alternative.pickle'), 'wb') as f:
            pickle.dump(all_possible_graphs_alternative, f)
    
        if not os.path.isdir(save_to):
            os.makedirs(save_to, mode=0o775)
        with open(os.path.join(save_to, 'allpossiblegaphsvstheoretical.pickle'), 'wb') as f:
            pickle.dump(all_possible_graphs, f)
        with open(os.path.join(save_to, 'allpossiblegaphsvstheoretical_alternative.pickle'), 'wb') as f:
            pickle.dump(all_possible_graphs_alternative, f)
    else:
        with open(os.path.join(save_to, 'allpossiblegaphsvstheoretical.pickle'), 'rb') as f:
            all_possible_graphs = pickle.load(f)
        with open(os.path.join(save_to, 'allpossiblegaphsvstheoretical_alternative.pickle'), 'rb') as f:
            all_possible_graphs_alternative = pickle.load(f)
        

    def theoretical_RA(nb_nodes, lmax, x): 
#         print("For x = {} (c = {}), Return 1-{}/{}".format(x, math.floor(nb_nodes*x), comb(nb_nodes-lmax, math.floor(x*nb_nodes)), comb(nb_nodes, math.floor(x*nb_nodes))))
        return 1-comb(nb_nodes-lmax, x)/comb(nb_nodes, x)
    
    i = 0
    for other_measure_params, graph_values in all_possible_graphs.items():
        x_values = {}
        y_values = {}
        stdevs = {}
        sorted_zipped_data = sorted(((k, v.mean, v.stdev) for k, v in graph_values.items())
                                , key=lambda x: x[0])
    
        x_values["empirical2"], y_values["empirical2"], stdevs["empirical2"] = map(list, zip(*sorted_zipped_data))
        stdevs["empirical2"] = None
        
        sorted_zipped_data = sorted(((k, statistics.mean([statistics.mean(v2) for v2 in v]), 0 if len(v) <= 1 else statistics.stdev([statistics.mean(v2) for v2 in v])) 
                                     for k, v in all_possible_graphs_alternative[other_measure_params]['chances_good_guess'].items())
                                , key=lambda x: x[0])
    
        x_values["empirical"], y_values["empirical"], stdevs["empirical"] = map(list, zip(*sorted_zipped_data))
        stdevs["empirical"] = None
        
        sorted_zipped_data = sorted(((k, statistics.mean([statistics.mean(v2) for v2 in v]), 0 if len(v) <= 1 else statistics.stdev([statistics.mean(v2) for v2 in v])) 
                                     for k, v in all_possible_graphs_alternative[other_measure_params]['maximal_anon'].items())
                                , key=lambda x: x[0])
    
        x_values["maximal_anon"], y_values["maximal_anon"], stdevs["maximal_anon"] = map(list, zip(*sorted_zipped_data))
        stdevs["maximal_anon"] = None
                
        other_measure_params_dict = dict(other_measure_params)
        
        f = functools.partial(theoretical_RA, other_measure_params_dict['nb_nodes'], other_measure_params_dict['rtprop_policy_max_hop_count'])
        x_values['theoretical'] = [c/other_measure_params_dict['nb_nodes'] for c in range1(other_measure_params_dict['nb_nodes'])]
        y_values['theoretical'] = [f(x) for x in list(range1(other_measure_params_dict['nb_nodes']))]
        stdevs['theoretical'] = None
        
        
        print()
        print("For params = {}".format(", ".join(map(lambda v: str(v[0])+": "+str(v[1]), other_measure_params))))
        c = [0.1,0.5]
        for x, y in zip(x_values['empirical'], y_values["empirical"]):
            if x in c:
                print("\t c = {} => {}".format(x, y))
            
        
        
        
        graph_description = {'x_label': x_param_label, 'y_label': y_stat_label}
        graph_description['title'] = ",\n".join(map(lambda v: str(v[0])+": "+str(v[1]), other_measure_params))
        
    
        # Look for a fit
        def linear_fit(x, a=None, b=None): 
            if x is None:
                return (0.7, 0)
            return a*x+b
        def power_fit(x, a=None, b=None,c=None):
            if x is None:
                return (2, 1, 0) 
            return numpy.power(x, a)*b + c
        def sqrt_fit(x, a=None, b=None):
            if x is None:
                return (1, 0) 
            return numpy.sqrt(x)*a + b
        def inverse_fit(x, a=None, b=None):
            if x is None:
                return (1, 0) 
            return a/x+b
        def inverse_sigmoidal_fit(x, L=None, k=None, y0=None):
            if x is None:
                return (1, 10, 0.4)
            return -numpy.log((L-x)/x)/k + y0
        def inverse_generalised_sigmoidal_fit(x, a=None, b=None,c=None,d=None,f=None,g=None):
            if x is None:
                return (0, 1, 1, 1, 1, 1)
            return -(1/f)*numpy.log((  numpy.power((b-a)/(x-a), g) - c   )/d)
                
        
#         candidate_functions = [power_fit, inverse_generalised_sigmoidal_fit, inverse_sigmoidal_fit, sqrt_fit]
#         for j, candidate_function in enumerate(candidate_functions):
#             try:
#                 popt, _ = curve_fit(candidate_function, numpy.array(x_values['empirical']), numpy.array(y_values['empirical']), p0=candidate_function(None))
#                 print("{} : {}".format(candidate_function.__name__, popt))
#             except RuntimeError:
#                 continue
#             x_values['empirical_fit_{}'.format(candidate_function.__name__)] = x_values['theoretical']
#             y_values['empirical_fit_{}'.format(candidate_function.__name__)] = [candidate_function(x, *popt) for x in x_values['empirical_fit_{}'.format(candidate_function.__name__)]]
#             stdevs['empirical_fit_{}'.format(candidate_function.__name__)] = None
#         
#         several_curves = ['theoretical', 'empirical'] + ([s for s in x_values.keys() if s.startswith('empirical_fit')])
#         colors = ['k', 'r'] + (['g', 'b', 'y', 'm']*math.ceil(len(several_curves)-2))
#         markers = ['', 'o'] + (['']*(len(several_curves)-2))
#         linewidths = [1,0] + ([1]*(len(several_curves)-2))
        
        several_curves = ['theoretical', 'empirical']
        for k in list(x_values.keys()):
            if k not in several_curves:
                del x_values[k]
                del y_values[k]
                del stdevs[k] 
        colors = ['k', 'b', 'g'] 
        markers = ['', 'o', 'o']
        linewidths = [1,1, 1]
        
        
        if save_to:
            graph_description['file_name'] = "{}_vs_{}".format(y_stat_file_name, x_param_file_name).lower() + "_"+str(i)
            i+=1
        do_plot_simple_metric(x_values, y_values, stdevs, graph_description, 
                              several_curves=several_curves, legend_placement='upper left', color=colors, marker=markers, linewidth=linewidths, xlim=(0,1), ylim=(0, 1.01),
                              save_to=save_to, overwrite_existing=overwrite_existing)



if __name__ == '__main__':
    folder = os.path.join(_DEFAULT_SAVING_FOLDER, 'measure_'+MEASURE_TITLE)
    
    do_measures = True
    do_plot = True
    
    if do_measures:
        measure_privacy(nb_nodes=10,
                                    nb_iters=2,
                                    nb_iters_same_topo_graph=2,
                                    corruption_ratio=0.33,
                                    save_results_to=folder,
                                    overwrite_existing=True)
    
    if do_plot:
        results = load_measures_results(folder, MEASURE_TITLE+".pickle")
        folder_graphs = os.path.join(folder, "graphs")
        
        print("\nPlotting graphs...")
        for stat in  MEASURE_STATS:
            plot_RECEIVER_ANONYMITY_vs_theoretical(results, save_to=folder_graphs, overwrite_existing=True, use_simple_collusion=True)

    
    