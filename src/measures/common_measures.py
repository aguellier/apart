# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
This module provides helper function to make simulations, experimentations and
measures. It in particular provides the function :func:`.generic_measure`, which
allows to run many network runs, and gather statistics on them.


.. seealso::
    See the examples measure modules that make use of the
    :func:`measures.common_measures.generic_measure` function. These may be
    found in the the `examples/` folder, or in the *Example* topic of the
    documentation.
'''
from collections import defaultdict, OrderedDict
import copy
import functools
import gc
import math
import os
import random
import re
import sys
import time

from common.custom_logging import *
from apart.core.network import NetworkParams, Network
from apart.simulation import run_simulation, SimulationParams
from measures.file_handling import matchstring_results_file_name, \
    format_net_states_file_name, save_network_state, format_results_file_name, save_measures_results, \
    load_measures_results


_NET_STATES_REL_FOLDER = 'net_states/'


def generic_measure(measure_title, save_results_to, net_params_combinations, metrics_computer_callback, 
                    sim_params=None, net_params=None, save_network_states=True, overwrite_existing=False,
                    force_recomputation=False, restrict_to_params=None,
                    log_to_file=True):
    """Generic function to run a series of network simulations, and gather statistics about them.
    
    For each parameter combination provided in net_params_combinations, this
    function runs  series of network simulations, computes statistics about them
    using the callback given in metrics_computer_callback (which also aggregates
    the results over several network runs).    
    
    More exactly, for each parameter combination (provided as dict) in the
    net_params_combinations list, one or several network simulations are
    launched (this depends on the value of the nb_iters parameter, given in this
    dict).
       
    Because measures may take a very long time, and may fail or be interrupted
    (*e.g.* because of a system shutdown), by default, the function checks if
    there are already some results saved for this series of measure. If so, it
    resumes to the next missing parameter combination. However, this behavior
    can be modified, by specifying force_recomputation to `True`.
    
    Finally, this function does *not* run several network simulations in
    parallel, and uses only one core. It is however theoretically possible to
    parallelize this work, since each parameter combination can be processed
    independently. To allow a form of parallelism, the python program must be
    launched several times. This is where the restrict_to_params argument
    becomes useful. It allows to restrict the set of parameters combinations to
    compute, and thus partition the measures, allowing to launch them in
    parallel. An example of this partition can be found in the
    :mod:`~apart.measures.measure_campains`.
    
    
    The function can be further tweaked:
        
        * For each parameter combination, a certain number of iterations *on the
          same topology graph* can be ran. And several runs for different
          topology graphs can also be run, for a given parameter set. See
          :func:`.make_params_combinations` for more information.
        * The function allows the specification of sim_params and net_param
          (respectively instances of :class:`~apart.simulation.SimulationParams`
          and :class:`~apart.core.network.NetworkParams`), common to all
          simulation that will be run. Of course, some of these parameters will
          be overridden by the parameters specified in net_params_combinations.
          That is, sim_params and net_params specify the *default* parameters.
        * If save_network_states is `True`, then the
          :obj:`~apart.core.network.NetworkManager` instance after each network
          simulation is saved in the sub-folder net_states/  within the folder
          specified by save_results_to.
        * If overwrite_existing is set to `True`, then when the simulation resulsts
          (and network states) are saved, previous measure results may be erased.
        * The log_to_file argument modifies the standard /O behavior of the function.
          By default, all information that is logged as per the network simulation
          (mainly by the code in the class :class:`~apart.core.node.Node`), allowing
          browsing of the log after the measures are finished. However, if log_to_file
          is set to `False`, these logging messages are made on the standard output.
    
    Args:
        measure_title (string): the name of the measure (user-defined, can be anything)
        save_results_to (string): folder in which to save the simulation results (and possibly the network states)
        net_params_combinations (list of dict): the set of parameter combinations. Should be the output of a call to :func:`.make_params_combinations`.   
        metrics_computer_callback (function): the function computing and aggregating (averaging) the statistics between each network run. 
                Must accept two arguments: one that is passed on in between every network run over the same combination parameters, 
                and one corresponding to the :obj:`~apart.core.network.NetworkManager` instance. 
        sim_params (:obj:`~apart.simulation.SimulationParams`, optional): the simulation parameters for all network runs. (Default: None)
        net_params (:obj:`~apart.core.network.NetworkParams`, optional): the (default) network parameters for all network runs. Overridden by the parameter combinations from the net_params_combinations argument. (Default: None) 
        save_network_states (bool, optional): if `True` the network state (*i.e.* :obj:`~apart.core.network.NetworkManager` instance) of each and every network run is saved to the disk. Can consume several gigabytes of disk memory. (Default: True)
        overwrite_existing (bool, optional): if `True`, then the simulation results (and possibly the network states) that are saved are allowed to erase existing files.  (Default: False)
        force_recomputation (bool, optional): if `False`, resumes a measure starting from where it was stopped previously. Otherwise, process *all* parameter combinations, even if measures for these parameters were already carried out (Default: False) 
        restrict_to_params (dict, optional): a dict indexed by the names of parameters (given as strings), with values specifying lists of values to process (parameter combinations with values not in these lists are skipped). These restrictions can be used along with the resuming of computations. When this argument is `None`, no restrictions are applied. (Default: None) 
        log_to_file (bool, optional): if `True`, redirects all logging output from the network run into a the file which name is based on the measure_tile argument (Default: True)
        
    Returns:
        list of dict: a list in which each element gives the results for one parameter combinations. Each element of the list is a dict of the form::
        
        {
         'measure_params': #the parameter combination
         'sim_params': sim_params, 'net_params': net_params, 
         'results':
             {
             'overall': # The final results of the measure. Usually these are the only that matter.
                 {'stat1': stat1_value, 'stat2': ...},
             'by_ite': # Result by iteration (there may be several iterations for a given parameter combination)
                 [{'stat1': stat1_val, 'stat2': ...}, ...]
             },
         }
    """
    
    if log_to_file:
        logger = _get_logger(log_to_file=os.path.join(save_results_to, measure_title+".log"))
    else:
        logger = _get_logger()
    
    logger.info("-"*80)
    logger.info("NEW MEASURE RUN")
    logger.info("")
    
    
    # Look for existing measure files
    sim_outputs_file_names=[]
    all_combinations_indexes_to_do = []

    if restrict_to_params:
        for comb_index, comb in enumerate(net_params_combinations):
            for p, v in restrict_to_params.items():
                if isinstance(v, list):
                    if comb[p] not in v: 
                        break
                else:
                    if comb[p] != v:
                        break
            else:
                all_combinations_indexes_to_do.append(comb_index)
    else:
        all_combinations_indexes_to_do = list(range(len(net_params_combinations)))
    
    
    # Compute all the intervals to do
    if len(all_combinations_indexes_to_do) == 0:
        param_comb_intervals = []
    elif len(all_combinations_indexes_to_do) == len(net_params_combinations):
        param_comb_intervals = [(1, len(net_params_combinations))]
    else:
        param_comb_intervals = []
        current_interval = (all_combinations_indexes_to_do[0], all_combinations_indexes_to_do[0])
        for comb_index in all_combinations_indexes_to_do[1:]:
            if current_interval[1] == comb_index-1:
                current_interval = (current_interval[0], comb_index)
            else:
                param_comb_intervals.append(current_interval)
                current_interval = (comb_index, comb_index)
        param_comb_intervals.append(current_interval)
        param_comb_intervals = [(i[0]+1, i[1]+1) for i in param_comb_intervals]
    
    
    logger.info("Starting measures for '{}'".format(measure_title))
    total_nb_combinations = len(net_params_combinations)
    logger.info("{} combinations of parameters to test in total".format(total_nb_combinations))

    if restrict_to_params:
        logger.info("Restriction to parameters {}".format(restrict_to_params))
    
    if param_comb_intervals:
        param_comb_intervals_str = ", ".join(("[{}, {}]".format(*i) for i in param_comb_intervals))
        logger.info("\t({} combinations, param_comb_intervals: {})".format(len(all_combinations_indexes_to_do), param_comb_intervals_str)) 
    else:
        logger.info("\tNothing to do")
    
    
    # Search in the folder, if there are already files for this measure, of the
    # type "[measure title]_i.pickle", then search for the greatest i, and set
    # combination_counter = i. This is useful to resume a series of measures
    # that crashed mid-way. This automatic feature can be explicitely disabled.
    combinations_indexes_to_do = set(all_combinations_indexes_to_do)
    if os.path.exists(save_results_to) and not force_recomputation:
        dir_listing = os.listdir(save_results_to)
        for file in dir_listing:
            match_obj = re.match(matchstring_results_file_name(measure_title), file)
            if match_obj is not None:
                sim_outputs_file_names.append(file)
                combinations_indexes_to_do.discard(int(match_obj.group(1))-1)
        
    combinations_indexes_to_do = list(sorted(combinations_indexes_to_do))
    
    if len(combinations_indexes_to_do) < len(all_combinations_indexes_to_do):
        logger.info("Resuming measure: only {} measures remaining.".format(len(combinations_indexes_to_do)))
    
    

        
    
    
    # Init randomness
    random.seed(math.ceil(time.time() * 10000))

    # Generate the simulation parameters and pre-set some general parameters
    if not sim_params:
        sim_params = SimulationParams(logging_level=logging.WARNING, print_nodes_tables=False, 
                                      draw_topology_graph=False, draw_routes=None, 
                                      time_of_simulation=0, automatic_oriented_ocomm_phase=False, 
                                      oriented_communication_sessions=None) 


    
    # MAIN LOOP
    # Iterate over all the combination of parameters to process
#     for combination_counter, current_interval in itertools.chain(*([(i, interval) for i in range1(*interval)] for interval in param_comb_intervals)):
    for combination_index in combinations_indexes_to_do:
#         combination_index = combination_counter - 1
        combination_counter = combination_index+1
        p_comb = net_params_combinations[combination_index]
        
        # Set the simulation parameters accordingly
        this_combination_sim_params = copy.deepcopy(sim_params)
        # A particular case for the ocom sessions: it is a simulation param
        if 'ocom_sessions' in p_comb:
            this_combination_sim_params.oriented_communication_sessions = p_comb['ocom_sessions']
        
        # Get the number of iterations to do for this param combination
        this_combination_nb_iters = p_comb['nb_iters']

        # Create a set of network parameters
        if net_params:
            this_combination_net_params = copy.deepcopy(net_params)
            this_combination_net_params.update_params(**p_comb)
        else:
            this_combination_net_params = NetworkParams(**p_comb)

        this_params_combination_text = functools.reduce(lambda acc, x: acc + x[0] + "=" + str(x[1]) + ", ", p_comb.items(), "")
        logger.info("Combination {} in {}. Starting {} simulations with network params {}".format(combination_counter, param_comb_intervals_str, this_combination_nb_iters, this_params_combination_text))
    
        # Prepare the structure for this simulation output
        this_param_combination_results = {'measure_params': copy.deepcopy(p_comb),
                                         'sim_params': this_combination_sim_params,
                                         'net_params': this_combination_net_params, 
                                         'results': {'overall': {}, 'by_ite': []}}

        # This structure allows to aggregate the stats between iterations, e.g.
        # so as to compute the mean on-the-go
        aggregated_results = None
#         shortcut_to_compute_mean = c.defaultdict(list)

        # For 1 combination of parameters iterate as many times as required by
        # nb_iters for this combination
        skip_to_next_combination = False
        percent_progress = 0
        previous_percent_decile = -1
        for i in range(this_combination_nb_iters):
            # logger.info("\t\t\t\tTo start simulation i = {}. Press any key...".format(i))
            # input()
            # logger.info("\t\t\t\t Starting simulation i = {}".format(i))

            # Print progression in a sort of progress bar
            percent_progress = math.floor((i + 1) * 100 / this_combination_nb_iters)
            percent_decile = math.floor(percent_progress / 10) * 10
            if percent_decile > previous_percent_decile:
                print("{}% ".format(percent_decile, i), end="", flush=True)
                previous_percent_decile = percent_decile

                if percent_progress == 100:
                    print()
                else:
                    print("... ", end="")

            # Run simulation once. If error, catch and begin again, up to 20 times
            for _ in range(20):
                try:
                    this_iteration_net_manager = run_simulation(sim_params=copy.deepcopy(this_combination_sim_params), net_params=copy.deepcopy(this_combination_net_params))
                    break
                except Exception as e:
                    logger.error("Error in network run : Exception '{}', msg = {}".format(e.__class__.__name__, str(e)))
            else:
                logger.error("Fatal error: after 20, could not run the network simulation properly. Skipping parameter combination")
                skip_to_next_combination = True
                break
            
            aggregated_results, this_iteration_results = metrics_computer_callback(aggregated_results, this_iteration_net_manager)
            

            # Compute relevant statistics
#             this_iteration_results = AdvancedNetworkStatsHelper.compute(stats_to_compute, this_iteration_net_manager)
            
            # If asked to, save the network state. Do so, identify it by the network uid
            if save_network_states:
                file_name = format_net_states_file_name(measure_title, combination_counter, i)
                save_network_state(this_iteration_net_manager, file_name=file_name, folder=os.path.join(save_results_to, _NET_STATES_REL_FOLDER))

            
            # Clean and free all memory possible
            del this_iteration_net_manager
            gc.collect()

            # Record the results in the 'by_ite' field of the current value
            # tested for num_nodes
            this_param_combination_results['results']['by_ite'].append(this_iteration_results)

            # Fill the structure that will be used to compute the overall mean
            # and std dev of the statistics computed
#             for s, sval in this_iteration_results['computed_stats'].items():
#                 shortcut_to_compute_mean[s].append(sval)

        # END for (end of simulations for this combination of params)
        
        if skip_to_next_combination:
            continue
        
        this_param_combination_results['results']['overall'] = aggregated_results

        # Write results of all iteration of this parameters combination to the disk
        logger.info("Writing results of parameters combination #{} of measure '{}' to disk on {}...".format(combination_counter, measure_title, time.asctime()))
        file_name = format_results_file_name(measure_title, combination_counter, this_combination_nb_iters) 
        save_measures_results(this_param_combination_results, folder=save_results_to, file_name=file_name, overwrite_existing=overwrite_existing)
        logger.info("")
        sim_outputs_file_names.append(file_name)
        

    # END for loop over all parameters combinations
    
    # Merge all intermediary files into one big pickle file, overwriting it if it exists
    logger.info("Merging all results of measures on {}".format(time.asctime()))
    merge_measures_outputs(save_results_to, sim_outputs_file_names, new_file_name=measure_title+".pickle", sort_key=lambda x: x['measure_params']['nb_nodes'], overwrite_existing=True)
        
    logger.info("Measures done.")



def merge_measures_outputs(folder, files_list, new_folder=None, new_file_name=None, restrict_to_params={}, sort_key=None, overwrite_existing=False):
    """Merge several files, each containing one or several serialized simulation results, and put them together in one big file.
    
    This function merges simulation results, saves them to a new file, and also returns the result. 
    
    Args:
        folder (string): the folder in which to search for the binary files of the serialized simulation results
        files_list (string): the files to merge together
        new_folder (string, optional): folder where to save the merged data. By default, the result file is saved in the specified folder argument (Default: None). 
        new_file_name (string, optional): name of the file to save the merged data. A random default name is used if none is provided. (Default: None)
        restrict_to_params (dict, optional): of a similar form the to argument of :func:`.generic_measure` of the same name. Allows to filter out some parameter combinations from the results. By default, no filtering is applied. (Default: None).
        sort_key (function, optional): function used to sort the measure results (Default: None).
        overwirte_existing (bool, optional): whether, if the file_path provided already exists, it should be overwritten or not (Default: False)
    
    Returns:
        list of dict: the merged results, in a structure of the same form as the output of :func:`.generic_measure`.
    
    Raises:
        :exc:`.MeasureException`: if at least one of the files merged does not contain the expected data (a list or a dict) 
    """
    logger = _get_logger()

    logger.info("Merging results from following files:")

    # Load and merge all files
    sims_outputs = []
    for file_name in files_list:
        file_path = os.path.join(folder, file_name)
        file_contents = load_measures_results(None, os.path.abspath(file_path))
        if isinstance(file_contents, dict): 
            # Case when the file contains result for *one* parameters combination
            file_contents = [file_contents]
        if not isinstance(file_contents, list):
            raise MeasureException("Error while merging measure results: got file contents of type '{}' (expected dict of list)".format(type(file_contents).__name__))
        
        logger.info("* {}".format(file_path))
        
        for one_result in file_contents:
#             if (one_result['measure_params']['nb_nodes'] == 20
#                 and one_result['measure_params']['rtprop_policy_p_reaccept'] == 0.7):
#                 print("FOUND")
#                 for k, v in one_result['measure_params'].items():
#                     print("{} : {}".format(k, v))
#                 exit()
            for p, v in restrict_to_params.items():
                if isinstance(v, list):
                    if one_result['measure_params'][p] not in v: 
                        break
                else:
                    if one_result['measure_params'][p] != v:
                        break
            else:
                sims_outputs.append(one_result)
                

    if sort_key is not None:
        logger.info("Sorting merged results according to provided key.")
        sims_outputs.sort(key=sort_key)
        
    if new_folder is None:
        new_folder = folder
    
    # Rewrite everything in one file
    return save_measures_results(sims_outputs, new_folder, new_file_name, overwrite_existing=overwrite_existing)



def merge_measures_outputs_with_prefix(folder, file_prefix, new_folder=None, new_file_name=None, restrict_to_params={}, sort_key=None, overwrite_existing=False):
    """Variant of :func:`.merge_measures_outputs` where files sharing a common prefix in their name are merged.
    
    The only difference with :func:`.merge_measures_outputs` is the second
    argument: based on a prefix, the file list is built automatically.
    
    Args:
        folder (string): the folder in which to search for the binary files of the serialized simulation results
        file_prefix (string): the common prefix of files from the specified folder that must be merged
        new_folder (string, optional): folder where to save the merged data. By default, the result file is saved in the specified folder argument (Default: None). 
        new_file_name (string, optional): name of the file to save the merged data. A random default name is used if none is provided. (Default: None)
        restrict_to_params (dict, optional): of a similar form the to argument of :func:`.generic_measure` of the same name. Allows to filter out some parameter combinations from the results. By default, no filtering is applied. (Default: None). 
        sort_key (function, optional): function used to sort the measure results (Default: None).
        overwirte_existing (bool, optional): whether, if the file_path provided already exists, it should be overwritten or not (Default: False)
    
    Returns:
        list of dict: the merged results, in a structure of the same form as the output of :func:`.generic_measure`.
    
    Raises:
        :exc:`.MeasureException`: if at least one of the files merged does not contain the expected data (a list or a dict)
    """
    logger = _get_logger()
    
    if new_file_name is None:
        new_file_name = file_prefix + ".pickle"
    if new_folder is None:
        new_folder = folder

    # Find files with specified prefix
    dir_listing = os.listdir(os.path.abspath(folder))
    files_list = []
    for file in dir_listing:
        if re.match(file_prefix+"(.+).pickle", file) is not None and file != file_prefix + ".pickle":
            files_list.append(os.path.join(folder, file))

    if not files_list:
        logger.warning("No file with prefix '{}' were found in folder '{}'.".format(file_prefix, folder))
        return

    return merge_measures_outputs(None, files_list, new_folder, new_file_name, restrict_to_params=restrict_to_params, sort_key=sort_key, overwrite_existing=overwrite_existing)



def make_params_combinations(nb_iters, nb_iters_same_topo_graph, measure_params_order, measure_params):
    """Helper function that creates the parameter combination structure.
    
    To make the generation of parameter combinations easy and dynamic, this
    function takes as input a dict (in measure_params) containing an entry for
    each parameters of the measure, and specifying for each of them a list of
    values. The function then creates a list of dict, by recursively generating
    all possible combinations of parameters, based on these list of values.
    
    The number of iterations per set of parameter, and the number of iteration
    for a same topology graph are handled independently to the rest of the
    parameters, and must be the of a specific form. See the function
    :func:`.preprocess_measure_input_params` for details on the form of each
    parameter. Any call to `make_params_combinations` should be preceded by a
    call to :func:`.preprocess_measure_input_params`.
    
    Args:
        nb_iters (dict): a dict indexed by the values of the parameter
            `'nb_nodes'`, representing the number of network simulations that
            must be run over each parameter combinations. This argument is made
            to depend on the value(s) of `'nb_nodes'` because the time taken by
            a simulation largely depends on the number of nodes in the
            simulation. It is advised to greatly decrease the number of
            iterations as the number of nodes in the network to be simulated
            increases. This parameter can also be a:obj:`collections.defaultdict`.
        nb_iters (list of dict): this parameter is similar to nb_iters, except
            that it allows to specify how many runs should be made on any given
            topology graph. It must be provided as a list that contains dict of
            the same form as the nb_iters argument. Each element of this list
            will yield different combination(s) of parameters. On the contrary,
            the nb_iters parameter does not make the number of parameter
            combinations augment.
        measure_params_order (list of string): the names of the measure
            parameters, in order (the order is only to be user-friendly in the
            printing of results, and does not actually matter)
        measure_params (dict): the desired measure parameters (should be
            processed and checked by a call to
            :func:`.preprocess_measure_input_params`). This dict (in
            measure_params) *must* contain an entry for each parameters of the
            measure (those present in measreu_params_order, plus `'nb_iters'`
            and `'nb_iters_same_topo_graph'`), and specify for each of them a
            list of values.
    
    Returns:
        list of dict: the list of parameter combinations, which is of the following form::
        
        [
            {'param1': p1_value,
             'param2': p2_value,
             ...
            },
            ...
        ]
    """
    def inner_make_params_combinations(params_dict, param_n=0, **kwargs):
        kwargs_copy = copy.deepcopy(kwargs)
        if not params_dict:
            this_param_combinations = OrderedDict((p, kwargs[p]) for p in measure_params_order)
            new_combinations = []
            for n_dict in nb_iters_same_topo_graph:
                this_param_combinations = copy.deepcopy(this_param_combinations)
                this_param_combinations['nb_iters'] = n_dict[kwargs['nb_nodes']]
                for _ in range(nb_iters[kwargs['nb_nodes']]):
                    this_param_combinations = copy.deepcopy(this_param_combinations)
                    this_param_combinations['topology_graph'] = Network.gen_topology_graph(kwargs['nb_nodes'])
                    new_combinations.append(this_param_combinations)
            return new_combinations
    
        res = []
        one_param_name = measure_params_order[param_n]
        one_param_values = params_dict.pop(one_param_name)
        for v in one_param_values:
            kwargs_copy[one_param_name] = v
            res.extend(inner_make_params_combinations(copy.deepcopy(params_dict), param_n=param_n+1, **kwargs_copy))
        return res
    
    return inner_make_params_combinations(measure_params)

def preprocess_measure_input_params(accepted_measure_parameters, supplied_measure_parameters):
    """Pre-processes and checks the validity of measure parameters.
    
    This function checks, that the supplied parameters (second argument) respect
    the specification given in first argument. The function does not return any
    data, but modifies the supplied_measure_parameters argument in-place. It
    processes the measure parameters so that they fulfill the following
    constraints:
    
        * For each parameter that is not supplied, the default value is used
          (as specified in accepted_measure_parameters)
        * For each supplied parameter, verify the type (specified in
          accepted_measure_parameters). If a single value was provided, create a
          one-element list from it.
        * Special case for the `'nb_iters'` parameter: if it is not present, a
          default one is created (a dict indexed by values of the `'nb_nodes`'
          parameter, with number of iterations in values). It if exists, its
          type is checked (should be a dict).
        * Special case for the `'nb_iters_same_topo_graph'` parameter: if it is not present, a
          default one is created (a list containing dicts indexed by values of
          the `'nb_nodes`' parameter, with number of iterations in values). It
          if exists, its type is checked (should be a list of dict).
    
    Args:
        accepted_measure_parameters (dict): a dict specifying in keys the name of the accepted parameters (in the form of strings), 
            and in value a 2-tuple (type, value) representing the expected type and default value of the parameter. The first element 
            of the tuple can be an iterable, containing several types, meaning that the parameter is allowed to have several types.
        supplied_measure_parameters (dict): a dict with parameter names as keys (in the form of strings), and any data in values.

    Raises:
        :exc:`.MeasureException`: if supplied_measure_parameters contains a
        paramter not allowed according to accepted_measure_parameters, or if a
        parameter is not of the expected type
    """
    all_accepted_parameters = list(accepted_measure_parameters.keys()) + ['nb_iters', 'nb_iters_same_topo_graph']
    if any((k not in all_accepted_parameters for k in supplied_measure_parameters.keys())):
        raise MeasureException("Wrong measure parameter(s) supplied: {}".format(",".join(["'"+str(k)+"'" for k in supplied_measure_parameters.keys() if k not in all_accepted_parameters])))
    
    for p_name, (p_type, p_default_value) in accepted_measure_parameters.items():
        if p_name not in supplied_measure_parameters:
                supplied_measure_parameters[p_name] = p_default_value
        elif (not isinstance(supplied_measure_parameters[p_name], p_type) and 
              (not isinstance(supplied_measure_parameters[p_name], list) or not all(isinstance(v, p_type) for v in supplied_measure_parameters[p_name]))):
            try:
                types = iter(p_type)
            except:
                types = [p_type] 
            raise MeasureException("Wrong parameter type for '{}', expected '{}' (or list of such elements)".format(p_name, "' or '".join([str(t.__name__) for t in types])))
        
        if not isinstance(supplied_measure_parameters[p_name], list):
            supplied_measure_parameters[p_name] = [supplied_measure_parameters[p_name]]
        supplied_measure_parameters[p_name].sort()
        
    # Always treat the case of 'nb_iters' and 'nb_iters_same_topo_graph'
    if 'nb_iters_same_topo_graph' not in supplied_measure_parameters:        
        supplied_measure_parameters['nb_iters_same_topo_graph'] = [defaultdict(lambda: 2)]
    elif isinstance(supplied_measure_parameters['nb_iters_same_topo_graph'], int):
        aux = supplied_measure_parameters['nb_iters_same_topo_graph'] 
        supplied_measure_parameters['nb_iters_same_topo_graph'] = [defaultdict(functools.partial(lambda x: x, aux))]
    elif isinstance(supplied_measure_parameters['nb_iters_same_topo_graph'], dict):
        supplied_measure_parameters['nb_iters_same_topo_graph'] = [supplied_measure_parameters['nb_iters_same_topo_graph']]
    elif isinstance(supplied_measure_parameters['nb_iters_same_topo_graph'], list) and all(isinstance(x, (int, dict)) for x in supplied_measure_parameters['nb_iters_same_topo_graph']):
        for i in range(len(supplied_measure_parameters['nb_iters_same_topo_graph'])):
            if isinstance(supplied_measure_parameters['nb_iters_same_topo_graph'][i], int):
                aux = supplied_measure_parameters['nb_iters_same_topo_graph'][i] 
                supplied_measure_parameters['nb_iters_same_topo_graph'][i] = defaultdict(functools.partial(lambda x: x, aux))
    else:#if  not isinstance(supplied_measure_parameters['nb_iters_same_topo_graph'], (dict, defaultdict))  or not all(isinstance(v, int) for v in supplied_measure_parameters['nb_iters_same_topo_graph']):
        raise MeasureException("The parameter 'nb_iters_same_topo_graph' is not of the expected form (int or dict, or list of (int or dict) indexed by nb_nodes)")
    
    if 'nb_iters' not in supplied_measure_parameters:        
        supplied_measure_parameters['nb_iters'] = {}
        for n in supplied_measure_parameters['nb_nodes']:
            supplied_measure_parameters['nb_iters'][n] = max(2, math.ceil(154.685 * math.exp(-0.029342 * n)))
            # This formula is an exponential fit of {{5, 130}, {7, 125}, {10,
            # 120}, {30, 65}, {50, 40}, {75, 10}, {100, 5}, {150, 3}}, a set of
            # nb_iters put by hand
            # The goal is to keep the time of  simulation runs manageable: The 
            # more nodes there are, the less iteration we make 
            # (because the simulation time augments exponentially with nodes...) 
    elif isinstance(supplied_measure_parameters['nb_iters'], int):
        aux = supplied_measure_parameters['nb_iters']
        supplied_measure_parameters['nb_iters'] = defaultdict(functools.partial(lambda x: x, aux))
    elif  not isinstance(supplied_measure_parameters['nb_iters'], (dict, defaultdict)) or not all(isinstance(v, int) for v in supplied_measure_parameters['nb_iters']):
        raise MeasureException("The parameter 'nb_iters' is not of the expected form (int or dict of int indexed by nb_nodes)")


__logger = None
def _get_logger(log_to_file=None):
    global __logger
    if __logger is None:
        
        __logger = logging.getLogger(MEASURE_LOGGER_NAME)
        __logger.propagate = False
        __logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(levelname)s | Measures (%(asctime)s) | %(message)s")
        handler1 = logging.StreamHandler(sys.stdout)
        handler1.setFormatter(formatter)
        __logger.addHandler(handler1)
        
        if log_to_file:
            folder, _a = os.path.split(log_to_file)
            if not os.path.isdir(folder):os.makedirs(folder, mode=0o775)
            handler2 = logging.FileHandler(log_to_file)
            handler2.setFormatter(formatter)
            __logger.addHandler(handler2)
    
    return __logger
    

class MeasureException(Exception):
    """Raised when an error occurs during a measure or network run, or when measure parameters are not well-formed"""
    pass