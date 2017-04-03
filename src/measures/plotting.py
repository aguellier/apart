# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
This module provides helper function to plot graphs constructed from the results
of simulations. It uses the :mod:`matplotlib.pyplot` module for all graph creations.

Note that, for the plotting of complex/custom graphs, the generic functions
provided in this module may not be flexible enough. However, the code for custom
function plotting can be inspired by the code in this module.
'''
from collections import defaultdict, OrderedDict
import itertools
import math
import os
import statistics

from common.utilities import print_recursive_structure, make_hashable
from measures.file_handling import save_graphs_simulation_results
from measures.network_statistics import RunningStat
import matplotlib.pyplot as plt
import dill as pickle

def plotfunc(measure_params):
    """Decorator for plotting functions.
    
    Modifies the behavior of a plotting function in the following way. If no
    'x_param' argument is given to the function, then the function is called on
    each parameter in `measure_params`.
    
    Args:
        measure_params (list of string): the list of the (names of) `x_param` parameters on which to call the decorated plotting function.
    """ 
    def actual_decorator(func):
        def new_plotfunc(*args, **kwargs):
            if 'x_param' not in kwargs or kwargs['x_param'] is None:
                for x_param in measure_params:
                    kwargs['x_param'] = (x_param, x_param)
                    func(*args, **kwargs)
                return
            elif isinstance(kwargs['x_param'], list):
                for x_param in kwargs['x_param']:
                    kwargs['x_param'] = x_param
                    func(*args, **kwargs)
                return
            else:
                func(*args, **kwargs)
        return new_plotfunc
    return actual_decorator


def process_plot_function_inputs(x_param, y_stat, restrict_to_other_params):
    """Checks and pre-processes inputs to a plotting function.
    
    Args:
        x_param (tuple): a tuple of strings, containing the name of the 
            parameter to put in abscissa of the graph, accompanied with label of
            the axis, plus optionally a label for the file name
        y_stat (tuple): similarl to `x_param`, but for the ordinate axis
        restrict_to_other_params (dict): a dict with parameter names as keys,
            and strings or lists of strings as values.
    
    Returns:
        (tuple, tuple, list): The processed input parameters
    """
        
    if (not isinstance(x_param, tuple) or not 2 <= len(x_param) <= 3 
        or not all(isinstance(v, str) for v in x_param)):
        raise PlottingException("Error : The function expects a parameter `x_param` as a tuple ('param reference name', 'param text name'[, 'param_name_for_file_name']).") 
    if (not isinstance(y_stat, tuple) or not 2 <= len(y_stat) <= 3 
        or not all(isinstance(v, str) for v in y_stat)):
        raise PlottingException("Error : The function expects a parameter `y_stat` as a tuple ('stat reference name', 'stat text name'[, 'stat_name_for_file_name']).") 
    
    if len(x_param) == 2:
        x_param = (x_param[0], x_param[1], x_param[0])
    
    if len(y_stat) == 2:
        y_stat = (y_stat[0], y_stat[1], y_stat[0])
    
    for p in restrict_to_other_params:
        if not isinstance(restrict_to_other_params[p], list):
            restrict_to_other_params[p] = [restrict_to_other_params[p]]
            
    return x_param, y_stat, restrict_to_other_params
    

def plot_simple_metric(experiments_results, ordered_measure_params_names, y_stat, x_param, several_curves=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    """Generic plotting function, producing a bar plot of one or several curves, based on data points.
    
    Produces a graph and saves it as .png file (or disaplys them on screen).
    This function is designed for plotting *simple* data; namely, simple curves
    made of data points. It accepts the results of simulations in arguments,
    processes these results, and plots them.
    
    Note that this function actually calls :func:`.do_plot_simple_metric` to
    plot the graph. Its main purpose is only to process the data from measures
    and simulations into input acceptable by the latter function. 
    
    Args:
        experiment_results (dict): typically, the output of a call to :func:`measures.common_measures.generic_measure`.
        ordered_measure_params_names (list of string): the names of the parameters of the measure. 
            The order of these parameters is preserved, but matters only in case
            they are displayed to the user (the function does not rely on that
            order to function).
        y_stat (tuple): the statistic to plot (i.e. the metric to plot in ordinate axis). Is processed by a call to :func:`.process_plot_function_inputs`.
        x_param (tuple): the parameter to use in abscissa. Is processed by a call to :func:`.process_plot_function_inputs`. (Default: None)
        several_curves (list of string, optional): if None, then only one curve is drawn in the 
            graph. If provided as a list of string, draws as many curves as
            there are strings in the list, using these strings as the curves labels.
        restrict_to_other_params (dict): a dict with parameter names as keys,
            and strings or lists of strings as values. This argument is similat
            to that of
            :func:`~measures.common_measures.generic_measure`,
            and allows to include in the graph only results for a certain subset
            of results.
        save_to (string, optional): path of the folder in which graphs should be saved. If `None`, 
            the graphs are instead displayed on screen. (Default: None)
        overwrite_existing (bool, optional): if graphs are saved and this argument is `True`, then the new graph may erase existing files. 
            This argument is passed down to :func:`~measures.file_handling.save_graphs_simulation_results`. (Default: False)
    """
    
    x_param, y_stat, restrict_to_other_params = process_plot_function_inputs(x_param, y_stat, restrict_to_other_params)
    x_param, x_param_label, x_param_file_name = x_param
    y_stat, y_stat_label, y_stat_file_name = y_stat
    
    # Not possible to make a graph with ocom_session as x axis
    if x_param == 'ocom_sessions':
        return
    
    if save_to is not None:
        save_to = os.path.join(save_to, 'vs_{}'.format(x_param_file_name))
    
    
    graph_description = {'x_label': x_param_label, 'y_label': y_stat_label, 
                                  'linestyle': '-'}
    
    if several_curves is None:
        all_possible_graphs_values = defaultdict(lambda: defaultdict(lambda: {'means': [], 'variances': []}))
    else:
        all_possible_graphs_values = defaultdict(lambda: dict((k, defaultdict(lambda: {'means': [], 'variances': []}))  for k in several_curves))
    
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
        x_value = make_hashable(x_value)
#         other_measure_params = tuple(sorted(((p, v) for p, v in experiment['measure_params'].items() if p != x_param and p != 'topology_graph'), 
#                                             key=lambda x: _MEASURE_PARAMS.index(x[0])))
        other_measure_params = list(sorted(((p, v) for p, v in experiment['measure_params'].items() if p != x_param and p in ordered_measure_params_names), 
                                            key=lambda x: ordered_measure_params_names.index(x[0])))
        other_measure_params.append(('nb_iters', experiment['measure_params']['nb_iters']))
        other_measure_params = make_hashable(other_measure_params)
        stat = experiment['results']['overall'][y_stat]
        
        if several_curves is None: 
            all_possible_graphs_values[other_measure_params][x_value]['means'].append(stat.mean)
            all_possible_graphs_values[other_measure_params][x_value]['variances'].append(stat.variance)
        else:
            for curve in several_curves: 
                all_possible_graphs_values[other_measure_params][curve][x_value]['means'].append(stat[curve].mean)
                all_possible_graphs_values[other_measure_params][curve][x_value]['variances'].append(stat[curve].variance)
    

    # Each graph must show a statistic (in y axis) according to a network
    # parameter (in x axis), where **only** this network parameter varies. Thus,
    # we make one graph per set of other network parameters
    all_possible_graphs_values = OrderedDict(sorted(all_possible_graphs_values.items(), key=lambda x: x[0]))
#     print_recursive_structure(all_possible_graphs_values)
    
    i = 0
    for other_measure_params, graph_values in all_possible_graphs_values.items():

        if several_curves is None:
            sorted_zipped_data = sorted(((k, statistics.mean(v['means']), math.sqrt(statistics.mean(v['variances'])) if math.sqrt(statistics.mean(v['variances'])) > 0 else (0 if len(v['means']) < 2 else statistics.stdev(v['means']))) 
                                        for k, v in graph_values.items())
                                        , key=lambda x: x[0])
        
            x_values, y_values, stdevs = map(list, zip(*sorted_zipped_data))
        else:
            x_values = {}
            y_values = {}
            stdevs = {}
            for curve, curve_values in graph_values.items():
                sorted_zipped_data = sorted(((k, statistics.mean(v['means']), math.sqrt(statistics.mean(v['variances']))) for k, v in curve_values.items())
                                        , key=lambda x: x[0])
            
                x_values[curve], y_values[curve], stdevs[curve] = map(list, zip(*sorted_zipped_data))
        
        graph_description['title'] = ",\n".join(map(lambda v: str(v[0])+": "+str(v[1]), other_measure_params))
        if save_to:
            graph_description['file_name'] = "{}_vs_{}".format(y_stat_file_name, x_param_file_name).lower() + "_"+str(i)
            i+=1
        
        # Actually plot the graph
        do_plot_simple_metric(x_values, y_values, stdevs, graph_description, several_curves=several_curves, save_to=save_to, overwrite_existing=overwrite_existing)
        
        if save_to:
            # Additionally, if there is one value only in the grpah, we log that value in a txt file
            if isinstance(x_values, list) and len(x_values) == 1 or isinstance(x_values, dict) and len(next(iter(x_values.values()))) == 0:
                with open(os.path.join(save_to, graph_description['file_name']+".txt"), 'w') as f:
                    f.write("Other params : {}\n".format(other_measure_params))
                    if isinstance(x_values, list):
                        f.write("For {} = {}, {} = {} with stdev = {}".format(x_param_label, x_values[0], y_stat_label, y_values[0], stdevs[0]))
                    else:
                        for curve in several_curves:
                            f.write("Curve {} : for {} = {}, {} = {} with stdev = {}".format(curve, x_param_label, x_values[curve][0], y_stat_label, y_values[curve][0], stdevs[curve][0]))
        
        
def do_plot_simple_metric(x_values, y_values, stdevs, graph_description, several_curves=None, save_to=None, overwrite_existing=False, color='b', marker='s', marker_size=6, linestyle='-', linewidth=1, legend_placement='upper right', xlim=None, ylim=None):
    """Actually plots a *simple metric*,  producing a bar plot of one or several curves, based on data points.
    
    This function is called by :func:`.plot_simple_metric`, but provided in a
    stand-alone mode for a more fine-grained control of its inputs.
    
    Args:
        x_values (list of numbers, or dict): the values in abscissa. If `several_curves` is not None, expected to be a dict indexed by the label of the curves, with lists of numbers in values.
        x_values (list of numbers, or dict): the values in ordinate. If `several_curves` is not None, expected to be a dict indexed by the label of the curves, with lists of numbers in values. 
        several_curves (list of string, optional): if None, then only one curve is drawn in the 
            graph. If provided as a list of string, draws as many curves as
            there are strings in the list, using these strings as the curves labels.
        save_to (string, optional): path of the folder in which graphs should be saved. If `None`, 
            the graphs are instead displayed on screen. (Default: None)
        overwrite_existing (bool, optional): if graphs are saved and this argument is `True`, then the new graph may erase existing files. 
            This argument is passed down to :func:`~measures.file_handling.save_graphs_simulation_results`. (Default: False)
        others (color, marker, marker_size, linestyle, linewidth, legend_placement, xlim, ylim): parameters passed down to the :mod:`matplotlib.pyplot` module. 
            Expected to be simple values when `several_curves` is None, and list of values otherwise.  
    """
    
    plt.clf()
    if several_curves is None:
        plt.errorbar(x_values, y_values, yerr=stdevs, color=color, linewidth=linewidth, linestyle=linestyle, marker=marker)
    else:
        if isinstance(marker, dict):
            curves_markers = marker
        elif isinstance(marker, list):
            markers = itertools.cycle(marker)
            curves_markers = dict((c, next(markers)) for c in several_curves)
        else:
            curves_markers = defaultdict(lambda: marker)
        if isinstance(color, dict):
            curves_colors = color
        elif isinstance(color, list):
            colors = itertools.cycle(color)
            curves_colors = dict((c, next(colors)) for c in several_curves)
        else:
            curves_colors = defaultdict(lambda: color)
        if isinstance(linewidth, list):
            linewidths = itertools.cycle(linewidth)
            curves_linewidth = dict((c, next(linewidths)) for c in several_curves)
        else:
            curves_linewidth = defaultdict(lambda: linewidth)
        for curve in several_curves:
            plt.errorbar(x_values[curve], y_values[curve], yerr=stdevs[curve], color=curves_colors[curve], linewidth=curves_linewidth[curve], linestyle=linestyle, marker=curves_markers[curve], markersize=marker_size, label=curve)
            plt.legend(loc=legend_placement)
    
    
    plt.xlabel(graph_description['x_label'])
    plt.ylabel(graph_description['y_label'])
    plt.title(graph_description['title'])
    plt.grid(True)
    if xlim:
        plt.xlim(*xlim)
    else:
        xmin, xmax = plt.xlim()
        plt.xlim(xmin * 0.99, xmax * 1.01)
    if ylim:
        plt.ylim(*ylim)
    
    if save_to:
        file_name = graph_description['file_name']
        save_graphs_simulation_results(plt, file_name=file_name, folder=save_to, overwrite_existing=overwrite_existing)
    else:
        plt.show()


def plot_histogram(experiments_results, ordered_measure_params_names, y_stat, print_yerr=False, label_rotation=0, xticks=None, max_bars=None, restrict_to_other_params={}, save_to=None, overwrite_existing=False):
    """Analogous to :func:`.plot_simple_metric`, but for plotting histograms.
    
    Args:
        experiment_results (dict): typically, the output of a call to :func:`measures.common_measures.generic_measure`.
        ordered_measure_params_names (list of string): the names of the parameters of the measure. 
            The order of these parameters is preserved, but matters only in case
            they are displayed to the user (the function does not rely on that
            order to function).
        y_stat (tuple): the statistic to plot (i.e. the metric to plot in ordinate axis). Is processed by a call to :func:`.process_plot_function_inputs`.
        restrict_to_other_params (dict): a dict with parameter names as keys,
            and strings or lists of strings as values. This argument is similat
            to that of
            :func:`~measures.common_measures.generic_measure`,
            and allows to include in the graph only results for a certain subset
            of results.
        save_to (string, optional): path of the folder in which graphs should be saved. If `None`, 
            the graphs are instead displayed on screen. (Default: None)
        overwrite_existing (bool, optional): if graphs are saved and this argument is `True`, then the new graph may erase existing files. 
            This argument is passed down to :func:`~measures.file_handling.save_graphs_simulation_results`. (Default: False)
        print_yerr (bool, optional): if `True`, the standard deviation of each bar is printed (Default: False)
        max_bars (int, optional): if set to `n` and the histogram has more than n different x values, the histogram is cut down to n bars. 
            And on the last bar are stacked and accumulatged all the remaining values. This avoids huge and unreadable bar plots. 
        others (label_rotation, xticks): parameters passed down to the :mod:`matplotlib.pyplot` module.   
    
    """
    _, y_stat, restrict_to_other_params = process_plot_function_inputs(("", ""), y_stat, restrict_to_other_params)
    y_stat, y_stat_label, y_stat_file_name = y_stat
    
    graph_description = {'x_label': "", 'y_label': y_stat_label}
    all_possible_histograms = defaultdict(lambda: defaultdict(lambda: {'means': [], 'variances': []}))
    for experiment in experiments_results:
        # Keep the experiment results only if they fit into the asked other parameters
        skip = False
        for p, v_list in restrict_to_other_params.items():
            if  experiment['measure_params'][p] not in v_list:
                skip = True
                break
        
        if skip:
            continue
        
        measure_params = list(sorted(((p, v) for p, v in experiment['measure_params'].items() if p in ordered_measure_params_names), key=lambda x: ordered_measure_params_names.index(x[0])))
        measure_params.append(('nb_iters', experiment['measure_params']['nb_iters']))
        measure_params = make_hashable(measure_params)
        hist_dict = experiment['results']['overall'][y_stat]
        for k,v in hist_dict.items():
            # DEBUG : compatibility with previous way of storing stats
            if isinstance(v, RunningStat):
                all_possible_histograms[measure_params][k]['means'].append(v.mean)
                all_possible_histograms[measure_params][k]['variances'].append(v.variance)
            else:
                all_possible_histograms[measure_params][k]['means'].append(v)
                all_possible_histograms[measure_params][k]['variances'].append(0)
    
    # Each graph must show a statistic (in y axis) according to a network
    # parameter (in x axis), where **only** this network parameter varies. Thus,
    # we make one graph per set of other network parameters
    all_possible_histograms = OrderedDict(sorted(all_possible_histograms.items(), key=lambda x: x[0]))
    
    i = 0
    for measure_params, hist_dict in all_possible_histograms.items():
        # Aggregate the results
        hist_dict = dict((k, {'mean': statistics.mean(v['means']), 'variance': statistics.mean(v['variances'])}) for k, v in hist_dict.items())
        
        normaliser = sum((v['mean'] for v in hist_dict.values()))
        hist_dict = OrderedDict(sorted(((k, {'mean': v['mean']/normaliser, 'stdev': math.sqrt(v['variance'])/normaliser}) for k, v in hist_dict.items()), key=lambda x: x[0]))
        
        # If there is only one histogram to draw, then record it at this point
        if save_to is not None and len(all_possible_histograms) ==1:
            if not os.path.exists(save_to):
                os.makedirs(save_to, mode=0o775)
            with open(os.path.join(save_to, y_stat_file_name.lower()+".pickle"), 'wb') as f:
                pickle.dump(hist_dict, f)
            
        
        # Cut the dict if its len is > to max_bars
        too_many_bars = False
        if max_bars is not None and len(hist_dict) > max_bars:
            too_many_bars = True
            rest_hist_dict = hist_dict
            hist_dict = OrderedDict(rest_hist_dict.popitem(last=False) for _ in range(max_bars))
        
        if print_yerr:
            yerr = [v['stdev'] for v in hist_dict.values()]
        else:
            yerr = None
            
        if xticks is None:
            xticks = [k for k in hist_dict.keys()]        
        if too_many_bars and not str(xticks[-1]).endswith('+'):
            xticks[-1] = "{}+".format(xticks[-1])
        
        x_values = list(range(len(hist_dict)))
        y_values = [v['mean'] for v in hist_dict.values()]
        
        
        plt.clf()
        plt.bar(x_values, y_values, yerr=yerr, align='center', width=0.5, ecolor='#EE1111')
        if too_many_bars:
            last_hist_key, last_value_height = hist_dict.popitem(last=True)
            last_value_height = last_value_height['mean']
            for k, v in rest_hist_dict.items():
#                 yerr = v['stdev'] if print_yerr else None
                plt.bar([last_hist_key], [v['mean']], align='center', width=0.5, bottom=last_value_height, lw=0.1, color='#AAAAFF', ecolor='#FFAAAA')
                last_value_height += v['mean']

            
        
        plt.xticks(x_values, xticks, rotation=label_rotation)
        plt.xlim(x_values[0]-0.5, x_values[-1]+0.5)
        plt.ylim(0, 1)
        
        plt.ylabel(graph_description['y_label'])
        plt.title( ",\n".join(map(lambda v: str(v[0])+": "+str(v[1]), measure_params)))
        plt.grid(True)
        
        if save_to:
            file_name = "{}".format(y_stat_file_name).lower() + "_"+str(i)
            save_graphs_simulation_results(plt, file_name=file_name, folder=save_to, overwrite_existing=overwrite_existing)
            i += 1
        else:
            plt.show()
        
        
        

class PlottingException(Exception):
    """Raised when an error occurs during the formatting of data into a graph"""
    pass