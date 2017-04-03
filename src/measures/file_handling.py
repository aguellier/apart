# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
This module provides helper function to manipulate the files that store
serialized measure results and graphs. It allows to check the existence of
folders and to harmonize the file names.
'''

import os
import time

from common.custom_logging import *
import dill as pickle


def save_measures_results(sims_outputs, folder=None, file_name=None, overwrite_existing=False):
    """This function saves the output produced by measures by pickling the results into a file on the disk.
    
    The folder and file_name arguments are passed on to
    :func:`.prepare_file_name`, with default folder /tmp/protocol_stats/.
    
    Args:
        sims_outputs (any): the results of the measures and simulations. Must be serializable with the `:mod:`dill` package.
        folder (string, optional): the folder in which to place the binary files of the pickled object (Default: None)
        file_name (string, optional): the name of the file to save the data to (Default: None)
        overwrite_existing (bool, optional): whether, if the file_name provided already exists in the specified folder, it should be overwritten or not (default: False)
    
    Returns 
        sim_outputs
    """
    file_path = prepare_file_name(folder, file_name, default_folder='/tmp/protocol_stats/', overwrite_existing=overwrite_existing, extension=".pickle")
   
    print("Saving into {}".format(file_path))
    
    with open(file_path, 'wb') as f:
        pickle.dump(sims_outputs, f)

    return sims_outputs

def load_measures_results(folder, file_name):
    """Load simulation results previously saved (pickled) into a file and returns the contents.
    
    It is allowed to specify the absolute path, file name included, in the
    file_name argument (that is, folder may be an empty string). The converse is
    also true: the file name argument can be empty if provided in the folder
    one.
    
    Args:
        folder (string): the folder in which to search for the binary files of the pickled.
        file_name (string): the name of the file to load (typically, a .pickle). 
    
    Returns:
        any: the unpickled contents of the file(s) specified 
    """

    complete_file_path = os.path.abspath(os.path.join(folder, file_name))

    with open(complete_file_path, 'rb') as f:
        sims_outputs = pickle.load(f)

    return sims_outputs

def save_graphs_simulation_results(plot_object, folder=None, file_name=None, overwrite_existing=False):
    """Save a generated graphs in a png file.
    
    The folder and file_name arguments are passed on to
    :func:`.prepare_file_name`, with default folder /tmp/protocol_graphs/.
    
    
    Arguments:
        plot_object: a matplotlib object on which savefig() can be called, and containing the graph to save
        folder (string, optional): the folder in which to save the graph (Default: None)
        file_name (string, optional: the name of the file to save the graph to (Default: None)
        overwrite_existing (bool, optional): whether, if the file_name provided already exists in the specified folder, it should be overwritten or not (default: False)
    """
    file_path = prepare_file_name(folder, file_name, default_folder='/tmp/protocol_graphs/', extension=".png", overwrite_existing=overwrite_existing)
    print("Saving graph '{}' to '{}'".format(file_name, file_path))
    plot_object.savefig(file_path,bbox_inches='tight')

def save_network_state(network_manager, folder=None, file_name=None, overwrite_existing=False):
    """Pickle and save on disk a :obj:`~apart.core.network.NetworkManager` instance.
    
    Arguments:
        network_manager (:obj:`~apart.core.network.NetworkManager`): the network manager object, containing all the network state.
        folder (string, optional): the folder in which to save the network state (Default: None)
        file_name (string, optional: the name of the file to save the network state to (Default: None)
        overwrite_existing (bool, optional): whether, if the file_name provided already exists in the specified folder, it should be overwritten or not (default: False)
        
    Returns:
        network_manager
    """
    file_path = prepare_file_name(folder, file_name, default_folder='/tmp/protocol_net_states', extension=".pickle", overwrite_existing=overwrite_existing)

    # Pickle in file
    with open(file_path, 'wb') as f:
        pickle.dump(network_manager, f)
        
    return network_manager



def format_results_file_name(measure_title, combination_number, nb_iters):
    """Formats the file name for the results of simulations
    
    Args:
        measure_title (string): the name of the measure/experimentations
        combination_number (int): the counter of the network simulations (useful when many network simulations are made, with may different parameter combinations)
        nb_iters (int): the number of iterations that were run for each combination
    
    Returns:
       string: The file name formatted using the provided arguments.
    """
    return "{}_combinaison_{}_{}_iterations.pickle".format(measure_title, combination_number, nb_iters)

def matchstring_results_file_name(measure_title):
    """Gives the regex string of the file name for simulation results.
    
    This function is useful to allow the search of files on the disk, and at the
    same time get information on which parameter combination these results refer
    to.
    
    Args:
        measure_title (string): the name of the measure/experimentations

    Returns:
        string: The regex string where the combination counter and number of iterations are left as `(\d+)` allowing to catch them in a Python :obj:`match` object. 
    """
    # This function can be used to match old file naming conventions
    return "{}_combinaison_(\d+)_(\d+)_iterations".format(measure_title)

def format_net_states_file_name(measure_title, combination_number, iter_number):
    """Formats the file name for the saving of network states.
    
    Args:
        measure_title (string): the name of the measure/experimentations
        combination_number (int): the counter of the network simulations (useful when many network simulations are made, with may different parameter combinations)
        iter_number (int): the counter of the specific iteration being saved 
    
    Returns:
        string: The file name formatted using the provided arguments.
    """
    
    return "{}_netstate_{}_iteration_{}.pickle".format(measure_title, combination_number, iter_number)
 
def matchstring_net_states_file_name(measure_title):
    """Gives the regex string of the file name for simulation results.
    
    This function is analogous to :func:`.matchstring_results_file_name`.
    
    Args:
        measure_title (string): the name of the measure/experimentations

    Returns:
        string: The regex string where the combination counter and iteration counter are left as `(\d+)` allowing to catch them in a Python :obj:`match` object. 
    """
    return "{}_netstate_(\d+)_iteration_(\d+)".format(measure_title)


def prepare_file_name(folder, file_name, default_folder='/tmp/', default_file_name="unknown", extension=".pickle", overwrite_existing=False):
    """Processes a file name and folder and outputs a canonic absolute path.
    
    This function takes an absolute path, under the form of a folder and a file
    name (note: the full path can be in one of the two, the other one being an
    empty string or None), and processes this path so that: if no file name is
    provided, the default file name is used; if no folder is provided, the
    default one is used; if the file name does not have an extension (or an
    invalid one), append the provided extension to the file name; if the new
    file should not overwrite and existing one, then append a random string to
    the file name.
    
    This function is meant to process the file names of files for graphs,
    simulation results, and network states alike.
    
    This function also creates the folder(s) if necessary.
    
    Args:
        folder (string): the folder in which the file should be saved.
        file_name (string): the name of the file that must be saved.
        default_folder (string, option): The default folder, if no folder is provided in argument (Default: /tmp)
        default_file_name (string, option): The default file name, if no folder is provided in argument (Default: unknown)
    
    Returns:
        string: a well formated full absolute path, where it is ensured that the folders in the basename all exist. 
    """
    if folder is None and file_name is None:
        folder = default_folder
        file_name = default_file_name+extension  
    
    
    if file_name is None:
        file_name = default_file_name+extension

    if not file_name.endswith(extension):
        file_name += extension  

    folder, file_name = os.path.split(os.path.join(folder, file_name))

    # If the folder does not exist, create it
    if not os.path.isdir(folder):
        os.makedirs(folder, mode=0o775)
    
    complete_file_path = os.path.abspath(os.path.join(folder, file_name))
    
    while not overwrite_existing and os.path.isfile(complete_file_path):
        complete_file_path += str(round(time.time() * 1000))+extension
    
    return complete_file_path

















