# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
This module represents the *hook* of the :mod:`~measures` package into the
simulation. Indeed, the class :class:`.NetworkStats` that it provides is used by
the network manager and the nodes during the simulation. More accurately, the
latter store information in a :class:`.NetworkStats` instance.

This module also provides two classes to ease the computation of means and
standard deviations, :class:`.RunningStat` and :class:`.RegulatStat`.  
'''
import abc
from collections import defaultdict
import itertools
import math
import statistics

import apart
from common.custom_logging import *
from apart.core.messages import MsgFlag
from apart.core.protocol_constants import RtPolicyReason
import dill as pickle


class NetworkStats(object):

    def __init__(self, net_manager, nb_nodes, sim_params):
        """An instance of this class gathers all statistics and information about a given network run.
        
        Each simulation/network run is associated with one instance of this
        class. More accurately, a :obj:`~apart.core.network.NetworkManager`
        always has a :obj:`.NetworkStats` instance as attribute, that it gives
        to the network and to nodes so that the latter fills it up with
        information and statistics about the network run.
        
        .. note:: This class defines attributes dynamically, depending one the values of
                the booleans in the :obj:`~apart.simulation.SimulationParams` instance
                related to the network run. For instance, if
                :attr:`~apart.simulation.SimulationParams.log_route_props` is `True`,
                then :obj:`~measures..network_statistics.NetworkStats`
                will have the attribute 
                :obj:`~measures.network_satistics.NetworkStats.nb_route_props_per_node`. 
                Otherwise, accessing this attribute will trigger an :exc:`AttributeError`.
    
        In practice, this class contains attributes that are filled up by the
        network manager, the network, and nodes. These includes, *e.g.* the
        number of messages sent/received by nodes, the number of route
        proposals, the total delay of oriented communications, etc. To avoid too
        heavy memory consumption, these pieces of information are often gathered
        and average *on the fly*, using a :obj:`.RunningStat` instance. Average
        often happens on nodes on the network, on on rounds. See each attribute
        for details.
        
        The methods of these class are the entry points for the network and the
        nodes to input information about the network into a :obj:`.NetworkStats`
        instance. The functions only logs information in existing attributes.
        """
        
        self._net_manager = net_manager
        self.network_start_time = 0
        """float: The time at which the network run was started, based on a call to `time.time()` (*i.e.* based on the system time, not SimPy time)"""
        self.network_end_time = 0
        """float: The time at which the network run ended, based on a call to `time.time()` (*i.e.* based on the system time, not SimPy time)"""
        
        if sim_params.log_ocom_latency:
            self.ocom_latency = RunningStat()
            """:obj:`.RunningStat`: The latency, in rounds, between the initialization of an ocom, and the receiving of the last message by the end-receiver. 
            
            Average over all oriented communications in the network.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_ocom_latency` is `True`.
            """
            self.ocom_latency_after_init = RunningStat()
            """:obj:`.RunningStat`: The latency, in rounds, between the sending of the first payload message of the ocom, and the receiving of the last one 
            
            Average over all oriented communications in the network
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_ocom_latency` is `True`.
            """

        if sim_params.log_end_topo_diss:
            self.end_topo_diss = -1  
            """int: The round number at which the topology is fully disseminated.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_end_topo_diss` is `True`.
            """
            
        if sim_params.log_end_ocom_phase:
            self.end_ocom_phase = -1  
            """int: The round number at which the oriented communication phase is finished.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_end_ocom_phase` is `True`.
            """
        
                
        if sim_params.log_and_store_all_real_msgs: 
            self.sent_link_msgs_per_node_per_round = defaultdict(lambda: defaultdict(list)) # OK
            """:obj:`collections.defaultdict`: All sent link messages by each node at each round.
            
            The dict is indexed by nodes, its values are dict, which indexes are
            batching rounds, and values are the messages sent by the node in the
            batching round. That is if d[n][r] = [m1, m2, ...], then node n sent
            the real messages m1, m2, ... in round r.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_and_store_all_real_msgs` is `True`.
            
            .. warning:: Heavy memory consumption
            """
            self.rcvd_link_msgs_per_node_per_round = defaultdict(lambda: defaultdict(list)) # OK
            """:obj:`collections.defaultdict`: All received link messages  by each node at each round.
            
            This dict is defined similarly as
            :attr:`.sent_link_msgs_per_node_per_round`, for received link
            messages.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_and_store_all_real_msgs` is `True`.
            
            .. warning:: Heavy memory consumption
            """
            
        if sim_params.log_sent_link_msgs_per_round:
            self.histogram_nb_sent_link_msgs_per_round_per_neighbor = defaultdict(lambda: 0)
            """:obj:`collections.defaultdict`: The histogram number of link messages sent to each neighbor at each round. Is expected to be one or two, but not much more
            
            Dict indexed by the number of link messages sent per round (all
            nodes confunded), and which values are the number of occurences of
            this number. That is, if d[n] = m, this means that on m occasions, a
            node (any one node in the network) sent n link messages during a round.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_sent_link_msgs_per_round` is `True`.
            """
        
        if sim_params.log_real_link_msgs:
            self.nb_sent_real_msgs_per_node = defaultdict(RunningStat) 
            """:obj:`collections.defaultdict`: Number of real link messages sent by each node at each round. 
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_link_msgs` is `True`.
            """
            self.nb_rcvd_real_msgs_per_node = defaultdict(RunningStat) 
            """:obj:`collections.defaultdict`: Number of real link messages received by each node at each round.
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_link_msgs` is `True`.
            """

        if sim_params.log_real_e2e_msgs:
            self.nb_end_sent_real_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Number of (real) messages end-sent by each node at each round.
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_e2e_msgs` is `True`.
            """
            self.nb_end_rcvd_real_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Number of (real) messages end-received by each node at each round.
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_e2e_msgs` is `True`.
            """
            
            self.nb_end_sent_ocom_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Number of (real) oriented communication message end-sent by each node at each round. 
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_e2e_msgs` is `True`.
            """
            self.nb_end_rcvd_ocom_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Number of (real) oriented communication message end-received by each node at each round.
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_e2e_msgs` is `True`.
            """
        
            self.nb_end_sent_ocom_payload_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Number of (real) oriented communication *payload* message end-sent by each node at each round, i.e. without counting ocom init.
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_e2e_msgs` is `True`.
            """
            self.nb_end_rcvd_ocom_payload_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Number of (real) oriented communication *payload* message end-received by each node at each round, i.e. without counting ocom init. 
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_real_e2e_msgs` is `True`.
            """
        
            
        if sim_params.log_traffic_rates_equilibrium:
            self.traffic_rate_status = {'equilibrium': RunningStat(), 
                                                       'lbound': RunningStat(),
                                                       'ubound': RunningStat()}# OK
            """dict: The equilibirum, minimum and maximum number of real messages that each node can send each round. 
            
            Dict indexed by {equilibrium, lbround, ubound}, which values are :obj:`.RunningStat`
            instances, averaging over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_traffic_rates_equilibrium` is `True`.
            """
        
        if sim_params.log_real_msgs_waiting_time or sim_params.log_histogram_real_msgs_waiting_time:
            self._real_msg_round_timestamp = {}
            """dict: Auxiliary structure to store the round in which a round enters a node's pools. Values from this dict are ``pop()`-ed to compute the delta of rounds afterwards."""
            self.real_msg_waiting_time = RunningStat()
            """:obj:`.RunningStat`: The number of rounds real messages spend waiting in pools.
            
            This value is averaged over all real messages, all nodes, and all rounds.
            
            Defined only if one of or both
            :attr:`~apart.simulation.SimulationParams.log_real_msgs_waiting_time` 
            or :attr:`~apart.simulation.SimulationParams.log_histogram_real_msgs_waiting_time` 
            are `True`.
            """

            self.histogram_real_msg_waiting_time = defaultdict(lambda: 0)
            """:obj:`collections.defaultdict`: The histogram of number of rounds a real messages waits in pools. 
            
            Dict indexed by number of rounds, values are number of occurrences.
        
            Defined only if one of or both
            :attr:`~apart.simulation.SimulationParams.log_real_msgs_waiting_time` 
            or :attr:`~apart.simulation.SimulationParams.log_histogram_real_msgs_waiting_time` 
            are `True`.
            """
            
        
        
        
        if sim_params.log_dummy_link_msgs:
            self.nb_sent_dummy_link_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Total number of dummy link messages sent by each node at each round. 
            
            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_dummy_link_msgs` is `True`.
            """
            self.nb_rcvd_dummy_link_msgs_per_node = defaultdict(RunningStat) # OK
            """:obj:`collections.defaultdict`: Total number of dummy link messages received by each node at each round. 

            Dict indexed by nodes, which values are :obj:`.RunningStat`
            instances, averaging over rounds.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_dummy_link_msgs` is `True`.
            """
            
          

        if sim_params.log_e2e_dummies:
            self.nb_sent_e2e_dummy_msgs = RunningStat() # OK
            """:obj:`.RunningStat`: Total number of end-to-end dummy messages sent by each node at each round.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_e2e_dummies` is `True`.
            """
            self.nb_rcvd_e2e_dummy_msgs = RunningStat() # OK
            """:obj:`.RunningStat`: Total number of end-to-end dummy message received by each node at each round.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_e2e_dummies` is `True`.
            """
            
        if sim_params.log_frequency_batch_intervention:
            self.nb_batch_intervention_remove_reals = RunningStat()
            """:obj:`.RunningStat`: The number of times each node had to modify the batch, by removing real messages.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_frequency_batch_intervention` is `True`.
            """
            
            self.nb_batch_intervention_replace_dummy_by_e2e_dummy = RunningStat()
            """:obj:`.RunningStat`: The number of times each node had to modify the batch, by replacing one dummy by an e2e dummy.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_frequency_batch_intervention` is `True`.
            """
            
            self.nb_batch_intervention_add_e2e_dummy = RunningStat()
            """:obj:`.RunningStat`: The number of times each node had to modify the batch, adding an e2e dummy plus link dummies.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_frequency_batch_intervention` is `True`.
            """  
            
            self.nb_batch_intervention_many_dummies = RunningStat()
            """:obj:`.RunningStat`: The number of times each node had to modify the batch by adding many dummies because no e2e dummy was possible.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_frequency_batch_intervention` is `True`.
            """
            
            self.nb_batch_intervention_default_dummy_bcast = RunningStat()
            """:obj:`.RunningStat`: The number of times each node resorted to the default option of sending one dummy to each neighbor.
            
            This value is averaged over rounds and nodes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_frequency_batch_intervention` is `True`.
            """
        

        if sim_params.log_route_props:
            self.nb_route_props_per_node = dict((reason, defaultdict(lambda: 0)) for reason in itertools.chain(['proposed', 'answered_as_receiver'], RtPolicyReason))
            """dict: The number of route proposals the node was involved in, with which role, and which outcomes. 
            
            Dict doubly indexed. First by type of *route proposal event*
            (`'proposed'`, `'answered_as_receiver'`, plus all
            :class:`~apart.core.protocol_constants.RtPolicyReason`), and then by
            nodes. Values are the number of occurences of the *event*. For
            instance, d['proposed'][n1] = 5 means that node n1 proposed 5
            routes.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_route_props` is `True`.
            """
            
        if sim_params.log_rt_props_latency:
            # The latency of route proposition, between the initial rt prop
            # message to the acceptation/refusal by the proposee
            self.rt_props_latency = RunningStat() # OK
            """:obj:`.RunningStat`: The latency of route proposals between the initial rt prop message to the acceptation/refusal by the proposee.
            
            This value is averaged over every route proposal in the network. 
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_rt_props_latency` is `True`.
            """
            
        if sim_params.log_histogram_real_msgs_per_round:
            self.histogram_nb_real_link_msgs_sent_per_round = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
            """:obj:`collections.defaultdict`: Histogram of real link messages sent per round. 
            
            Dict triply indexed: dict[n][[node][neighbor] = v, means that the
            node *node* sent n real messages to neighbor *neighbor* on v
            occasions during the network run
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_histogram_real_msgs_per_round` is `True`.            
            """  
            self.histogram_nb_real_link_msgs_rcvd_per_round = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
            """:obj:`collections.defaultdict`: Histogram of real link messages received per round. 
            
            Dict triply indexed: dict[n][[node][neighbor] = v, meaning that the
            node "node" received n real messages from neighbor "neighbor" on v
            occasions during the network run.
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_histogram_real_msgs_per_round` is `True`.
            """

        if sim_params.log_ocom_routes:
            self.ocom_routes = {}
            """dict: Minimal description of all oriented communication routes used in the network.
            
            The dict is indexed by oriented communication identifier. For each
            such communication, it specifies the end sender and its first hop,
            plus helper and its first hop (*i.e.* values of the dict are dicts,
            indexed by {'end_sender', 'helper', 'end_rcvr',
            'first_hop_end_sender'}).
            
            Defined only if :attr:`~apart.simulation.SimulationParams.log_ocom_routes` is `True`.
            """
        
        
    def log_end_topo_diss(self, round):
        """Function logging the end of the topology dissemination phase.
        
        Function called by the network manager
        (:obj:`~apart.core.network.NetworkManager`). Modifies attribute
        :attr:`.end_topo_diss` if it exists.
        
        Args:
            round (int): the round number at which the topology dissemination finished.
        """
        try:
            self.end_topo_diss = round
        except AttributeError:
            pass
        
    def log_end_ocom_phase(self, round):
        """Function logging the end of the oriented communication phase.
        
        Function called by the network manager
        (:obj:`~apart.core.network.NetworkManager`). Modifies attribute
        :attr:`.end_ocom_phase` if it exists.
        
        Args:
            round (int): the round number at which the oriented communications finished.
        """
        try:
            self.end_ocom_phase = round
        except AttributeError:
            pass
    
    def log_msg_placed_in_pool(self, node, neighbor, msgs):
        """Function logging the insertion of a message in the pools.
        
        Function called by a node (:obj:`~apart.core.node.Node`).Modifies only
        protected/private attributes.
        
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is inserting the message in its pools.
            neighbor (int): the index of the node's neighbor, specifying in which pool the message is inserted
            msg (:obj:`~apart.core.messages.LinkMsg`): the message inserted in the pool
        """
        try:
            for m in msgs:
                self._real_msg_round_timestamp[m] = node.batching_round
        except AttributeError:
            pass
        
    def log_sent_batch(self, node, batch, nb_reals, nb_dummies, nb_e2e_dummies, remove_real=0, replace_dummy_by_e2e_dummy=0 ,add_e2e_dummy=0, add_many_dummies=0, default_dummy_broadcast=0):
        """Function logging the sending of a batch of messages at the end of a round
        
        Function called by a node (:obj:`~apart.core.node.Node`). Modifies the
        following attributes if they exist: 
        
            * :attr:`.nb_sent_real_msgs_per_node`,
            * :attr:`.nb_end_sent_real_msgs_per_node`,
            * :attr:`.nb_end_sent_ocom_msgs_per_node`,
            * :attr:`.nb_end_sent_ocom_payload_msgs_per_node`,
            * :attr:`.sent_link_msgs_per_node_per_round`,
            * :attr:`.nb_sent_dummy_link_msgs_per_node`,
            * :attr:`.nb_sent_e2e_dummy_msgs`,
            * :attr:`.real_msg_waiting_time`,
            * :attr:`.histogram_real_msg_waiting_time`,
            * :attr:`.nb_batch_intervention_remove_reals`,
            * :attr:`.nb_batch_intervention_replace_dummy_by_e2e_dummy`,
            * :attr:`.nb_batch_intervention_add_e2e_dummy`,
            * :attr:`.nb_batch_intervention_many_dummies`,
            * :attr:`.nb_batch_intervention_default_dummy_bcast`,
            * :attr:`.histogram_nb_real_link_msgs_sent_per_round`,
            * :attr:`.histogram_nb_sent_link_msgs_per_round_per_neighbor`.
                
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is sending the batch (`node.batching_round` gives the current round)
            batch (list of :obj:`~apart.core.messages.LinkMsg`): the batch of message being sent
            nb_reals (int): number of real messages in the batch 
            nb_dummies (int): number of dummy (link) messages in the batch
            nb_e2e_dummies (int): number of end-to-end dummy messages in the batch
            remove_real (int, optional): number of real messages that were manually removed from the batch to fulfill the traffic rates constraints (first type of *batch intervention*)
            replace_dummy_by_e2e_dummy (int, optional): number of times a dummy in the batch was replaced by an end-to-end dummy to fulfill the traffic rates constraints (second type of *batch intervention*)
            add_e2e_dummy (int, optional): number of end-to-end dummy messages added in the batch to fulfill the traffic rates constraints (third type of *batch intervention*) 
            add_many_dummies (int, optional): number of dummy messages added (because, at the time, no end-to-end dummy could be produced) in order to fulfill the traffic rates constraints (fourth type of *batch intervention*) 
            default_dummy_broadcast (int, optional): equals to 1 if the node resorted to a simple broadcast of one dummy message to each neighbor, because the batch was empty (last type of *batch intervention*)
        """        
        try:
            self.nb_sent_real_msgs_per_node[node.id].push_value(nb_reals)
        except AttributeError:
            pass
        
        try:
            nb_end_sent = nb_ocom_end_sent = nb_ocom_end_sent_payload = 0
            for m in (m for m in batch if m.flag is not MsgFlag.DUMMY):
                is_end_sent = (m.additional_info.get('end_sender', None) == node.id)
                nb_end_sent += is_end_sent
                is_ocom_end_sent = (is_end_sent and m.additional_info.get('is_ocom', None) is True)
                nb_ocom_end_sent += is_ocom_end_sent
                nb_ocom_end_sent_payload += (is_ocom_end_sent 
                                             and m.additional_info.get('is_ocom_payload', None) is True 
                                             and m.additional_info.get('ocom_end_sender', None) == node.id) 
            self.nb_end_sent_real_msgs_per_node[node.id].push_value(nb_end_sent)
            
            if self._net_manager.network_phase is apart.core.network.NetPhase.OCOM:
                self.nb_end_sent_ocom_msgs_per_node[node.id].push_value(nb_ocom_end_sent)
                self.nb_end_sent_ocom_payload_msgs_per_node[node.id].push_value(nb_ocom_end_sent_payload)
        except AttributeError:
            pass
        
        try:
            self.sent_link_msgs_per_node_per_round[node.id][node.batching_round].extend((m for m in batch if m.flag is not MsgFlag.DUMMY))
        except AttributeError:
            pass
        
        try:     
            self.nb_sent_dummy_link_msgs_per_node[node.id].push_value(nb_dummies)
        except AttributeError:
            pass
        
        try:
            self.nb_sent_e2e_dummy_msgs.push_value(nb_e2e_dummies)
        except AttributeError:
            pass  
        
        try:
            for m in (m for m in batch if m.flag is not MsgFlag.DUMMY):
                try:
                    round_timestamp = self._real_msg_round_timestamp.pop(m)
                    self.real_msg_waiting_time.push_value(node.batching_round-round_timestamp)
                    self.histogram_real_msg_waiting_time[node.batching_round-round_timestamp] += 1
                except KeyError:
                    assert m.additional_info['is_e2e_dummy']
                    pass
        except AttributeError:
            pass
        
        try:
            self.nb_batch_intervention_remove_reals.push_value(remove_real > 0)
            self.nb_batch_intervention_replace_dummy_by_e2e_dummy.push_value(replace_dummy_by_e2e_dummy > 0)
            self.nb_batch_intervention_add_e2e_dummy.push_value(add_e2e_dummy > 0)
            self.nb_batch_intervention_many_dummies.push_value(add_many_dummies > 0)
            self.nb_batch_intervention_default_dummy_bcast.push_value(default_dummy_broadcast > 0)
        except AttributeError:
            pass
        
        try:
            nb_reals_per_neighbor = defaultdict(lambda: 0)
            for m in (m for m in batch if m.flag is not MsgFlag.DUMMY):
                nb_reals_per_neighbor[m.sent_to] += 1
            for neighbor, nb in nb_reals_per_neighbor.items():
                self.histogram_nb_real_link_msgs_sent_per_round[nb][node.id][neighbor] += 1
        except AttributeError:
            pass
        
        try:
            neighbor = batch[0].sent_to
            nb_msgs_one_neighbor = sum(m.sent_to == neighbor for m in batch)
            self.histogram_nb_sent_link_msgs_per_round_per_neighbor[nb_msgs_one_neighbor] += 1
        except AttributeError:
            pass
    
    def log_round_rcvd_msgs(self, node, real_messages, nb_reals, nb_dummies, nb_e2e_dummies):  
        """Function logging the receiving of messages in a round.
        
        Function called by a node (:obj:`~apart.core.node.Node`). Modifies the
        following attributes if they exist: 
        
            * :attr:`.nb_rcvd_real_msgs_per_node`,
            * :attr:`.nb_end_rcvd_real_msgs_per_node`,
            * :attr:`.nb_end_rcvd_ocom_msgs_per_node`,
            * :attr:`.nb_end_rcvd_ocom_payload_msgs_per_node`,
            * :attr:`.rcvd_link_msgs_per_node_per_round`,
            * :attr:`.nb_rcvd_dummy_link_msgs_per_node`,
            * :attr:`.nb_rcvd_e2e_dummy_msgs`,
            * :attr:`.histogram_nb_real_link_msgs_rcvd_per_round`.
                
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is received the messages (`node.batching_round` gives the current round)
            real_messages (list of :obj:`~apart.core.messages.LinkMsg`): the real of messages received
            nb_reals (int): number of real messages received
            nb_dummies (int): number of dummy (link) messages received
            nb_e2e_dummies (int): number of end-to-end dummy messages received
        """
        try:
            self.nb_rcvd_real_msgs_per_node[node.id].push_value(nb_reals)
        except AttributeError:
            pass
        
        try:
            nb_end_rcvd = nb_ocom_end_rcvd = nb_ocom_end_rcvd_payload = 0
            for m in real_messages:
                is_end_rcvd = (m.additional_info.get('end_rcvr', None) == node.id)
                nb_end_rcvd += is_end_rcvd
                is_ocom_end_rcvd = (is_end_rcvd and m.additional_info.get('is_ocom', None) is True)
                nb_ocom_end_rcvd += is_ocom_end_rcvd
                nb_ocom_end_rcvd_payload += (is_ocom_end_rcvd 
                                             and m.additional_info.get('is_ocom_payload', None) is True 
                                             and m.additional_info.get('ocom_end_rcvr', None) == node.id)
                
            self.nb_end_rcvd_real_msgs_per_node[node.id].push_value(nb_end_rcvd)
            
            if self._net_manager.network_phase is apart.core.network.NetPhase.OCOM:
                self.nb_end_rcvd_ocom_msgs_per_node[node.id].push_value(nb_ocom_end_rcvd)
                self.nb_end_rcvd_ocom_payload_msgs_per_node[node.id].push_value(nb_ocom_end_rcvd_payload)
        except AttributeError:
            pass
        
        try:
            self.rcvd_link_msgs_per_node_per_round[node.id][node.batching_round].extend((m for m in real_messages if m.flag is not MsgFlag.DUMMY))
        except AttributeError:
            pass
        
        try:
            self.nb_rcvd_dummy_link_msgs_per_node[node.id].push_value(nb_dummies)
        except AttributeError:
            pass
        
        try:
            self.nb_rcvd_e2e_dummy_msgs.push_value(nb_e2e_dummies)
        except AttributeError:
            pass  
        
        try:
            nb_reals_per_neighbor = defaultdict(lambda: 0)
            for m in (m for m in real_messages if m.flag is not MsgFlag.DUMMY):
                nb_reals_per_neighbor[m.sent_by] += 1
            for neighbor, nb in nb_reals_per_neighbor.items():
                self.histogram_nb_real_link_msgs_rcvd_per_round[nb][node.id][neighbor] += 1
        except AttributeError:
            pass
    
    
    def log_rt_prop(self, node, rt_prop_event, nb_to_log=1, nb_rounds_to_complete=None):
        """Function logging information about route proposal.
        
        Function called by a node (:obj:`~apart.core.node.Node`). Modifies the
        following attributes if they exist: 
        
            * :attr:`.nb_route_props_per_node`,
            * :attr:`.rt_props_latency`.
           
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is concerned by the route proposal
            rt_prop_event (str or :obj:`~apart.core.protocol_constants.RtPolicyReason`): the event that is to be logged. Can be 'received', 'proposed', 'rt_prop_event', or a :obj:`~apart.core.protocol_constants.RtPolicyReason` value.
            nb_to_log (int, optional): number occurence of the event type to log (Default: 1)
            nb_rounds_to_complete (int, optional): if the event type in `rt_prop_event` is a :obj:`~apart.core.protocol_constants.RtPolicyReason` value, this argument specified the number of rounds that were necessary to complete the route proposal, starting from the initial rtprop message (Default: None)
        """
        try:
            try:
                self.nb_route_props_per_node[rt_prop_event][node.id] += nb_to_log
            except KeyError:
                if rt_prop_event != 'received':
                    logging.warning('Can not log route proposition event of type "{}"'.format(rt_prop_event))
                    
        except AttributeError:
            pass
        
        try:
            if rt_prop_event in RtPolicyReason: 
                # The condition is True only at the end of a route proposition:
                # it has been accepted or refused
                self.rt_props_latency.push_value(nb_rounds_to_complete)
        except AttributeError:
            pass
        
    def log_ocom_latency(self, node, end_sender_id, ocom_init_started_at_round, first_payload_sent_at_round):
        """Function logging the latency of an oriented communication.
        
        Function called by a node (:obj:`~apart.core.node.Node`). Modifies the
        following attributes if they exist: 
        
            * :attr:`.ocom_latency`,
            * :attr:`.ocom_latency_after_init`.
           
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is concerned by the oriented communication (typically, the end-receiver)
            end_sender_id (int): index of the end-sender node
            ocom_init_started_at_round (int): the round at which the initialisation of the oriented communication in question was started.
            first_payload_sent_at_round (int): the round at which the end-sender started sending payload data in the oriented communication in question.
        """
        try:
            self.ocom_latency.push_value(node.batching_round-ocom_init_started_at_round)
            self.ocom_latency_after_init.push_value(node.batching_round-first_payload_sent_at_round) 
        except AttributeError:
            pass
        
        
    def log_traffic_rate_equilibrium(self, node, equilibrium, lbound_real_msgs, ubound_real_msgs):
        """Function logging the traffic rates counters and equilibirum.
        
        Function called by a node (:obj:`~apart.core.node.Node`). Modifies the
        :attr:`.traffic_rate_status` attribute if it exists.
           
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is calling the function
            equilibrium (int): equilibirum of the node in the current round
            lbound_real_msgs (int): lower bound on real messages of the node in the current round
            ubound_real_msgs (int): upper bound on real messages of the node in the current round
        """
        try:
            self.traffic_rate_status['equilibrium'].push_value(equilibrium)
            self.traffic_rate_status['lbound'].push_value(lbound_real_msgs)
            self.traffic_rate_status['ubound'].push_value(ubound_real_msgs)
                                                                        
        except AttributeError:
            pass
        
    def log_ocom_route(self, node, ocomid, end_sender, helper, end_rcvr, first_hop_end_sender=None, first_hop_helper=None):
        """Function logging the oriented communication routes
        
        Function called by a node (:obj:`~apart.core.node.Node`). Modifies the
        :attr:`.ocom_routes` attribute if it exists.
           
        Args:
            node (:obj:`~apart.core.node.Node`): the node that is concerned by the oriented communication
            end_sender (int): index of the end-sender of the oriented communication
            helper (int): index of the indirection node/helper of the oriented communication
            end_rcvr (int): index of the end-receiver of the oriented communication
            first_hop_end_sender (int, optional): Index of the first hop node or the route chosen by the end-sender, along with the circuit identifier of the first hop. To be supplied when the  `node` calling the function is the end-sender. (Default: None).
            first_hop_helper (int, optional): Index of the first hop node or the route chosen by the idirection node/helper, along with the circuit identifier of the first hop. To be supplied when the  `node` calling the function is the helper. (Default: None).
        """
        try:
            if ocomid not in self.ocom_routes:
                self.ocom_routes[ocomid] = {}
            if first_hop_end_sender is not None:
                self.ocom_routes[ocomid]['end_sender'] = end_sender
                self.ocom_routes[ocomid]['helper'] = helper
                self.ocom_routes[ocomid]['end_rcvr'] = end_rcvr
                self.ocom_routes[ocomid]['first_hop_end_sender'] = first_hop_end_sender
            elif first_hop_helper is not None:
                assert (self.ocom_routes[ocomid]['end_sender'] == end_sender and 
                        self.ocom_routes[ocomid]['helper'] == helper and 
                        self.ocom_routes[ocomid]['end_rcvr'] == end_rcvr)
                self.ocom_routes[ocomid]['first_hop_helper'] = first_hop_helper
        except AttributeError as e:
            if not str(e).endswith("ocom_routes'"):
                raise e


class Stat(abc.ABC):
    """Abstract base class for :class:`.RegularStat` and :class:`.RunningStat`.
    
    This class (or more exactly, its derivatives), are meant to provide
    shortcuts to compute means, standard deviation, variances, minimum, and
    maximum values.
    """
    @abc.abstractproperty
    def nb_values(self):
        """int: Total number of values provided to the class instance.""" 
        pass
    
    @abc.abstractproperty
    def mean(self):
        """float: Mean of all values provided to the class instance"""
        pass
    
    @abc.abstractproperty
    def variance(self):
        """float: Variance of all values provided to the class instance"""
        pass
    
    @abc.abstractproperty
    def stdev(self):
        """float: Standard deviation of all values provided to the class instance"""
        pass
        
    @abc.abstractproperty
    def min(self):
        """float: Minimum of all values provided to the class instance"""
        pass
    
    @abc.abstractproperty
    def max(self):
        """float: Maximum of all values provided to the class instance"""
        pass
    
    @abc.abstractproperty
    def total(self):
        """float: Total sum of all values provided to the class instance"""
        pass
    
    def __str__(self):
        return "Total {}, Mean {}, Stdev {}, Min {}, Max {}".format(self.total, self.mean, self.stdev, self.min, self.max)
    
    def __repr__(self):
        return self.__str__()

class RunningStat(Stat):

    def __init__(self):
        """Comutes statistics on streaming data.
        
        Instead of computing directly the average etc. on a list of values, this
        class allows to compute values arriving in stream. This saves a lot a of
        memory when gathering statistics in the network with
        :class:`.NetworkStats`. Values are given one by one to an instance of
        this class by using the :meth:`.push_value` method. The attribtues
        :attr:`.mean`, :attr:`.stdev` can be accessed at any time, even before
        the stream of data is still ongoing. 
        
        Note that, while the average, minimum, and maximum can be easily computed on
        streaming data, the standard deviation and variance are more tricky. The
        implementation of this class is inspired from
        https://www.johndcook.com/blog/standard_deviation/
        """
        self.__nb_values= 0
        self.__old_mean = 0.0
        self.__new_mean = 0.0
        self.__old_stdev = 0.0
        self.__new_stdev = 0.0
        self.__min = None
        self.__max = None
        
        
    def push_value(self, v):
        """Updates all statistics (average, standard deviation) according to the specified value.
        
        Args:
            v (float): the new value of the data stream
        """
        self.__nb_values += 1
        
        if self.__nb_values == 1:
            self.__old_mean = self.__new_mean = self.__min = self.__max = v 
            self.__new_mean = v
            self.__old_stdev = 0.0
        else:
            self.__new_mean = self.__old_mean + (v - self.__old_mean)/self.__nb_values
            self.__new_stdev = self.__old_stdev + (v - self.__old_mean)*(v - self.__new_mean)

            self.__old_mean, self.__old_stdev = self.__new_mean, self.__new_stdev
            
            self.__min = min(self.__min, v)
            self.__max = max(self.__max, v)
            
    @property
    def nb_values(self):
        """int: Total number of values provided to the class instance.""" 
        return self.__nb_values
    
    @property
    def mean(self):
        """float: Mean of all values provided to the class instance"""
        return self.__new_mean
    
    @property
    def variance(self):
        """float: Variance of all values provided to the class instance"""
        try:
            return math.sqrt(self.__new_stdev/(self.__nb_values-1))
        except ZeroDivisionError:
            return 0.0
    @property
    def stdev(self):
        """float: Standard deviation of all values provided to the class instance"""
        return math.sqrt(self.variance)
        
    @property
    def min(self):
        """float: Minimum of all values provided to the class instance"""
        return self.__min
    
    @property
    def max(self):
        """float: Maximum of all values provided to the class instance"""
        return self.__max
    
    @property
    def total(self):
        """float: Total sum of all values provided to the class instance"""
        return self.mean*self.nb_values
        
class RegularStat(Stat):
    
    def __init__(self, values):
        """Computes statistics in a "regular manner" on a supplied list of values.
    
        Args:
            values (list of numbers): the values on which the statistics must be computed.
        """
        values = list(values)
        if len(values):
            self.__min = min(values)
            self.__max = max(values)
            self.__total = sum(values)
            self.__mean = statistics.mean(values)
            try:
                self.__variance = statistics.variance(values, xbar=self.__mean)
            except statistics.StatisticsError:
                self.__variance = 0
        else:
            self.__min = 0
            self.__max = 0
            self.__total = 0
            self.__mean = 0.0
            self.__variance = 0.0

    @property
    def nb_values(self):
        """int: Total number of values provided to the class instance.""" 
        return self.__nb_values
    
    @property
    def mean(self):
        """float: Mean of all values provided to the class instance"""
        return self.__mean
    
    @property
    def variance(self):
        """float: Variance of all values provided to the class instance"""
        return self.__variance
    
    @property
    def stdev(self):
        """float: Standard deviation of all values provided to the class instance"""
        return math.sqrt(self.variance)
        
    @property
    def min(self):
        """float: Minimum of all values provided to the class instance"""
        return self.__min
    
    @property
    def max(self):
        """float: Maximum of all values provided to the class instance"""
        return self.__max
    
    @property
    def total(self):
        """float: Total sum of all values provided to the class instance"""
        return self.__total
        
        
        