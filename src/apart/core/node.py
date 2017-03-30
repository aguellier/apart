# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html




"""
This module contains the code of a network node. 

The :class:`~apart.core.node.Node` class that it defines represents the core of
the protocol implementation. It models all the behavior of the nodes (mainly
message sending and receiving).

This module also defines the :exc:`.ProtocolError` exception.
"""
from _collections import defaultdict
import copy
import itertools
import logging
import math
import random

from apart.core.controlled_traffic_rates import TrafficController
from apart.core.messages import MsgFlag, LinkMsg, MsgInnerFlag,\
    MsgInnerHeader, EndToEndDummyError
from apart.core.pools import NeighborMsgPoolSet
from apart.core.protocol_constants import GROUP_G, F_NULL, GROUP_P,\
    RtPolicyReason
from apart.core.tables import RoutingTable as RT, PrevRoutingTable as PRT
from common.utilities import pprinttable, range1
from apart.crypto import Elgamal_keygen, Elgamal_enc, Reenc_nopk, \
    Elgamal_scalar_exp, Elgamal_enc_nopk, Elgamal_ctxt_div, \
    Elgamal_accumulator_check, Elgamal_key_div, Elgamal_plain_mult, SHA3_hash, \
    Elgamal_dec, Reenc_one, Elgamal_key_mult, Elgamal_accumulator_add, \
    DecryptionError, group_inverse, Reenc_pk


class Node(object):
    
    def __init__(self, env, net, net_manager, node_id, neighbors):
        """This class models a node in the network. 
    
        It builds and manages the internal state of the node. The main function :meth:`.run()`
        is a SimPy process, and consist in an infinite loop where a message performs
        routing information and exchange messages with other nodes in the simulation
    
        The constructor builds the internal state of the node, and register the
        :meth:`run` method as a  SimPy process. In particular, the constructor
        builds all routing tables, initializes the node's src and dst values,
        its message spools and traffic rates counters
        
        Arguments:
            env (:obj:`simpy.core.Environment`): the SimPy environment
            net (:obj:`~apart.core.network.Network`): the Network class instance which contains the node
            net_manager (:obj:`~apart.core.network.NetworManager`): the network manaer instance
            node_id (int): the node's identity (an integer)
            neighbors (int list): list of neighbors nodes in the topology graph   
        """

        # "Global" variables
        self._env = env
        """The SimPy environment of the simulation"""
        self._net = net
        """The network the node belongs to"""
        self._net_manager = net_manager
        """The network manager instance, that coordinates the nodes"""

        # Initialize node information
        self._id = node_id
        """The node's identity"""
        self._src = 2 * (self.id + 1)
        """The node's secret src value (should be random, is simplified here)"""
        self._dst = 3 * (self.id + 1)
        """The node's secret dst value (should be random, is simplified here)"""
        
        # Initialize the neighbors and their keys
        self._neighbors = neighbors
        """The list of node's neighbors in the topology graph"""
        
        # By default, a node is not corrupted
        self._is_corrupted = False
        """The corruption state of a node. Every node is honest by default, and :meth:`.set_corruption_state` can be called to modify that"""    
            
        # Generate the node's keys
        (self._pk, self._sk) = Elgamal_keygen(self._net.params.secparam)
        """The long_term key pair of the node"""
        
        # Create routing tables
        self._rt = RT(self.id, self._net.network_uid)
        """The node's main routing table"""
        self._prt = PRT(self.id, self._net.network_uid)
        """The node's "previous hop" routing table, to store multiple previous hop for each RT entry"""
        
        # Initialise the round counter
        self._batching_round = 0
        """The current batching round"""

        
        # Initialise the neighbor pools (for message re-ordering)
        n_min, f = self._net.params.batching_nmin, self._net.params.batching_f
        self._msg_pools = NeighborMsgPoolSet(n_min, f, self, neighbors)
        """The node's pools of messages, one for each neighbor"""
 
        # Initialize the inbox of the node as a simple set and a counter. The
        # set is meant to contain only *real* mesasges,  while the counter gives
        # the number of received dummy messages
        self.__incoming_real_msgs = set()
        """The node's incoming message box"""
        self.__nb_incoming_dummy_msgs = defaultdict(lambda: 0)
        """The node's counter for receiver dummy messages"""
        
        # Initialise an instance of traffic controller
        self.__traffic_counters = TrafficController(self._net.params.dummypol_deltar, self.neighbors)
        """The counters for the controller traffic rate mechanism"""
        
        # The node needs a structure to store its pending route proposals (the
        # one it sends, and the one it receives). Those are indexed by the cid
        # they are related to
        self._pending_rt_props = {'rcvd': {}, 'sent': {}, 'to_relay': {}, 'to_answer': {}, 'to_repropose': {}}
        """Information on pending route proposals, either as proposer, proposee, receiver or relay"""
        
        self._ongoing_ocom = {'sender': {}, 'helper': {}, 'to_receive': {}}
        """Information on ongoing oriented communications, as a sender or helper"""
        
        self._must_self_propose = True
        """By default, when a node is created, it should self-propose as its first action (when its process is started, see :meth:`.run`)"""
        
        self._process_is_idle = False
        """Indicates whether the node is "finished", w.r.t. the simualtion"""
        
    @property
    def id(self):
        """int: The node's identity (an integer)"""
        return self._id
    
    @property
    def neighbors(self):
        """list of int: The node's neighbor in the topology graph"""
        return self._neighbors
    
    @property
    def pk(self):
        """int: The node's long-term public key"""
        return self._pk
    
    @property
    def rt(self):
        """:obj:`~core.tables.RoutingTables`: The node's routing table"""
        return self._rt
    
    @property
    def prt(self):
        """:obj:`~core.tables.PrevRoutingTable`: The node's previous hop routing table"""
        return self._prt

    @property
    def batching_round(self):
        """int: The current batching round the node is in"""
        return self._batching_round
    
    @property
    def is_idle(self):
        """bool: Becomes ``True`` when a node does not receive not send messages in a round."""
        return self._process_is_idle
    
    def set_corruption_state(self, corrupted=True):
        """Sets the corruption state of the node"""
        self._is_corrupted = corrupted
    
    @property
    def is_corrupted(self):
        """bool: True is the node is corrupted"""
        return self._is_corrupted
    
        
        

    def __str__(self):
        return "Node {} (src = {}, dst = {}, msgs in pools = {})".format(self.id, self._src, self._dst, self._msg_pools.n_real)
    
    def __repr__(self):
        return self.__str__()
    
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    # Functions governing the node process
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    def run(self):
        """The node's SimPy process, and main function.
        
        The node SimPy process basically consists in an infinite loop (as long
        the attribute :attr:`~apart.core.network.NetworkManager.is_running` of
        :attr:`_net_manager` is ``True``) that makes batching round pass. That
        is, the loop begins with a SimPy timeout event of the duration of the
        batching time interval; then looks at incoming messages, and finally
        sends messages according to the batching strategy and the controlled
        traffic rate mechanism.
        
        This function does not make the difference between the topology
        dissemination phase, and the oriented communication phase. It is just
        the core running function.
        """
        
        if self._must_self_propose:
            self._must_self_propose = False
            # Initialization of the node's routing tables (i.e. insert the node
            # itself in its routing table)
            self_pseudo = SHA3_hash(pow(self._dst, self._src, GROUP_P))
            cone = Elgamal_enc(self._pk, 1)
            cprop = Elgamal_enc(self._pk, self._dst)
            chopcount = Elgamal_enc(self._pk, GROUP_G)
            cwhoisontheroute = Elgamal_enc(self._pk, " {} ".format(self.id))
            self_rt_entry = {RT.PSEUDO: self_pseudo, RT.CONE: F_NULL,  RT.NEXT_NODE: F_NULL,  RT.NEXT_CID:  F_NULL, 
                             RT.TIMESTAMP:  self._net.timestamp, RT.REPROPOSED: True, RT.IN_USE: True, 
                             RT.ACTUAL_RCVR: self.id, RT.ACTUAL_LENGTH: 0}
            
            self_rt_entry[RT.ROWID] = self.rt.insert_entry(**self_rt_entry)
    
            # The first action(s) that the node perform is a self-proposal
            self._propose_route(self_rt_entry, cprop, cone, chopcount, cwhoisontheroute)

        # Then iterate as long as the network manager says the network is
        # running.
        while self._net_manager.is_running:            
            logging.debug3("Batching round {} for node {}".format(self.batching_round, self.id))
            
            # Test whether the node is "idle" or not. I.e. if the node has no
            # incoming messages AND has no non-dummy messages to send
            if self.batching_round > 1 and not self.__incoming_real_msgs \
               and not any((p.n_real for p in self._msg_pools)) \
               and not any((len(d) > 0 for d in self._pending_rt_props.values())):
                self._process_is_idle = True
                # Ugly "tweak" below: by the way SimPy works, only the last node
                # needs to check termination of the system. This avoids useless
                # calls to ``check_network_idleness``
                if self.id == self._net.nb_nodes-1: self._net_manager.check_network_idleness() 
            else:
                self._process_is_idle = False
            
            # Whether the node is idle or not, it still needs to look at the
            # messages it received (at least to count the number of dummy
            # messages it received), and to send out messages (at least to send
            # one dummy to each neighbor). A node may start a round being
            # flagged "idle", and become "non-idle" during that round
            
                        
            # First, look at messages received in the previous round
            self._round_process_received_messages()
                        
            # And secondly, send out what needs to be sent
            self._round_send_messages()
            
            # Wait for the next round
            yield self._env.timeout(self._net.params.batching_t_interval) 
            
            self._next_round()
        
        # To finish properly, look at the last received messages, which 
        # should always be only dummy messages
        self._round_process_received_messages()
        
        # When a node comes out of the loop, check that it has indeed finished
        #all its business
        assert self._msg_pools.n_real == 0
        assert not self.__incoming_real_msgs
        if not all(len(d) == 0 for d in self._pending_rt_props.values()):
            logging.error("Node {} stopped while it still has pending route proposals:{}".format(self.id, self._pending_rt_props))
        if not all(len(d) == 0 for d in self._ongoing_ocom.values()):
            logging.error("Node {} stopped while it still has ongoing oriented communications : {}.".format(self.id, self._ongoing_ocom))

    
    
    def _next_round(self):
        """Advances the node to the next batching round
        
        This function is called at the end of each round, in particular to
        increase the batching round number
        """
        self._batching_round += 1
        self.__traffic_counters.next_round()
   
            
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    # Functions relating to the *receiving* of link messages and their
    # processing
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    def _receive_msg(self, m):
        """Depose a message in the node's incoming message box
        
        This function is called by other nodes of the network to place a
        link message in the node's incoming message box,
        ``self.__incoming_real_msgs``. This box is emptied at the *next* round, so
        messages will be counted as received in the next round. Note that dummy
        messages are simply counted and discarded
        
        Arguments:
            * m (:obj:`~apart.core.message.LinkMsg`): the message 
        """
        self._process_is_idle = False

        assert m.sent_by in self.neighbors and m.sent_to == self.id, "A message lost its way: {} was received by node {} (which has neighors {})".format(m, self.id, self.neighbors)

        if m.flag is MsgFlag.DUMMY:
            self.__nb_incoming_dummy_msgs[m.sent_by] += 1
        else:
            self.__incoming_real_msgs.add(m)
            
         
    def _round_process_received_messages(self):        
        """At the beginning of the round, process the received messages.
        
        Function called at the beginning of every round, to loop over the (real)
        messages received in the previous round. For each type of messages (i.e.
        for each :class:`MsgFlag`), an different message processing function is
        called. The incoming traffic counters are also updated in this function.
        Swap the "incoming_real_msgs" and "nb_incoming_dummy_messages" into
        local variables, and reinitialize these attributes to empty
        """
        real_msgs_last_round = self.__incoming_real_msgs
        self.__incoming_real_msgs = set()
        nb_dummy_msgs_last_round = self.__nb_incoming_dummy_msgs 
        self.__nb_incoming_dummy_msgs = defaultdict(lambda: 0)

        # Update the counters for real and dummy messages
        self.__traffic_counters.inc_current_n_i_real(inc=len(real_msgs_last_round))
        for (neighbor, n_dum) in nb_dummy_msgs_last_round.items():
            self.__traffic_counters.inc_current_n_i_dum(neighbor, inc=n_dum)
        
        # Count the number of received e2e dummies in the round
        nb_e2e_dummy_msgs = 0
        
        for rcvd_msg in real_msgs_last_round:
#             rcvd_msg = real_msgs_last_round.pop()
            
            logging.debug3("Node {} has received message {} from {} in round {} (time is {})".format(self.id, rcvd_msg, rcvd_msg.sent_by, self.batching_round, self._net.timestamp))

            # Basic check that the message received is indeed intended to the node
            assert (self.id == rcvd_msg.sent_to), "Node {} received a message not intended to it: {}".format(self.id, rcvd_msg)

            # Handle the received message appropriately, in accordance with the
            # dictionary handle_msg (that acts as a switch/case).
        
            assert rcvd_msg.flag in Node.__msg_handlers, "Protocol error: node {} received a link message of unsupported type {}.\nMessage is : {}".format(self.id, rcvd_msg.flag, rcvd_msg)
            
            try:
                Node.__msg_handlers[rcvd_msg.flag](self, rcvd_msg)
            except EndToEndDummyError:
                # If this exception is raised, that means (i) that the message
                # was a payload one and (ii) that is was actually and end to end
                # dummy
                nb_e2e_dummy_msgs += 1
            
        
        logging.debug2("Node {}, at round {}, has received {} real messages (including {} e2edum) and {} dummies".format(self.id, self.batching_round, len(real_msgs_last_round), nb_e2e_dummy_msgs, sum(nb_dummy_msgs_last_round.values())))
        
        # Log the received messages, for experiments and measures on the
        # protocol
        self._net_manager.net_stats.log_round_rcvd_msgs(self, real_msgs_last_round, 
                                                        len(real_msgs_last_round), 
                                                        sum(nb_dummy_msgs_last_round.values()), 
                                                        nb_e2e_dummy_msgs)
    
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    # Functions relating to the *sending* of link messages, message pool
    # management, and traffic rates enforcement
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    def _place_msg_in_pool(self, n, *msgs):
        """Puts a message in a neighbor pool.
            
        Function used by the node itself, to insert messages in its own neighbor
        pools. The main reason to do such a function is to automatically modify the
        ``self._process_is_idle`` flag to ``False``
        
        Arguments:
            * n (int): the neighbor 
            * *msgs (:obj:`~apart.core.messages.LinkMsg`): the messages to insert
        """
        self._process_is_idle = False
        
        self._net_manager.net_stats.log_msg_placed_in_pool(self, n, msgs)
        
        self._msg_pools[n].add_msgs(*msgs)

    def _round_send_messages(self):
        """Send messages at the end of the round.
        
        Function called by the node itself (FROM :meth:`.run`) at the end of
        every round (after the processing of received messages. It follows the
        batching and dummy policy described in the thesis: sample a batch from
        neighbor pools, compute traffic rates constraints, adjust the batch to
        fit the constraints, and send it. If no messages are available in pools,
        send one dummy to each neighbor.
        """
#         # DEBUG: put stress on node 0 by sending it many messages
#         if self.id != 0:
#             entries = self.rt.lookup(fields=[RT.CONE, RT.NEXT_NODE, RT.NEXT_CID], constraints={RT.ACTUAL_RCVR: 0})
#             if len(entries) > 0:
#                 entry = random.choice(entries)
#                 next_node = entry[RT.NEXT_NODE]
#                 next_cid = entry[RT.NEXT_CID]
#                 cone = entry[RT.CONE]
#                 c1 = Elgamal_enc_nopk(cone, 0)
#                 c2 = Elgamal_enc_nopk(cone, 1)
#                 for _ in range(3):
#                     m = LinkMsg(sent_by=self.id, sent_to=next_node, c1=c1, c2=c2, flag=MsgFlag.PAYLOAD, cid=next_cid)
#                     self._place_msg_in_pool(next_node, m)
            
        
        
        # Insert dummy messages in a random fraction of neighbor pools
        neighbor_fraction = random.sample(self.neighbors, math.ceil(len(self.neighbors)*self._net.params.dummypol_fdum))
        for n in neighbor_fraction:
            self._msg_pools[n].add_dummy_msg()
            
        
        # Get the margins that the node has in terms of sending out real messages
        current_equilibrium, lbound_real_msgs, ubound_real_msgs = self._traffic_rates_round_status()
        
        assert lbound_real_msgs <= -current_equilibrium <= ubound_real_msgs, "{} <= {} <= {}".format(lbound_real_msgs, current_equilibrium, ubound_real_msgs)
        
        # Log the equilibrium and bounds
        self._net_manager.net_stats.log_traffic_rate_equilibrium(self, current_equilibrium, lbound_real_msgs, ubound_real_msgs)
        
        # Example : If lbound_real_msgs = 5, that means the node must send AT
        # LEAST 5 real messages in this round. Alternatively, by construction of
        # the protocol, it can also send dummy messages so that its n_o_dum
        # budget becomes 5 or greater The current_equilibrium gives a hint on
        # how the node is doing in terms of sending/receiving balance. An
        # current_equilibrium lower to zero means that the node receiver more
        # messages than it sent out recently, and greater to zero conversely.
        # Here, by construction the current_equilibrium is never greater than
        # zero
        
        # Transform the current_equilibrium into a sampling bias. The lower
        # current_equilibrium is (below zero), the larger the bias must towards real
        # messages We use a simple inverse exponential function, tuned to
        # return 0.5 if equilibrium is -2.
#         sample_bias = 1-math.exp(current_equilibrium/(-2/math.log(0.5)))#7.2)
        sample_bias = 1-math.exp(ubound_real_msgs/(3/math.log(0.5)))#7.2)
        if sample_bias == 1.0:
            sample_bias = 0.9999999
        
                
        batches = self._msg_pools.sample_neighbor_batches(sample_bias)
        # batches = {n1 : {"reals": ..., "dummies": ...}, ...}

        assert len(set((len(b['reals'])+len(b['dummies']) for b in batches.values()))) <= 1, "Unadapted batches of node {} in round {} do not all have the same size:\n,".format(self.id, self.batching_round) + \
                                                                                              "\n".join(["\tNeighbor {}, len = {}: {}".format(n, len(b['reals'])+len(b['dummies']), b) for n,b in batches.items()])
        
        # Intervene on the batch if necessary
        batch, nb_reals, nb_dummies_per_neighbor, nb_e2e_dummies, batch_interventions = \
            self._adapt_batch(batches, current_equilibrium, lbound_real_msgs, ubound_real_msgs)
    
        logging.debug2("Node {}, at round {}, is going to send batch with {} real messages (including {} e2edum) and {} dummies".format(self.id, self.batching_round, nb_reals, nb_e2e_dummies, nb_dummies_per_neighbor))
        
        # For measures and experiments on the protocol, record
        # information on the batch sent out
        self._net_manager.net_stats.log_sent_batch(self, batch, nb_reals, sum(nb_dummies_per_neighbor.values()), nb_e2e_dummies, **batch_interventions)
        
        self._update_traffic_counters(nb_reals, nb_dummies_per_neighbor)
        self._send_batch(batch)
    
    def _traffic_rates_round_status(self):
        """Gives the current state of of counters of the controlled traffic rate mechanism.
        
        Function called before the sampling of a batch, to get the current state
        of the controlled traffic rate mechanism
        
        Returns:
            (int, int, int): the *equilibrium* in terms of sent/received real
            messages, and the upper and lower bounds on the real messages that
            can/must be sent in this round.
        """
        # An "current_equilibrium" lower than 0 means the node received more messages than
        # it sent out recently. And conversely if it is greater to 0
        current_equilibrium = self.__traffic_counters.current_equilibrium
        o_dum_budget = -self.__traffic_counters.total_n_o_dum_budget
        i_dum_budget = self.__traffic_counters.total_n_i_dum_budget
        
        # From the current_equilibrium and the dummy budgets, bounds on the number and
        # nature of messages to be sent in the round are deduced. The upper
        # bound is set so that AT EACH ROUND, the traffic rate equation is
        # respected (i.e. no n_o_real is left unresolved). But for the lower
        # bound, we relax the constraint on several rounds. That is, it is
        # allowed for a node to receive more than it sends out in a round.
        # However, the upper bound does ensure that AFTER delta_r rounds,
        # n_o_real is resolved.
        
        ubound_real_msgs = i_dum_budget - current_equilibrium
        lbound_real_msgs = o_dum_budget + self.__traffic_counters.last_unresolved_n_i_real
        
        # Note : ``lbound_real_msgs = o_dum_budget - current_equilibrium`` would
        # be the "stricter" lower bound corresponding to enforcing the traffic
        # rate equation PERFECTLY AT EACH ROUND, for both sending and receiving
        
        return current_equilibrium, lbound_real_msgs, ubound_real_msgs
    
    
    def _adapt_batch(self, batches, current_equilibrium, lbound_real_msgs, ubound_real_msgs):
        """Post processing of batches, to adapt them after sampling from the pools.
        
        Arguments:
        * batches (dict of dict of list of messages): the batches, per neighbor,
                     divided between reals and dummy ones
        * current_equilibrium (int): the traffic rate equilibirum in the current
                    round
        * lbound_real_msgs (int): the minimum number of rela messages to send in
                    the round
        * ubound_real_msgs (int): the maximum number of rela messages to send in
                    the round
        
        Returns:
            * batch (list of messages): the adapted batch, with message for all neighbors
            * nb_reals (int): the number of real message sin the batch
            * nb_dummies_per_neighbors (dict int->int): the number of dummy messages per neighbor
            * nb_e2e_dummies (int): the number of end to end dummy message sin the batch
            * batch_interventions (dict str->int): the nature of the modifications made on the batch
        """
        nb_reals = sum((len(b['reals']) for b in batches.values()))
        nb_dummies_per_neighbor = dict((n, len(b['dummies'])) for n, b in batches.items())
        nb_e2e_dummies = 0
        
        # For statistics and measures on the protocol, log the modifications made on the batch
        batch_interventions = {'remove_real': 0, 'replace_dummy_by_e2e_dummy': 0,
                              'add_e2e_dummy': 0, 'add_many_dummies': 0,
                              'default_dummy_broadcast': 0}
        
        # Adjust the batch to fit into the bounds posed by the controlled
        # traffic rates
        while nb_reals > ubound_real_msgs:
            # Select a batch from any neighbor that has at least one real
            # message, and replace the first real message in that batch with
            # a dummy
            n,b = random.choice([(n,b) for (n,b) in batches.items() if b['reals']])
            b['reals'].pop()
            b['dummies'].append(LinkMsg.create_dummy(sent_by=self.id, sent_to=n))
            nb_reals -= 1
            nb_dummies_per_neighbor[n] += 1
            batch_interventions['remove_real'] += 1
        while nb_reals + min(nb_dummies_per_neighbor.values()) < lbound_real_msgs:
            # Two options : augment min(dummies)  with simple dummy messages, 
            # or nb_reals with end to end dummy messages. BUT recall that
            # e2e dummy can not always be sent, especially at the beginning 
            # of the network life.
            # We choose the
            # following : if there exist a neighbor batch in which removing a
            # dummy does not decrease min(dummies), replace a dummy in this
            # batch by and end-to-end dummy. Otherwise, add one end-to-end
            # dummy and n-1 link dummies. And during these operations, if
            # no neighbor could be found to send an end-to-end dummy, we
            # simply fill the gap with link dummies
            intervention_ok = False
            # First, look if we can replace a link dummy with an e2e dummy
            min_nb_dummies = min(nb_dummies_per_neighbor.values())
            potential_neighbors = [(n,b) for (n,b) in batches.items() if len(b['dummies']) - min_nb_dummies]
            if len(potential_neighbors) > 0:
                random.shuffle(potential_neighbors)
                # Even though we want to send one e2e dummy to one neighbor,
                # we have to test them all: at the beginning of the network,
                # a node has actually no route in its RT, and thus no way to
                # send e2e. Here, we hope that the node knows at least one
                # route, and we must find the neighbor that begins it
                for (n, b) in potential_neighbors:
                    try:
                        b['reals'].append(self._craft_e2e_dummy(to_neighbor=n))
                        b['dummies'].pop()
                        nb_reals += 1
                        nb_dummies_per_neighbor[n] -= 1
                        nb_e2e_dummies += 1
                        batch_interventions['replace_dummy_by_e2e_dummy'] += 1
                        intervention_ok = True
                        break
                    except EndToEndDummyError:
                        continue
            if not intervention_ok:
                # Otherwise if the above trial of intervention failed, try to add
                # one e2e dummy and k-1 link dummies (k = nb neighbors)
                shuffled_neighbors = random.sample(self.neighbors, k=len(self.neighbors))
                # The for loop here is in the same idea as the preceding one: we test all neighbors to see if one
                # can actually get a e2e dummy.
                for (n, b) in batches.items():
                    try:
                        batches[shuffled_neighbors[0]]['reals'].append(self._craft_e2e_dummy(to_neighbor=n))
                        for n2 in shuffled_neighbors[1:]:
                            batches[n2]['dummies'].append(LinkMsg.create_dummy(sent_by=self.id, sent_to=n2))
                            nb_dummies_per_neighbor[n2] += 1
                        nb_reals += 1
                        nb_e2e_dummies += 1
                        batch_interventions['add_e2e_dummy'] += 1
                        intervention_ok = True
                        break
                    except EndToEndDummyError:
                        continue
            if not intervention_ok:
                # If still the above two trial of intervention failed
                # (normally, because no neighbor can receive an e2e dummy,
                # which happends at the beginning of the network life, when
                # the node does not know any route), we need to resort
                # solely to link  dummies: add the same amount of link dummy
                # message to each neighbor
                assert len(self.rt.lookup()) == 1
                nb_dum_missing = lbound_real_msgs - nb_reals - min(nb_dummies_per_neighbor.values())
                for n in self.neighbors:
                    batches[n]['dummies'].extend((LinkMsg.create_dummy(self.id, n) for _ in range(nb_dum_missing)))
                    nb_dummies_per_neighbor[n] += nb_dum_missing
                batch_interventions['add_many_dummies'] += 1
                
        
        assert len(set((len(b['reals'])+len(b['dummies']) for b in batches.values()))) <= 1, "Batches of node {} in round {} do not all have the same size:\n,".format(self.id, self.batching_round) + \
                                                                                              "\n".join(["\tNeighbor {}, len = {}: {}".format(n, len(b['reals'])+len(b['dummies']), b) for n,b in batches.items()]) +\
                                                                                              "\nInterventions on batch : {}".format(batch_interventions) + \
                                                                                              "\nBtw, my neighbors are: {}".format(self.neighbors)
                                                                                              
        
        # After this phase, validate the removal of the selected messages from the pools
        self._msg_pools.remove_batches(batches)
        
        # Merge all in one big batch
        batch = list(itertools.chain(*(b['reals']+b['dummies'] for b in batches.values())))
                
        # After all this, if the batch turns out to be empty  (e.g. because
        # no message is available from the pools), send one dummy to each
        # neighbor
        if len(batch) == 0:
            batch = [LinkMsg.create_dummy(sent_by=self.id, sent_to=n) for n in self.neighbors]
            nb_dummies_per_neighbor = dict((n, 1) for n in self.neighbors)
            nb_reals = 0
            nb_e2e_dummies = 0
            batch_interventions['default_dummy_broadcast'] += 1
        
        # Just a simple verification: that when real msgs must be removed, the number of dummies do not need to be modified (I think that would mean that lbound < ubound...)
        assert not (batch_interventions['remove_real']) or not (batch_interventions['replace_dummy_by_e2e_dummy'] + batch_interventions['add_e2e_dummy'] + batch_interventions['add_many_dummies'])
        
        return batch, nb_reals, nb_dummies_per_neighbor, nb_e2e_dummies, batch_interventions
    
    def _update_traffic_counters(self, nb_sent_real_msgs, nb_sent_dummy_msgs_per_neighbor):
        """Updates the traffic counters for the controlled traffic rates mechanism
        
        Function called after the formation (and adjustment) of the batch, to
        update the traffic counters.
        
        Arguments:
            * nb_sent_real_msgs (int): the number of sent real messages in the
                                    current round
            * nb_sent_dummy_msgs_per_neighbor (dict of int->int): the number of
                                    sent real dummy per neighbor in the 
                                    current round
        """
            
        # Update the n_o_real and n_o_dum counter
        self.__traffic_counters.inc_current_n_o_real(inc=nb_sent_real_msgs)
        for (n, v) in nb_sent_dummy_msgs_per_neighbor.items():
            self.__traffic_counters.inc_current_n_o_dum(n, inc=v)
        
        # Then, equilibrate the sending and receiving counters : whichever is
        # bigger than the other is going to "consume" the unresolved other
        self.__traffic_counters.equilibrate_n_o_real_and_n_i_real()
        
        # If n_o_real is still unresolved, resolve it further with the n_i_dum
        # budget. By the "strict" upper bound enforced above, there should be no
        # remaining unresolved n_o_real, but let's check anyway
        self.__traffic_counters.resolve_n_o_real_with_dummies()
        assert self.__traffic_counters.total_unresolved_n_o_real == 0, "Sending budget of node {} not large enough to send {} real messages.".format(self.id, nb_sent_real_msgs)
        
        # Secondly, see if the n_o_dum budget can take of some of the unresolved
        # n_i_real, if the later is not zero by now. Contrarily to the case above
        self.__traffic_counters.resolve_n_i_real_with_dummies()
        # However, the value of the round r-delta_r must be resolved by now
        assert self.__traffic_counters.last_unresolved_n_i_real == 0, "Not all constraints were resolved within the round window: n_i_real(delta_r) still unresolved."
        

    def _craft_e2e_dummy(self, to_neighbor):
        """Creates and returns an end-to-end dummy message.
        
        Helper function, called during the adjustment of the sampled batch to
        the traffic rates constraints, to craft an end-to-end dummy message
        towards a random receiver, with the constraint that the route must begin
        with neighbor specified by ``to_neighbor``
        
        If for some reason, the node does not have a route beginning with the 
        specified neighbor, the function raises an error 
        
        Arguments:
            * to_neighbor (int): The neighbor to sent the e2e dummy to.
        
        Returns:
            (:obj:`~apart.core.messages.LinkMsg` or None): the e2e message
            
        Raise:
            :exc:`~apart.core.message.EndToEndDummyError`: If no route beginning with the
                                    specified neighbor was found
        """
        
        # Search the routing table for a route that has the specified neighbor as next hop
        entries = self.rt.lookup(fields=[RT.CONE, RT.NEXT_CID, RT.ACTUAL_RCVR], constraints={RT.NEXT_NODE: to_neighbor})
        
        # No entries found means no route with this next hop
        if len(entries) < 1:
            raise EndToEndDummyError("No route beginning with neighbor {}".format(to_neighbor))


        entry = random.choice(entries)
        c1 = Elgamal_enc_nopk(entry[RT.CONE], MsgInnerHeader(MsgInnerFlag.DUMMY))
        c2 = Elgamal_enc_nopk(entry[RT.CONE], MsgInnerHeader(MsgInnerFlag.DUMMY))
        cid = entry[RT.NEXT_CID]
        actual_rcvr = entry[RT.ACTUAL_RCVR]

        return LinkMsg(sent_by=self.id, sent_to=to_neighbor, 
                       c1=c1, c2=c2, 
                       flag=MsgFlag.PAYLOAD, cid=cid, 
                       additional_info={'end_sender': self.id, 'end_rcvr': actual_rcvr, 'is_e2e_dummy': True})
    
    def _send_batch(self, batch):
        """Sends out a batch of messages, simulating latency in delivery.
        
        Function that does the sending of a whole batch of message. The batch
        contains a heterogeneous collection of messages for different neighbors.
        It is here that is computed the simulated the time taken to send (and
        process) all these messages. This latency is actually simulated in
        :meth:`_do_send_link_msg`. Note that, if there are really many messages in
        the batch, the simulated latency added in this function might exceed the
        time of a round. We'll see what happens in that case
        
        Arguments:
            * batch (:class:`list` of :obj:`~apart.core.messages.LinkMsg`): the
                            batch of link messages to send
        """
        
        # Make sure messages are sent in random order
        random.shuffle(batch)
        
        # Simulation of the the latency in the sending and propagation of link
        # messages
        base_latency = self._net.params.communication_latency
        accumulated_latency = 0
        
        # If implemented, AES encryption of headers should be done here: fetch
        # link keys in KeyTable, encrypt header *in place*, send message

        for m in batch:
            # The latency is accumulated
            accumulated_latency += round(base_latency*random.uniform(0.8, 1.2))
            assert accumulated_latency < self._net.params.batching_t_interval
            self._do_send_link_msg(m, accumulated_latency)

            
    def _do_send_link_msg(self, m, latency):
        """Sends one link message to a neighbor, with some delivery latency.
        
        Simulates the latency in each message sending using a SimPy timeout event,
        and delivers the message to the adequate node's
        ``self.__incoming_real_msgs`` box
        
        Arguments:
            * m (:obj:`~apart.core.messages.LinkMsg`): the link message to send.
            * latency (int): the delivery latency to simuulate.
        """   
        
        def delayed_send_callback(event):
            m = event.value
            # Send the message to the adequate neighbor
            logging.debug3("Node {} is sending msg {} to {} in round {} (time is{})".format(m.sent_by, m, m.sent_to, self._net.nodes[m.sent_by].batching_round, self._net.timestamp))
            self._net.nodes[m.sent_to]._receive_msg(m)
        
        # Create a timeout event with the specified latency, and "manually"
        # register the callback to effectivelly send the message (manually as
        # opposed to using "yield")
        timeout_event = self._env.timeout(latency, m)
        timeout_event.callbacks.append(delayed_send_callback)  


    
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    # Functions relating to route proposals and topology dissemination. Some
    # functions are message handlers, called by
    # :meth:`self._round_process_received_messages`. They are denoted by a
    # "_handle" prefix
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    def _propose_route(self, rt_entry, cprop, cone, chopcount, cwhoisontheroute, except_neighbor=None):
        """Makes the node propose a routing table entry to its neighbors
        
        This function makes self-proposals as well as relayed ones. It is based
        on a routing table entry, and is provided with all the ciphertexts to
        include in the rt prop messages. 
        
        In the case of a relayed proposition, ``except_neighbor`` will be equal
        to the neighbor from which the node just acquired the route. Indeed, it
        is useless to propose the route to the one that proposed it.
        
        Arguments:
            * rt_entry (:obj:`dict` or :obj:`sqlite3.Row`): the routing table entry to propose
            * cprop (:obj:`~apart.crypto.crypto.Ctxt`): the ciphertext containing the *dst* value of the receiver
            * cone (:obj:`~apart.crypto.crypto.Ctxt`): the ciphertext of one, for (re-)encryption
            * chopcount (:obj:`~apart.crypto.crypto.Ctxt`): the ciphertext containing the length of the route being proposed
            * cwhoisontheroute (:class:`~apart.crypto.crypto.Ctxt`): the ciphertext containing the nodes alreeady on the route proposed
            * except_neighbor (int): the neighbor to which the node must *not* propose the route 
        """
        logging.debug("Node {} is proposing a route towards node {} at round {}".format(self.id, rt_entry[RT.ACTUAL_RCVR], self.batching_round))
        
        # Create one route proposal for each neighbor
        prt_entries = []
        additional_info = {'actual_rcvr': rt_entry[RT.ACTUAL_RCVR], 'actual_proposer': self.id, 
                           'actual_proposee': None, 'round_rt_prop_started': self.batching_round, 
                           'actual_length': rt_entry[RT.ACTUAL_LENGTH]+1}
        
        for n in (n for n in self.neighbors if n != except_neighbor):
            new_cid = self._net_manager.get_next_cid(self.id, n)
            additional_info['actual_proposee'] = n
            msg_additional_info = copy.deepcopy(additional_info)
            
            rt_prop_msg_1 = LinkMsg(sent_by=self.id, sent_to=n,  
                                  c1=cprop, c2=cone,
                                  flag=MsgFlag.RTPROP, cid=new_cid, seq_index=1,  
                                  additional_info=msg_additional_info)
            rt_prop_msg_2 = LinkMsg(sent_by=self.id, sent_to=n,  
                                  c1=chopcount, c2=cwhoisontheroute,
                                  flag=MsgFlag.RTPROP, cid=new_cid, seq_index=2,
                                  additional_info=msg_additional_info)
            
            # Insert both messages in the pool for neighbor n
            self._place_msg_in_pool(n, rt_prop_msg_1, rt_prop_msg_2)
            
            # Note the new "previous hops" that may be formed at the end of the proposal
            prt_entries.append({PRT.PREV_NODE: n, PRT.PREV_CID: new_cid, 
                                PRT.RT_ROWID: rt_entry[RT.ROWID]})
            
            self._pending_rt_props['sent'][(n, new_cid)] = {'is_selfprop': rt_entry[RT.ACTUAL_RCVR] == self.id, 
                                                            'proposee': n, 'cid': new_cid,
                                                            'proposed_rt_entry_id': rt_entry[RT.ROWID],
                                                            'additional_info': additional_info}
            
            

        # Log the rt prop
        self._net_manager.net_stats.log_rt_prop(self, 'proposed', nb_to_log=len(prt_entries))
        
        # Insert all prt entries in one go
        self.prt.insert_entries(prt_entries)
        
    
    # ==========================================================================
    # Route proposals: Proposee, step 1
    # ==========================================================================
    def _handle_rt_prop_msg(self, rt_prop_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP` messages (received by proposee)
        
        A route proposal starts with *two*
        :attr:`~apart.core.messages.MsgFlag.RTPROP` messages. The one called
        "the first" contains the encrypted dst value, the one called "the second" contains 
        information on the route, to be used as part of the route proposal
        policy. Throughout the whole route proposal, these two messages are going to be
        carried around, from proposer to proposee, and from proposer to relay
        nodes to receiver and back.
        
        This function waits for the two
        :attr:`~apart.core.messages.MsgFlag.RTPROP` before calling
        :meth:`._proposee_answer_rt_prop` to proces sthe route proposal, and
        answer to the proposer.
        """
            
        # When a node receives a rt prop message, it can not process it straight
        # away. It must wait for the other rt prop message. These two messages
        # may arrive in any order.
        new_cid = rt_prop_msg.cid
        proposer = rt_prop_msg.sent_by
        is_fst_protocol_msg = (rt_prop_msg.seq_index == 1)
        if (proposer, new_cid) not in self._pending_rt_props['rcvd']:
            self._pending_rt_props['rcvd'][(proposer, new_cid)] = {'proposer': proposer, 'cid': new_cid,
                                                                 'additional_info': rt_prop_msg.additional_info}
        
        pending_rt_prop = self._pending_rt_props['rcvd'][(proposer, new_cid)]
            
        if is_fst_protocol_msg:
            pending_rt_prop.update({'cprop': rt_prop_msg.c1,
                                    'cone': rt_prop_msg.c2})
        else:
            pending_rt_prop.update({'chopcount': rt_prop_msg.c1,
                                    'cwhoisontheroute': rt_prop_msg.c2})
            
        if 'cprop' in pending_rt_prop and 'chopcount' in pending_rt_prop:
            self._proposee_answer_rt_prop(pending_rt_prop)
            
    def _proposee_answer_rt_prop(self, pending_rt_prop):
        """Process a route proposal as a proposee (first step)
        
        The node processes ciphertexts, in particular computing dst**src
        (homomorphically) for the pseudonym, and then sends back two
        :attr:`~apart.core.messages.MsgFlag.RTPROP_ANSWER` messages to the
        proposer. 
        
        To allow the proposer to differentiate the first message (containing
        dst**src encrypted notably) from the second one (containing the yes/no
        answer stating whether the proposee accepts the route or not), the later
        has a cid that we say *related* to the first message's cid. The domain
        of cids and related cids are not overlapping (see
        :meth:`~apart.core.network.Network.get_related_cid`)
        """  
        
        # When both the rt prop and the rt prop additional info messages were
        # received, process the route proposition
        proposer = pending_rt_prop['proposer']
        cid = pending_rt_prop['cid']
        cprop = pending_rt_prop['cprop']
        cone = pending_rt_prop['cone']
        chopcount = pending_rt_prop['chopcount']
        cwhoisontheroute = pending_rt_prop['cwhoisontheroute']
        
        
        # Log the_rt prop
        self._net_manager.net_stats.log_rt_prop(self, 'received')
        
        # Generate a temporary key pair
        (pk_tmp, sk_tmp) = Elgamal_keygen(self._net.params.secparam)
        
        
        # Process the base ciphertexts
        c1 = Reenc_nopk(cone, Elgamal_scalar_exp(cprop, self._src))
        c2 = Elgamal_enc_nopk(cone, pk_tmp)
        
        # c3 and c4 are supposed to each contains a yes/no answer 
        c3, c4 = self._proposee_route_proposal_policy_encrypted_decision(cone, chopcount, cwhoisontheroute)
        
        rt_prop_answer_msg_1 = LinkMsg(sent_by=self.id, sent_to=proposer, 
                                       c1=c1, c2=c2, flag=MsgFlag.RTPROP_ANSWER, cid=cid, seq_index=1)
        rt_prop_answer_msg_2 = LinkMsg(sent_by=self.id, sent_to=proposer, 
                                       c1=c3, c2=c4, flag=MsgFlag.RTPROP_ANSWER, cid=cid, seq_index=2)
        
         
        # Insert both messages in the pool for the adequate neighbor
        self._place_msg_in_pool(proposer, rt_prop_answer_msg_1, rt_prop_answer_msg_2)
        
        self._pending_rt_props['rcvd'][(proposer, cid)]['tmpkeypair'] = (pk_tmp, sk_tmp)


    def _proposee_route_proposal_policy_encrypted_decision(self, cone, chopcount, cwhoisontheroute):
        """Computes and returns the encrypted the yes/no answer stating whether the proposee accepts the route or not
        
        Called by the receiver on the ciphertexts contained in the
        :attr:`~apart.core.messages.MsgFlag.RTPROP_INFO` message, this function
        returns two ciphertexts. The first contains g**(lmax-l), and the second
        g**0/1. These will eventually be given to the end-receiver, who will
        transform that into a yes/no answer (see :meth:`._receiver_process_rt_prop`
        
        Arguments:
            * cone (Ctxt): one encrypted under the same pk as the other two 
                            ciphertexts (used for re-encryption)
            * chopcount (Ctxt): encrypts g**l, received by the proposee in a 
                            :attr:`~apart.core.messages.MsgFlag.RTPROP_INFO` message
            * cwhoisontheroute (Ctxt): encrypts [n1, n2, ...], received by the proposee 
                                        in a :attr:`~apart.core.messages.MsgFlag.RTPROP_INFO` message
            
        Returns:
            * cdecision_hopcount (Ctxt): the value lmax-l encrypted
            * cdecision_loop (Ctxt): ciphertext encrypting ``True`` if the node is already on the route
        """
        cmaxhopcount = Elgamal_enc_nopk(cone, pow(GROUP_G, self._net.params.rtprop_policy_max_hop_count))
        cdecision_hopcount = Elgamal_ctxt_div(cmaxhopcount, chopcount)
        cdecision_loop = Elgamal_accumulator_check(cwhoisontheroute, " {} ".format(self.id))
        return cdecision_hopcount, cdecision_loop
    
        
    
    
    
    
    
    # ==========================================================================
    # Route proposals: Proposer, step 2
    # ==========================================================================
    def _handle_rt_prop_answer_msg(self, rt_prop_answer_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP_ANSWER` messages  (received by proposer)
        
        Two cases arise: the proposer is also the receiver (self-proposal), or
        not. In the first case, the proposer waits for the *two*
        :attr:`~apart.core.messages.MsgFlag.RTPROP_ANSWER` messages, processes
        them (by calling :meth:`._receiver_process_rt_prop`), and sends back two
        :attr:`~apart.core.messages.MsgFlag.RTPROP_FINAl` messages (by calling
        :meth:`._proposer_send_rt_prop_final_msg`).
        
        In the second case, the proposer will sent the ciphertexts on a return
        trip to the distant receiver. For that, it can process the two
        :attr:`~apart.core.messages.MsgFlag.RTPROP_ANSWER` messages
        independently, and forward each one as soon as it receives in,
        transforming it into a
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message (by
        calling :meth:`._rt_prop_return_trip_relay_fwd`).
        """
        # The proposer excepts to receive *two* rt prop answer messages. They
        # may arrive in any order. In the case of a self-proposition, the
        # proposer must wait for both messages to arrive two before processing
        # them. In case of a relayed proposition, they can be processed independently
        proposee = rt_prop_answer_msg.sent_by
        cid = rt_prop_answer_msg.cid
        assert (proposee, cid) in self._pending_rt_props['sent']
        pending_rt_prop = self._pending_rt_props['sent'][(proposee, cid)]
        is_fst_protocol_msg = (rt_prop_answer_msg.seq_index == 1)
        
        if pending_rt_prop['is_selfprop']:
            # The node is the self-proposer. Consequently wait for both rt prop
            # answer messages to be received
            if is_fst_protocol_msg:
                # This is the first rt prop answer message
                pending_rt_prop.update({'cpseudo': rt_prop_answer_msg.c1,
                                        'cpktmp': rt_prop_answer_msg.c2})
            else:
                # This is the second rt prop answer message 
                pending_rt_prop.update({'caccept_hopcount': rt_prop_answer_msg.c1, 
                                        'caccept_loop': rt_prop_answer_msg.c2})
            
            # When both are received, answer to the proposee
            if 'cpseudo' in pending_rt_prop and 'caccept_hopcount' in pending_rt_prop:
                ctxts = self._receiver_process_rt_prop(*[pending_rt_prop[cname] for cname in ['cpseudo', 'cpktmp', 'caccept_hopcount', 'caccept_loop']])
    
                self._proposer_send_rt_prop_final_msg(True, proposee, cid, ctxts[0], ctxts[1])
                self._proposer_send_rt_prop_final_msg(False, proposee, cid, ctxts[1], ctxts[3])
                
                # Deleted the ongoing rt prop entry
                del self._pending_rt_props['sent'][(proposee, cid)]
        else:
            # The proposer is not the receiver. It must "tranform" the rt prop
            # answer msgs into rt prop relay fwd ones, and act "as if" it was
            # really relaying one            
            prev_rcid = None
            is_fst_protocol_msg = (rt_prop_answer_msg.seq_index == 1)
            additional_info = copy.deepcopy(rt_prop_answer_msg.additional_info)
            additional_info.update(end_sender=self.id, end_rcvr=pending_rt_prop['additional_info']['actual_rcvr'])
            self._rt_prop_return_trip_relay_fwd(is_fst_protocol_msg, proposee, cid, prev_rcid, 
                                                rt_prop_answer_msg.c1, rt_prop_answer_msg.c2, 
                                                additional_info=additional_info)
            
            # After the second message has been processed, we can safely delete
            # information concerning the pending prop (the necessary information
            # will be stored elsewhere, as if the node was a simple relay)
            if 'one_msg_processed_already' in pending_rt_prop:
                del self._pending_rt_props['sent'][(proposee, cid)]
            else:
                pending_rt_prop['one_msg_processed_already'] = True
    
    
    
    
    
    # ==========================================================================
    # Route proposals: Return trip, forward direction
    # ==========================================================================
    def _handle_rt_prop_relay_fwd_msg(self, rt_prop_relay_fwd_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` messages
        
        This type of message is received by relays or  end-receiver of route
        proposals, or by the helper in an oriented comm initialisation.
        
        In the first case, each of the two related
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` can be processed
        independetly, and this function simply calls
        :meth:`._rt_prop_return_trip_relay_fwd`
        
        
        In the second case and third case, the function defers the treatement of
        the :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message to
        :meth:`._receiver_handle_rt_prop_relay_fwd_msg`.
        """
        # There are two mains cases when the node receives a rt prop relay fwd
        # message: either it is a simple relay, either it is the receiver. 
        prev_node = rt_prop_relay_fwd_msg.sent_by
        prev_cid = rt_prop_relay_fwd_msg.cid
        prev_rcid = rt_prop_relay_fwd_msg.rcid
        is_fst_protocol_msg = (rt_prop_relay_fwd_msg.seq_index == 1)
        
        if (prev_node, prev_cid, prev_rcid) in self._pending_rt_props['to_answer']:
            # If the node is receiver AND it is the second rt prop relay fwd msg it
            # receives, then the below condition will be true.
            is_receiver = True
        else:
            # Otherwise, try to handle the rt prop relay fwd msg as a relay. The
            #function returns ``True`` if the node is actually the receiver (and
            #it is the first rt prop relay fwd message that it sees
            is_receiver = self._rt_prop_return_trip_relay_fwd(is_fst_protocol_msg, prev_node, prev_cid, prev_rcid,
                                                              rt_prop_relay_fwd_msg.c1, rt_prop_relay_fwd_msg.c2,
                                                              rt_prop_relay_fwd_msg.additional_info)
            
        
        # If the node is the receiver, a different processing is applied
        if is_receiver:
            self._receiver_handle_rt_prop_relay_fwd_msg(prev_node, prev_cid, prev_rcid, rt_prop_relay_fwd_msg)
        
    
    def _rt_prop_return_trip_relay_fwd(self, is_fst_protocol_msg, prev_node, prev_cid, prev_rcid, c1, c2, additional_info=None):
        """Relays a route proposal answer on the way forward, from proposer to receiver
        
        This function is called either by: a node that itself received a
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message, or by
        proposers that itself received a
        :attr:`~apart.core.messages.MsgFlag.RTPROP_ANSWER` message for a non-
        self-proposal. In the first case, the node can either be a simple relay
        (and must forward the
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message), or the
        end-receiver (that must bounce back a
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` message).
        
        If the node is *not* the receiver, the role of this function is to
        process the ciphertexts involved in the route proposal (mainly by re-
        encrypting them, and by adding in its own temporary public key), and to
        forward the message. This function is meant to process
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` messages *one by
        one*. Recall that, for the same route proposal, 4 ciphertexts need to do
        the return trip from proposee to receiver, and thus two messages are
        necessary. The "first protocol message" is defined as the one carrying
        the encrypted dst**src value (and the encryted temporary public key). A
        node makes the distinction between the first and second protocol
        mesasge, because the later has a rcid different from the first, but
        deterministically bound to it, according to the definition of
        :meth:`apart.core.network.Network.get_related_rcid`.
        
        If the node is the receiver, this function return ``True``, whch has the
        effect of aborting the function early and coming back to
        :meth:`._handle_rt_prop_relay_fwd_msg`.
        
        Arguments:
            * is_fst_protocol_msg (bool): True if the
                                    :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message is the
                                    one carrying the encrypted dst**src value
            * prev_node (int): the previous relay node (or the proposee, if this function was called by the proposer)
            * prev_cid (int): the cid with which the previous node sent its message 
            * prev_rcid(int or None): the rcid with which the previous relay node sent its message, 
                                    or None if this function is called by the proposer  
            * c1 (:class:`~apart.crypto.crypto.Ctxt`): the first ciphertext to relay (either dst**src or the encrypted result of the route length test)
            * c2 (:class:`~apart.crypto.crypto.Ctxt`): the second ciphertext to relay (either pk_tmp or the encrypted result of the route loop test)
            * additional_info (:obj:`dict` or None): the additional info to carry along the return trip. For purposes of the simualtion and measurements 
            
        Returns:
            * bool or None: True if the node is the end-receiver of the route proposal 
        """
        # First of all, check if a temporary storage used for forwarding the rt
        # prop relay fwd messages was already created. If so, that measn the
        # node is *not* the receiver, and that the first of the two rt prop
        # relay fwd message was already processed
        if ('fwd', prev_node, prev_cid, prev_rcid) in self._pending_rt_props['to_relay']:
            info = self._pending_rt_props['to_relay'][('fwd', prev_node, prev_cid, prev_rcid)]
            next_node =  info['next_node']
            next_cid =  info['next_cid']
            new_rcid =  info['new_rcid']
            cone = info['cone']
            pk_tmp = info['pk_tmp'] 
            
            del self._pending_rt_props['to_relay'][('fwd', prev_node, prev_cid, prev_rcid)]
        else:
            # Get the (unique) next hop on the route. This "next hop" will tell
            # if the node is a relay or the receiver
            rows = self.rt.joint_lookup(PRT, fields=[RT.CONE, RT.NEXT_NODE, RT.NEXT_CID], join_on=[RT.ROWID, PRT.RT_ROWID], constraints={PRT.PREV_NODE: prev_node, PRT.PREV_CID: prev_cid})
            assert len(rows) == 1, "Protocol error: node {} must relay a {} message, but can not seem to find a unique next hop (rows returned: {})".format(self.id, str(MsgFlag.RTPROP_RELAY_FWD), [[v for v in r] for r in rows])
            
            # A null next hop indicates that the end of the route is reached,
            # i.e. the node is not a realy, but a receiver
            if rows[0][RT.NEXT_NODE] == F_NULL:
                return True
            else:
                # The node is a simple relay, and this is the first out of two
                # rt prop relay fwd messages
                next_node = rows[0][RT.NEXT_NODE]
                next_cid = rows[0][RT.NEXT_CID]
                new_rcid = self._net_manager.get_next_rcid(self.id, next_node)
                cone = rows[0][RT.CONE]
                (pk_tmp, sk_tmp) = Elgamal_keygen(self._net.params.secparam)
                
                # Store information for the *second* rt prop relay fwd message
                self._pending_rt_props['to_relay'][('fwd', prev_node, prev_cid, prev_rcid)] = \
                                {'next_node': next_node, 'next_cid': next_cid, 'new_rcid': new_rcid,
                                 'cone': cone, 'pk_tmp': pk_tmp}
                
                # Store information needed to later relay the two rt prop relay
                # bwd messages
                self._pending_rt_props['to_relay'][('bwd', next_node, next_cid, new_rcid)] = \
                        {'node_to_proposee': prev_node, 'cid_to_proposee': prev_cid, 'rcid_to_proposee': prev_rcid,
                        'cone': cone,'sk_tmp': sk_tmp}             
        
        # The value of the rcid depends on whether it is the first or second
        #message (according to the protocol, not the order of receiving).
        #Likewise, the processing of ciphertext differs slighlty
        if is_fst_protocol_msg:
            c1 = Reenc_nopk(cone, Elgamal_key_div(self._sk, c1)) 
            # The second ciphertext of the first message contains the
            # (accumulation of) temporary public key(s). The node must thus
            # multiply in its own pk. 
            c2 = Reenc_nopk(cone, Elgamal_key_div(self._sk, Elgamal_plain_mult(c2, pk_tmp)))
        else:
            c1 = Reenc_nopk(cone, Elgamal_key_div(self._sk, c1)) 
            c2 = Reenc_nopk(cone, Elgamal_key_div(self._sk, c2))
        
        
        # Craft and send the message
        new_rt_prop_relay_fwd_msg = LinkMsg(sent_by=self.id, sent_to=next_node, 
                                              c1=c1, c2=c2, 
                                              flag=MsgFlag.RTPROP_RELAY_FWD, cid=next_cid, rcid=new_rcid, seq_index=(1 if is_fst_protocol_msg else 2),
                                              additional_info=additional_info)
        
        self._place_msg_in_pool(next_node, new_rt_prop_relay_fwd_msg)
        
        return False
    
    
    
    # ==========================================================================
    # Route proposals: receiver, and bounce back of return trip
    # ==========================================================================                    
    def _receiver_handle_rt_prop_relay_fwd_msg(self, prev_node, prev_cid, prev_rcid, rt_prop_relay_fwd_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` messages, specifically for receivers of route proposals, or for helpers in a oriented communication initialisation
        
        This function is mainly meant to model the behavior of the end-receiver
        in a route proposal. But, because relay rt prop fwd messages are also
        used during oriented communication initialisation, this function also
        takes into account that possibility.
        
        If the node is a receiver of route proposal, the function waits for both
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` messages to be
        received, processes the ciphertexts, and sends back
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` messages
            
        If the node is a helper of oriented communications, it deffers the
        processing to :meth:`._helper_process_ocom_init_msg`.
        
        Arguments:
            * prev_node (int): the previous relay node
            * prev_cid (int): the cid with which the previous node sent its message 
            * prev_rcid(int or None): the rcid with which the previous relay node sent its message
            * rt_prop_relay_fwd_msg (:obj:`~apart.core.messages.LinkMsg`): the :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message received
        """
        # Initialise the temporary storage
        if (prev_node, prev_cid, prev_rcid) not in self._pending_rt_props['to_answer']:
                self._pending_rt_props['to_answer'][(prev_node, prev_cid, prev_rcid)] = {}
        
        # Boldly try to see if the message is part of an oriented communication
        # initialisation
        if self._pending_rt_props['to_answer'][(prev_node, prev_cid, prev_rcid)] == 'ocom':
            is_ocom = True
            inner_header = Elgamal_dec(self._sk, rt_prop_relay_fwd_msg.c1)
            del self._pending_rt_props['to_answer'][(prev_node, prev_cid, prev_rcid)]
        else:
            try:
                inner_header = Elgamal_dec(self._sk, rt_prop_relay_fwd_msg.c1)
                is_ocom = inner_header.flag is MsgInnerFlag.OCOM_INIT
                self._pending_rt_props['to_answer'][(prev_node, prev_cid, prev_rcid)] = 'ocom'
            except AttributeError:
                is_ocom = False
        if is_ocom:
            self._helper_process_ocom_init_msg(rt_prop_relay_fwd_msg, inner_header)
            return
        
        # If we arrive here, that means the node is a receiver in a route proposal
        

        
        pending_rt_prop = self._pending_rt_props['to_answer'][(prev_node, prev_cid, prev_rcid)]
        is_fst_protocol_msg = (rt_prop_relay_fwd_msg.seq_index == 1)
        
        if is_fst_protocol_msg:
            d = {'cpseudo': rt_prop_relay_fwd_msg.c1, 'cpktmp': rt_prop_relay_fwd_msg.c2}
        else:
            d = {'chopcount': rt_prop_relay_fwd_msg.c1, 'cwhoisontheroute': rt_prop_relay_fwd_msg.c2}
        pending_rt_prop.update(d)
        
        if 'cpseudo' in pending_rt_prop and 'chopcount' in pending_rt_prop:
            # When both prop relay fwd messages are received, the receiver can
            # make an answer, and send back a rt prop relay bwd message
            ctxts = self._receiver_process_rt_prop(*[pending_rt_prop[cname] for cname in ['cpseudo', 'cpktmp', 'chopcount', 'cwhoisontheroute']])
            additional_info = copy.deepcopy(rt_prop_relay_fwd_msg.additional_info)
            additional_info.update(end_sender=self.id, end_rcvr=rt_prop_relay_fwd_msg.additional_info['end_sender'])

            self._rt_prop_return_trip_relay_bwd(True, rt_prop_relay_fwd_msg.sent_by, rt_prop_relay_fwd_msg.cid, prev_rcid, 
                                                ctxts[0],ctxts[1], None,
                                                additional_info=additional_info)
            self._rt_prop_return_trip_relay_bwd(False, rt_prop_relay_fwd_msg.sent_by, rt_prop_relay_fwd_msg.cid, prev_rcid, 
                                                ctxts[2],ctxts[3], None,
                                                additional_info=additional_info)
            
            del self._pending_rt_props['to_answer'][(prev_node, prev_cid, prev_rcid)]
            

                                                                 
        
    def _receiver_process_rt_prop(self, cpseudo, cpktmp, caccept_hopcount, caccept_loop):
        """Processes the four ciphertexts crafted by the proposee, to compute the pseudonym and the yes/no answer on the route acceptation
        
        Al lthe ciphertexts given in argument are encrypted under the receiver's
        secret key (uniquely), and all the output ciphertexts are encrypted
        under the product of temporary secret key (that is contained in ``cpktmp``.
        
        Arguments:
            * cpseudo (:class:`~apart.crypto.crypto.Ctxt`): the dst**src value 
            * cpktmp (:class:`~apart.crypto.crypto.Ctxt`): the product of temporary public key, 
                                    accumulated during the forwarding from proposer to receiver
            * caccept_hopcount (:class:`~apart.crypto.crypto.Ctxt`): the hop count test, g**(lmax-l) 
            * caccept_loop (:class:`~apart.crypto.crypto.Ctxt`): the loop test, g**0/1
            
        Returns:
            * :class:`~apart.crypto.crypto.Ctxt`: the pseudonym, encrypted under pktmp
            * :class:`~apart.crypto.crypto.Ctxt`: the value one, encrypted under pktmp
            * :class:`~apart.crypto.crypto.Ctxt`: the yes/no answer on route acceptance, encrypted under pktmp
            * :class:`~apart.crypto.crypto.Ctxt`: the value one, encrypted under pktmp
        """
        # Log the_rt prop
        self._net_manager.net_stats.log_rt_prop(self, 'answered_as_receiver')
        
        # Here, the receiver must simply process the ciphertexts. 
        # The function returns a set of ciphertexts to send back accordingly, but does not actually sends any messages
        pseudo = SHA3_hash(Elgamal_dec(self._sk, cpseudo))
        pktmp = Elgamal_dec(self._sk, cpktmp)
        
        accept_hopcount = Elgamal_dec(self._sk, caccept_hopcount)
        acceptrt = accept_hopcount in [GROUP_G**i % GROUP_P for i in range1(0, self._net.params.rtprop_policy_max_hop_count)]
        acceptrt = acceptrt and not Elgamal_dec(self._sk, caccept_loop)
        
        cpseudo = Elgamal_enc(pktmp, pseudo)
        cone = Elgamal_enc(pktmp, 1)
        cacceptrt = Elgamal_enc(pktmp, acceptrt)
        other_cone = Elgamal_enc(pktmp, 1)
        
        return cpseudo, cone, cacceptrt, other_cone
    

    # ==========================================================================
    # Route proposals: Return trip, backward direction
    # ==========================================================================
    def _handle_rt_prop_relay_bwd_msg(self, rt_prop_relay_bwd_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` messages
        
        It is very similar to :meth:`._handle_rt_prop_relay_fwd_msg`, but for
        the backward direction of the return trip. This type of message is
        received by relays or the proposer of a route proposals, or by the end-
        sender in an oriented comm initialisation.
        
        In the first case, each of the two related
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` can be processed
        independetly, and this function simply calls
        :meth:`._rt_prop_return_trip_relay_bwd`.
        
        
        In the second case, the function defers the treatement of
        the :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_VWD` message to
        :meth:`._proposer_handle_rt_prop_relay_bwd_msg`.
        
        In the third case, if the node is actually an end-sender in an oriented 
        communication, the function defers the processing to 
        :meth:`._sender_ocom_init_finalise`
        
        In any case, the function also manages the deletion of the temporary
        information on the route proposal when it is no longer needed.
        """
        # There are two mains cases when the node receives a rt prop relay bwd
        # message: either it is a simple relay, either it is the original
        # proposer.
        node_to_receiver = rt_prop_relay_bwd_msg.sent_by
        cid_to_receiver = rt_prop_relay_bwd_msg.cid
        rcid_to_receiver = rt_prop_relay_bwd_msg.rcid
        is_fst_protocol_msg = (rt_prop_relay_bwd_msg.seq_index == 1)
        
        # First of all, test if this message is part of an oriented 
        # communication initalisation
        try:
            ocomid = self._ongoing_ocom['sender'][('rt_prop_relay_bwd_shortcut', node_to_receiver, cid_to_receiver, rcid_to_receiver)]
        except KeyError:
            ocomid = None
        if ocomid:
            self._sender_ocom_init_finalise(ocomid, rt_prop_relay_bwd_msg)
            return
            
        # If the node is not the sender in an oriented communication, 
        # it must be that it is a relay or proposer in a route proposal
        assert ('bwd', node_to_receiver, cid_to_receiver, rcid_to_receiver) in self._pending_rt_props['to_relay']
        pending_rt_prop = self._pending_rt_props['to_relay'][('bwd', node_to_receiver, cid_to_receiver, rcid_to_receiver)]

        
        if pending_rt_prop['rcid_to_proposee'] is None:
            # If the node is the proposer, call the function specific to this case
            self._proposer_handle_rt_prop_relay_bwd_msg(pending_rt_prop, rt_prop_relay_bwd_msg)
        else:
            # Otherwise, try to handle the rt prop relay bwd msg as a relay. 
            self._rt_prop_return_trip_relay_bwd(is_fst_protocol_msg, pending_rt_prop['node_to_proposee'], pending_rt_prop['cid_to_proposee'], pending_rt_prop['rcid_to_proposee'], 
                                                rt_prop_relay_bwd_msg.c1, rt_prop_relay_bwd_msg.c2, pending_rt_prop['sk_tmp'],
                                                rt_prop_relay_bwd_msg.additional_info)
        
        # Manage the deletion of the temporary storage of the route proposal
        # information
        if 'one_msg_processed_already' in pending_rt_prop:
            # If the node ahs finished relaying the two related rt prop relay
            # bwd msgs, delete the pending rt prop
            del self._pending_rt_props['to_relay'][('bwd', node_to_receiver, cid_to_receiver, rcid_to_receiver)]
        else:
            pending_rt_prop['one_msg_processed_already'] = True
    
    # Below function used by : relays upon rt prop relay bwd msg, receivers upon
    #rt prop relay fwd msg
    def _rt_prop_return_trip_relay_bwd(self, is_fst_protocol_msg, node_to_proposee, cid_to_proposee, rcid_to_proposee, c1, c2, sk_tmp, additional_info=None):
        """Relays a route proposal answer on the way back, from receiver to proposer
        
        This function is the alter ego of
        :meth:`._rt_prop_return_trip_relay_fwd`, but slightly simpler.
        
        This function is called either by: a node that itself received a
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` message, or by a
        receiver that received a
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD` message  and wants
        to bounce back the ciphertexts on the return trip.
        
        Note that the tuple (node_to_proposee, cid_to_proposee,
        rcid_to_proposee) is actually used to retrive the temporary information
        stored on the node during the relaying forward of the return trip. This
        temporary information mainly says where and how to relay back to the proposer.
        
        Arguments:
            * is_fst_protocol_msg (bool): True if the
                                    :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` message is the
                                    one carrying the encrypted pseudonym
            * node_to_proposee (int): the next relay node, towards the proposee and proposer
            * cid_to_proposee (int): the next cid to use 
            * rcid_to_proposee(int or None): the next rcid to use
            * c1 (:class:`~apart.crypto.crypto.Ctxt`): the first ciphertext to relay (either the pseudonym0 
                                        or the encrypted yes/no answer on the route acceptance)
            * c2 (:class:`~apart.crypto.crypto.Ctxt`): the second ciphertext to relay (always an encryption of one)
            * additional_info (:obj:`dict` or None): the additional info to carry along the return trip. For purposes of the simualtion and measurements 
        """
        
        if sk_tmp is not None: # sk_tmp is None when this fucntion is called by the receiver
            c1, c2 = self._rt_prop_return_trip_bwd_partial_dec(c1, c2, sk_tmp)
        
        # Craft and send the rt prop relay bwd msg 
        new_rt_prop_relay_bwd_msg = LinkMsg(sent_by=self.id, sent_to=node_to_proposee,
                                          c1=c1, c2=c2, 
                                          flag=MsgFlag.RTPROP_RELAY_BWD, cid=cid_to_proposee, rcid=rcid_to_proposee, seq_index=(1 if is_fst_protocol_msg else 2),
                                          additional_info=additional_info)

        self._place_msg_in_pool(node_to_proposee, new_rt_prop_relay_bwd_msg)
        
    def _rt_prop_return_trip_bwd_partial_dec(self, c1, c2, sk_tmp):
        """Helper function for relay nodes on the way backward of the return trip to cancell out their temporary secret key"""
        c2 = Reenc_one(Elgamal_key_div(sk_tmp, c2))
        c1 = Reenc_nopk(c2, Elgamal_key_div(sk_tmp, c1))
        return c1, c2
    
    
    # ==========================================================================
    # Route proposals: Proposer, end of 3rd step
    # (note: in self-proposals, 2nd and 3rd step are made in one go)
    # ==========================================================================
    def _proposer_handle_rt_prop_relay_bwd_msg(self, pending_rt_prop, rt_prop_relay_bwd_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` messages, specifically for proposers of route proposals
        
        This function is mainly meant to model the behavior of the proposer in a
        *relayed* route proposal, upon return of the answer from the receiver.
       
        
        Here, each
        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` message can be
        processed independently. The proposer only needs to "transform" them into 
        :attr:`~apart.core.messages.MsgFlag.RTPROP_FINAL` messages.
        

        Arguments:
            * pending_rt_prop (:obj:`dict`): information on the route proposal
            * rt_prop_relay_bwd_msg (:obj:`~apart.core.messages.LinkMsg`): the :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_BWD` message received
        """
        
       
        # If the node is a proposer of a route proposal, it must "transform" the
        # rt pro prelay bwd msgs into rt prop final ones, and send them back. Do
        # not forget to partially decrypt the ciphertexts also
        is_fst_protocol_msg = (rt_prop_relay_bwd_msg.seq_index == 1)
        
        c1, c2 =self._rt_prop_return_trip_bwd_partial_dec(rt_prop_relay_bwd_msg.c1, rt_prop_relay_bwd_msg.c2,  pending_rt_prop['sk_tmp'])
#         pending_sent_rt_prop = self._pending_rt_props['sent'][(pending_rt_prop['node_to_proposee'], pending_rt_prop['cid_to_proposee'])]
        self._proposer_send_rt_prop_final_msg(is_fst_protocol_msg, pending_rt_prop['node_to_proposee'], pending_rt_prop['cid_to_proposee'], c1, c2)

    def _proposer_send_rt_prop_final_msg(self, is_fst_protocol_msg, proposee, cid, c1, c2):
        """Craft and sends the :attr:`~apart.core.messages.MsgFlag.RTPROP_FINAL` messages to the proposee
        
        Called by the proposer, this function handles the two messages part of
        the route proposal independently form each other
        
        Arguments: 
            * pending_rt_prop (:obj:`dict`): temporary information on the route proposal
            * is_fst_protocol_msg (bool): True if the message is carrying the encrypted pseudonym
             * c1 (:class:`~apart.crypto.crypto.Ctxt`): the first ciphertext to relay (either the pseudonym0 
                                        or the encrypted yes/no answer on the route acceptance)
            * c2 (:class:`~apart.crypto.crypto.Ctxt`): the second ciphertext to relay (always an encryption of one)
        """ 
                     
        # Craft a rt prop final messages, and place it in the right pool
        rt_prop_final_msg = LinkMsg(sent_by=self.id, sent_to=proposee, 
                                     c1=c1, c2=c2, flag=MsgFlag.RTPROP_FINAL, cid=cid,  
                                     seq_index=(1 if is_fst_protocol_msg else 2))

        self._place_msg_in_pool(proposee, rt_prop_final_msg)
        

    
    

    
    
    # ==========================================================================
    # Route proposals: Proposee, 2nd and FINAL step
    # ==========================================================================
    def _handle_rt_prop_final_msg(self, rt_prop_final_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.RTPROP_FINAL` messages  (received by proposees)
        
        This function merely waits for the two messages part of the route
        proposal to arrive, and calls :meth:`._proposee_process_rt_prop_final`.
        """
        # Similarly to rt prop and rt prop answer messages, a node expects two
        # rt prop final answer mesasges, in any order
        prev_cid = rt_prop_final_msg.cid
        prev_node = rt_prop_final_msg.sent_by
        is_fst_protocol_msg = (rt_prop_final_msg.seq_index == 1)
        assert (prev_node, prev_cid) in self._pending_rt_props['rcvd']
        pending_rt_prop = self._pending_rt_props['rcvd'][(prev_node, prev_cid)]
        
        if is_fst_protocol_msg:
            # This is the first rt prop final message
            pending_rt_prop['cpseudo'] = rt_prop_final_msg.c1
            if 'cacceptrt' in pending_rt_prop:
                self._proposee_process_rt_prop_final(pending_rt_prop)
        else:
            # This is the second rt prop final message
            pending_rt_prop['cacceptrt'] = rt_prop_final_msg.c1
            if 'cpseudo' in pending_rt_prop:
                self._proposee_process_rt_prop_final(pending_rt_prop)
        
   
    def _proposee_process_rt_prop_final(self, pending_rt_prop):     
        """Final processing of the route proposal by the proposee, and decision to accept the route or not
        
        The node, as a proposee, first sees it if accepts the route and if it
        must re-propose it or not (by calling
        :meth:`__route_proposal_policy_final_decision`). If not, it stops. If
        so, it inserts a new routing table entry, and sees if it reproposes it
        
        Arguments:
            * pending_rt_prop (:obj:`dict`): temporary stored information on the
                                    route proposal. Contains in particular the (encrypted) pseudonym
        """
        # First, get the pseudo and the yes/no answer stating that the node
        # accepts the route or not
        (_, sk_tmp) = pending_rt_prop['tmpkeypair']
        pseudo = Elgamal_dec(sk_tmp, pending_rt_prop['cpseudo'])
        acceptrt = Elgamal_dec(sk_tmp, pending_rt_prop['cacceptrt'])
        cid = pending_rt_prop['cid']
        
        # Run the route proposal policy
        accept, reprop, reason = self._route_proposal_policy_final_decision(pseudo, acceptrt)
        assert not accept and reason == RtPolicyReason.REFUSE_ITSELF or pending_rt_prop['additional_info']['actual_rcvr'] != self.id
        assert not accept or not (pending_rt_prop['additional_info']['actual_length'] > self._net.params.rtprop_policy_max_hop_count and reason is not RtPolicyReason.ACCEPT_FIRST_KNOWN_ROUTE)
        assert not accept or " {} ".format(self.id) not in pending_rt_prop['cwhoisontheroute'][0]
        
        if accept:
            logging.debug("Node {} is accepting a route towards node {} at round {} (reason : {})".format(self.id, pending_rt_prop['additional_info']['actual_rcvr'], self.batching_round, reason))
            
            # Insert rt entry
            rt_entry = {RT.PSEUDO: pseudo, RT.CONE: pending_rt_prop['cone'], 
                        RT.NEXT_NODE: pending_rt_prop['proposer'],  RT.NEXT_CID: cid, 
                        RT.TIMESTAMP:  self._net.timestamp, RT.REPROPOSED: (reprop is True),
                        RT.IN_USE: True, RT.ACTUAL_RCVR: pending_rt_prop['additional_info']['actual_rcvr'],
                        RT.ACTUAL_LENGTH: pending_rt_prop['additional_info']['actual_length']}
            rt_entry[RT.ROWID] = self.rt.insert_entry(**rt_entry)
            
            # If it is the *first* learned route towards the pseudo, do not
            # relay it straight away (unless we are at the beginning of the
            # network, in which case there is no choice: nodes only have routes
            # towards their neighbors)
            if reprop == 'later' and len(self.rt.lookup()) >= len(self.neighbors):
                self._pending_rt_props['to_repropose'][pseudo] = True
                self._env.process(self._delayed_relay_rt_prop(rt_entry, pending_rt_prop, delay=self._net.params.batching_t_interval*random.choice(range(200,350))))
            elif reprop:
                self._relay_rt_prop(rt_entry, pending_rt_prop)
        else:
            logging.debug("Node {} is refusing a route towards node {} at round {} (reason : {})".format(self.id, pending_rt_prop['additional_info']['actual_rcvr'], self.batching_round, reason))
        
        
        # Log the_rt prop
        self._net_manager.net_stats.log_rt_prop(self, reason, nb_rounds_to_complete=self.batching_round-pending_rt_prop['additional_info']['round_rt_prop_started'])
        
        del self._pending_rt_props['rcvd'][(pending_rt_prop['proposer'], cid)]
        
    def _route_proposal_policy_final_decision(self, pseudo, acceptrt):
        """Implementation of the route proposal policy: returns booleans indicating if a proposee must accept a route, and possibly re-propose it
        
        Depending on various critera, such as route length, number of routes
        already known, and so on, the route may be accepted or not. And if
        accepted, it may or may not be re-proposed by the node.
        
        Arguments:
            * pseudo (int): the pseudonym of the receiver of the route to accept (or refuse)
            * acceptrt (bool): the yes/no answer computed jointly by the
                    proposee and receiver, that depends on the length of the
                    route and whether the node is already on the route or not.
                    If ``acceprt is False``, the route will automatically
                    refused
        Returns:
            * (bool): True if the route is accepted
            * (bool): True if the route must be re-proposed
            * (str): The reason of acceptation or refusal of the route
        """
        if pseudo == SHA3_hash(pow(self._dst, self._src, GROUP_P)):
            return (False, False, RtPolicyReason.REFUSE_ITSELF) 
        

        # Get information on the routes already known towards the pseudonym
        routes_to_pseudo = self.rt.lookup(fields=[RT.TIMESTAMP, RT.REPROPOSED, RT.IN_USE], constraints={RT.PSEUDO: pseudo})
        

        if len(routes_to_pseudo) == 0:
            # If the node learns about the receiver for the first time, it
            # accepts straight away in **any** case. But mark to relay the
            # proposition "later", i.e. in a delayed fashion
            return (True, 'later', RtPolicyReason.ACCEPT_FIRST_KNOWN_ROUTE)
        elif not acceptrt: 
            # If the encrypted decision (on hop count and loops) says no,
            # then refuse the route
            return (False, False, RtPolicyReason.REFUSE_ENC_DEC)
#         elif sum(r[RT.IN_USE] for r in routes_to_pseudo) == self._net.params.rtprop_policy_max_routes:
#             return (False, False, RtPolicyReason.REFUSE_TOO_MANY_ROUTES_NO_REPLACEMENT)
#         else:
#             reason = RtPolicyReason.ACCEPT_REACCEPT_NO_REPLACEMENT
        elif not (random.uniform(0,1) <= self._net.params.rtprop_policy_p_reaccept):
            # Randomly refuse the route, because the node already has >= 1
            return (False, False, RtPolicyReason.REFUSE_REACCEPT)
        else:
            # Else, see if the new route is going to replace an existing one
#             if len(routes_to_pseudo) <= self._net.params.rtprop_policy_max_routes:
#                 effective_p_replace = self._net.params.rtprop_policy_p_replace
#             else:
            effective_p_replace = pow(self._net.params.rtprop_policy_p_replace, max(1, len(routes_to_pseudo)-self._net.params.rtprop_policy_max_routes))
            replace = (random.uniform(0,1) <= effective_p_replace)#self._net.params.rtprop_policy_p_replace)
            if not replace and sum(r[RT.IN_USE] for r in routes_to_pseudo) == self._net.params.rtprop_policy_max_routes:
                # If no replacement, AND we hit the maximum number of routes 
                # in use, refuse
                return (False, False, RtPolicyReason.REFUSE_TOO_MANY_ROUTES_NO_REPLACEMENT)
            elif not replace:
                # If no replacement, but the maximum number of routes is not 
                # reached, accept
                reason = RtPolicyReason.ACCEPT_REACCEPT_NO_REPLACEMENT
            else:
                # If remplacement, find and flag the oldest existing rt entry 
                # as "unused", meaning that it
                # is not used by the node for *sending*, but it is kept for
                # relaying (otherwise, routes break) 
                replaced_entry = min((r for r in routes_to_pseudo if r[RT.IN_USE]), key=lambda r: r[RT.TIMESTAMP])
                self.rt.update_entries(update={RT.IN_USE: False}, constraints={RT.ROWID: replaced_entry[RT.ROWID]})
                reason = RtPolicyReason.ACCEPT_REACCEPT_REPLACEMENT
                 
                
        
        # If the function arrives here, that means the route is accepted. We now
        # decide if the node must relay the proposition or not. Basically: yes 
        # if and only if the node never proposed this pseudonym            
        receiver_already_proposed = any(r[RT.REPROPOSED] for r in routes_to_pseudo)
        relay_prop = not receiver_already_proposed
        return (True, relay_prop, reason)
            
    def _relay_rt_prop(self, rt_entry, pending_rt_prop):
        """Called by a proposee to relay a route proposal
        
        If, according to the route proposal policy (see
        :meth:`._route_proposal_policy_final_decision`), a proposee must relay
        the route it just learned, this function makes some preparation before
        calling :meth:`._propose_route`.
        """
        # Lower the flag stipulating that the node must re-propose a route 
        # towards the pseudonym being proposed (mechanisms that is here to 
        #ensure connectivity
        self._pending_rt_props['to_repropose'].pop(rt_entry[RT.PSEUDO], None)
        
        # Re-encrypt and add sk
        cone = Reenc_one(Elgamal_key_mult(self._sk, pending_rt_prop['cone']))
        cprop = Reenc_nopk(cone, Elgamal_key_mult(self._sk, pending_rt_prop['cprop']))
        
        # Update the information on the route 
        chopcount =  Elgamal_plain_mult(pending_rt_prop['chopcount'], GROUP_G)
        chopcount = Reenc_nopk(cone, Elgamal_key_mult(self._sk, chopcount))
        
        cwhoisontheroute = Elgamal_accumulator_add(pending_rt_prop['cwhoisontheroute'], " {} ".format(self.id)) 
        cwhoisontheroute = Reenc_nopk(cone, Elgamal_key_mult(self._sk, cwhoisontheroute))
        
        self._propose_route(rt_entry, cprop, cone, chopcount, cwhoisontheroute, except_neighbor=pending_rt_prop['proposer'])
            
    def _delayed_relay_rt_prop(self, rt_entry, pending_rt_prop, delay):
        # Wait for the delay to pass
        yield self._env.timeout(delay)

        # Verify that the route still needs to be proposed. I.e. if by now, 
        # at least one route was proposed for this pseudo, connectivity is ensured, 
        # and there is no "necessity"
        if rt_entry[RT.PSEUDO] not in self._pending_rt_props['to_repropose']:
            return
        
        # Else, mark the entry as reproposed, and effectively relay the proposal
        self.rt.update_entries(update={RT.REPROPOSED: True}, constraints={RT.ROWID: rt_entry[RT.ROWID]})
        self._relay_rt_prop(rt_entry, pending_rt_prop)
        
          
            
            
                    
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    # Functions relating to the oriented communications, and commuinication
    # session initialisation
    # ==========================================================================
    # ##########################################################################
    # ==========================================================================
    def start_oriented_communications(self, ocom_sessions=None):
        """Public function, called by the network manager, to ask the node to start the oriented communication phase
        
        Start an oriented communication is started with each receiver (and
        a different helper each) specified in `ocom_sessions`.
        
        If the `ocom_sessions` argument is not provided, a random one is
        generated.
        
        The `ocom_sessions` argument can be of the following form:
            * None: the node then chooses a random number of random communication partners
            * `dict[r] = list(data)` indicating thatthe node must send the list of data to each receiver r 
            * `dict[r] = n` indicating that node must send n random pieces of data to each receiver r
            * `(n_r, n_data [, n_data2])` indicating that the node must choose n_r random receivers, and 
              send them n_data random messages each. If n_data2 is provided, then the node 
              chooses a random number of messages to send in [n_data, n_data2], different 
              for each receiver  
        
        Arguments:
            ocom_sessions(None or dict or tuple): the sessions to initiate 
               
        """ 
        if ocom_sessions is None or isinstance(ocom_sessions, tuple) and 2 <= len(ocom_sessions) <= 3:
            if ocom_sessions:
                nb_receivers = ocom_sessions[0]
                if len(ocom_sessions) == 2:
                    aux = ocom_sessions[1]
                    nb_data = lambda: aux
                else:
                    aux1 = ocom_sessions[1]
                    aux2 = ocom_sessions[2]
                    nb_data = lambda: random.randint(aux1, aux2) 
            else:
                nb_receivers = random.randint(1,5)
                nb_data = lambda: random.randint(1, 10)
            
            # Ensure that the below "while True" does not loop forever
            nb_receivers = min(nb_receivers, self._net.params.nb_nodes-1)
            

            ocom_sessions = []
            for _ in range(nb_receivers):
                while True:
                    receiver = random.randrange(self._net.params.nb_nodes)
                    if receiver != self.id:#and receiver not in ocom_sessions:
                        break
                ocom_sessions.append((receiver, ["data {}".format(i) for i in range(nb_data())]))
        elif (isinstance(ocom_sessions, list) and 
              all(isinstance(s, tuple) and len(s) == 2 and 
                  isinstance(s[0], int) and s[0] in range(self._net.params.nb_nodes) and
                  isinstance(s[1], (list, int))
                  for s in ocom_sessions)):
            
            for index, (receiver, data) in list(enumerate(ocom_sessions)):
                if isinstance(data, int):
                    ocom_sessions[index] = (receiver, ["data {}".format(i) for i in range(data)])
        else:
            raise ProtocolError('Node: oriented communication argument is not well formed ({})'.format(ocom_sessions))
    
            
        
#         print("Node {}, sessions = {}".format(self.id, ocom_sessions))
        for receiver, data in ocom_sessions:
            self._start_ocom_session(receiver, data)
            
    def _start_ocom_session(self, receiver, data_list):
        """Starts an oriented communication to send the specified pieces of data to the specified receiver.
        
        This function begins by choosing a random helper, then obtains the
        shares of the receiver's dst value, sends the first set of messages part
        of the ocom initialisation to the helper. It also initiates the
        temporary storage for information on the communication session.
        
        Arguments:
            * receiver (int): identity of the receiver
            * data_list (object list): a list of heterogeneous object to send (usually, strings) 
        """
        
        new_ocomid = self._net_manager.get_next_ocomid()
        
        # Choose a helper at random, by its pseudonym (generator used, avoiding
        # to fit all in memory)
        entries = self.rt.lookup(fields=[RT.CONE, RT.NEXT_NODE, RT.NEXT_CID, RT.ACTUAL_RCVR], 
                                            constraints=RT.NEXT_NODE+"!=? AND "+RT.IN_USE+"=?", 
                                            constraints_bindings=[F_NULL, True])
        
        # Choose an entry at random
        entry_helper = random.choice(entries)
        cone = entry_helper[RT.CONE]
        next_node = entry_helper[RT.NEXT_NODE]
        next_cid = entry_helper[RT.NEXT_CID]
                
        # Now, contact the receiver "offline" to get the shares. To model that,
        # we directly invoke the receiver node's function
        sh1, csh2 = self._net.nodes[receiver]._receiver_ocom_init_shares(new_ocomid, Reenc_one(cone))
        
        # Generate the other necessary material
        (pk_ocom, sk_ocom) = Elgamal_keygen(self._net.params.secparam)
        (pk_tmp, sk_tmp) = Elgamal_keygen(self._net.params.secparam)
        csh1 = Elgamal_enc(pk_ocom, sh1)
        msg_additional_info = {'is_ocom': True, 'end_sender': self.id, 'end_rcvr': entry_helper[RT.ACTUAL_RCVR], 
                               'ocom_helper': entry_helper[RT.ACTUAL_RCVR], 
                               'ocom_end_rcvr': receiver, 'ocom_end_sender': self.id}
        
        logging.debug("Node {} initiates an oriented communication ocomid({}) with helper {}, end-sender {}, and data = {}".format(self.id, new_ocomid, msg_additional_info['ocom_helper'], msg_additional_info['ocom_end_rcvr'], data_list))
        
        # Prepare the first set of  messages part of the oriented communication
        # initialisation: 4 payload messages, and 2 rt prop relay fwd ones      
        info = [pk_ocom, None, csh1[0], csh1[1]]
        for i in range(4):
            c1 = Elgamal_enc_nopk(cone, MsgInnerHeader(MsgInnerFlag.OCOM_INIT, ocomid=new_ocomid, seq_index=i))
            c2 = Elgamal_enc_nopk(cone, info[i]) if i != 1 else Reenc_nopk(cone, csh2)
            m = LinkMsg(sent_by=self.id, sent_to=next_node, 
                         c1=c1, c2=c2, 
                         flag=MsgFlag.PAYLOAD, cid=next_cid,
                         additional_info=msg_additional_info)
            self._place_msg_in_pool(next_node, m)
        
        # For the helper to be able to answer, we send rtproprelay fwd messages.
        # This is done differently as described in the thesis, because here the
        # implementation of the route proposal policy directly gives us two
        # rtproprelay fwd msg (no need to send two with different rcid)
        rcid = self._net_manager.get_next_rcid(self.id, next_node)
        c1 = Elgamal_enc_nopk(cone, MsgInnerHeader(MsgInnerFlag.OCOM_INIT, ocomid=new_ocomid, seq_index=4))
        c2 = Elgamal_enc_nopk(cone, pk_tmp)
        m1 = LinkMsg(sent_by=self.id, sent_to=next_node, 
                     c1=c1, c2=c2, 
                     flag=MsgFlag.RTPROP_RELAY_FWD, cid=next_cid, rcid=rcid, seq_index=1,
                     additional_info=msg_additional_info)
        m2 = LinkMsg(sent_by=self.id, sent_to=next_node, 
                     c1=c1, c2=c2, 
                     flag=MsgFlag.RTPROP_RELAY_FWD, cid=next_cid, rcid=rcid, seq_index=2,
                     additional_info=msg_additional_info)
        self._place_msg_in_pool(next_node, m1, m2)
        
        
        # Store the info on the ocom init
        self._ongoing_ocom['sender'][new_ocomid] = {'end_rcvr': receiver, 'helper': entry_helper[RT.ACTUAL_RCVR],
                                                    'data_list': data_list,
                                                    'route_to_helper': {'node': entry_helper[RT.NEXT_NODE], 
                                                                        'cid': entry_helper[RT.NEXT_CID],
                                                                        'rcid': rcid ,
                                                                        'cone': cone},
                                                    'sk_ocom': sk_ocom,
                                                    'sk_tmp': sk_tmp}
        
        # This second storage is for when the sender will receive back the rt
        # prop relay bwd msg, to easily recognise that it is part of an ocomm
        # init
        self._ongoing_ocom['sender'][('rt_prop_relay_bwd_shortcut', entry_helper[RT.NEXT_NODE], entry_helper[RT.NEXT_CID], rcid)] = new_ocomid
        
        # For debugging purposes, also specify to the receiver what it is
        # supposed to receive. This is done directly, by access to the
        # :obj:`~apart.core.node.Node` object of the receiver.
        self._net.nodes[receiver]._ongoing_ocom['to_receive'][new_ocomid] = {'data_list': set(data_list),
                                                                             'started_at_round': self.batching_round}
        
        # Log the ocom route for statistics and measures
        self._net_manager.net_stats.log_ocom_route(self, new_ocomid, end_sender=self.id, 
                                                   helper=self._ongoing_ocom['sender'][new_ocomid]['helper'],
                                                   end_rcvr=self._ongoing_ocom['sender'][new_ocomid]['end_rcvr'],
                                                   first_hop_end_sender=dict((k, v) for k, v in self._ongoing_ocom['sender'][new_ocomid]['route_to_helper'].items() if k in ['node', 'cid']))

        
        
        
    def _receiver_ocom_init_shares(self, new_ocomid, helper_cone):
        """Computes the share of the receivers's dst values
        
        This function basically models the *offline* exchange that an end-sender
        and end-receiver must perform prior to any oriented communication.
        
        Arguments:
            * new_ocomid (int): the oriented communication id chosen by the end-sender
            * helper_cone (:obj:`~apart.crypto.crypto.Ctxt`): the end-sender's encryption
                                            of one towards the helper she chose
        
        Returns:
            * (int): the first share of the receiver's dst value
            * (:obj:`~apart.crypto.crypto.Ctxt`): the second share of the
                        receiver's dst value, encrypted using helper_cone
        """
        sh1 = random.randint(1, GROUP_P-1)
        sh2 = self._dst*group_inverse(sh1) % GROUP_P
        
        return sh1, Elgamal_enc_nopk(helper_cone, sh2)
        
        
            
        
    def _handle_payload_msg(self, payload_msg):
        """Handler of :attr:`~apart.core.messages.MsgFlag.PAYLOAD` messages
        
        A node that receives a :attr:`~apart.core.messages.MsgFlag.PAYLOAD`
        message is either a relay, or its intended receiver. This is determined
        by a routing table lookup for the "next hop". 
        
        A relay node will simply send the message forward on the route. A
        receiver has several choices. The payload message can be of several
        type: an end-to-end dummy, an actual payload, or a routing message part
        of an oriented communication initialisation. This function then calls
        the appropriate *inner message handler*. 
        
        Note: end-to-end dummies are detected early by the receiver,and do not
        have their own inner message handler
        
        Raises:
            EndToEndDummyError: if the payload message is actually an end-to-end
                    dummy. This ugly fix is here to allow counting the number of
                    received e2e dummy in
                    :meth:`._round_process_received_messages`.
        """ 
        # Fail early: instead of a rt lookup to see if we are the receiver,
        # first boldly try to decrypt the message and test the inner flag 
        try:
            if Elgamal_dec(self._sk, payload_msg.c1).flag is MsgInnerFlag.DUMMY:
                raise EndToEndDummyError("Payload message is an end-to-end dummy")
        except (DecryptionError, AttributeError):
            pass
        
        entries = self.rt.joint_lookup(PRT, fields=[RT.NEXT_NODE, RT.NEXT_CID], join_on=(RT.ROWID, PRT.RT_ROWID), 
                             constraints={PRT.PREV_NODE: payload_msg.sent_by, PRT.PREV_CID: payload_msg.cid})
        assert len(entries) == 1, "Entry = {}, Msg = {}".format([[v for v in r] for r in entries], payload_msg)
        entry = entries[0]
        
        if entry[RT.NEXT_NODE] == F_NULL:
            # The node is the receiver. Its action depends on the inner flag in
            # the first ciphertext of the message.
            inner_header = Elgamal_dec(self._sk, payload_msg.c1)
            
            try:
                inner_flag = inner_header.flag
            except AttributeError:
                # This means it is a "regular" payload message, with plain data in it.
                # In a real implementation, we would send the data to the application layer
                assert False, "Unknown inner message type. Message is {}".format(payload_msg)
                # Because assertions are not run when python is invoked with
                # "-O", we still make a failover
                return
            
            Node.__payload_msgs_handler[inner_flag](self, payload_msg, inner_header)  
        else:
            # The node is a simple relay 
            c2 = Reenc_one(Elgamal_key_div(self._sk, payload_msg.c2))
            c1 = Reenc_nopk(c2, Elgamal_key_div(self._sk, payload_msg.c1))
            new_payload_msg = LinkMsg(sent_by=self.id, sent_to=entry[RT.NEXT_NODE], 
                                      c1=c1, c2=c2, 
                                      flag=MsgFlag.PAYLOAD, cid=entry[RT.NEXT_CID], 
                                      additional_info=payload_msg.additional_info)
            
            self._place_msg_in_pool(entry[RT.NEXT_NODE], new_payload_msg)
            
    def _helper_process_ocom_init_msg(self, msg, inner_header):
        """Models the behavior of the helper in an oriented communication.
        
        In an oriented communication, the first seven messages are control ones,
        aimed at initializing the route (i.e. the send-senderand the helper
        computing the helper's pseudonym towards the end-receiver). Then, there
        are one more message per piece of data that the end-sender wishes to
        send.
        
        This funtion processes all these messages, managing a temporary storage
        to link them all together.
        
        Arguments:
            * msg (:obj:`~apart.core.messages.LinkMsg`): the received mesasge.
                        Can be a :attr:`~apart.core.messages.MsgFlag.PAYLOAD`
                        message, or a
                        :attr:`~apart.core.messages.MsgFlag.RTPROP_RELAY_FWD`
                        one.
            * inner_header (:obj:`~apart.core.message.MsgInnerHeader`): the
                        (decrypted) inner header,containing information on the
                        oriented communication session
        """
        node_to_sender = msg.sent_by
        cid_to_sender = msg.cid
        ocomid = inner_header.ocomid
        seq_index = inner_header.seq_index
        data = Elgamal_dec(self._sk, msg.c2)
        
         
        try:
            ongoing_ocom = self._ongoing_ocom['helper'][ocomid]
        except KeyError:
            self._ongoing_ocom['helper'][ocomid] = {'nb_msgs_relayed': 0,  
                                                    'ocom_end_rcvr': msg.additional_info['ocom_end_rcvr'], 
                                                    'ocom_end_sender': msg.additional_info['ocom_end_sender']}
            ongoing_ocom = self._ongoing_ocom['helper'][ocomid]
            
        if seq_index <= 4:
            # Wait to receive all the six first messages sent by the end-sender,
            # and when it is the case, send back two rt prop relay bwd msgs
            if seq_index == 4:
                if msg.seq_index == 1:
                    ongoing_ocom['pk_tmp'] = data
                else:
                    ongoing_ocom['rcid'] = msg.rcid
                    
            else:
                data_name = ['pk_sender_ocom', 'sh2', 'csh1_0', 'csh1_1']
                ongoing_ocom[data_name[seq_index]] = data
            
            # When all six messages from the ocom have been received, the helper can
            # start answering
            if all(d in ongoing_ocom for d in ['pk_sender_ocom', 'sh2', 'csh1_0', 'csh1_1', 'rcid', 'pk_tmp']):
                c = (ongoing_ocom.pop('csh1_0'), ongoing_ocom.pop('csh1_1'))
                c = Elgamal_scalar_exp(Elgamal_plain_mult(c, ongoing_ocom.pop('sh2')), self._src)
                c = Reenc_pk(ongoing_ocom.pop('pk_sender_ocom'), c)
                
                pk_tmp = ongoing_ocom.pop('pk_tmp')
                c1 = Elgamal_enc(pk_tmp, c[0])
                c2 = Elgamal_enc(pk_tmp, 1)
                c3 = Elgamal_enc(pk_tmp, c[1])
                c4 = Elgamal_enc(pk_tmp, 1)
                
                rcid = ongoing_ocom.pop('rcid')
                additional_info = copy.deepcopy(msg.additional_info)
                additional_info.update({'end_sender': self.id, 'end_rcvr': ongoing_ocom['ocom_end_sender']})
                self._rt_prop_return_trip_relay_bwd(True, node_to_sender, cid_to_sender, rcid, 
                                                    c1,c2, None, additional_info=additional_info)
                self._rt_prop_return_trip_relay_bwd(False, node_to_sender, cid_to_sender, rcid, 
                                                    c3,c4, None, additional_info=additional_info)
                
        elif seq_index == 7:
            # This is the payload message carrying the helper's pseudonym
            # towards the end-receiver
            
            # Choose a next hop for this pseudo. Note: the following code
            # handles well the special case when the slected helper is actually
            # the end-receiver.
            entries = self.rt.lookup(fields=[RT.CONE, RT.NEXT_NODE, RT.NEXT_CID],
                                     constraints={RT.PSEUDO: data, RT.IN_USE: True})
            assert len(entries) > 0, "pseudo = {}".format(data, self.display_node_table(table="rt"))
            entry = random.choice(entries)
            
            # Store information on the route
            ongoing_ocom['route_to_receiver'] = {'node': entry[RT.NEXT_NODE],
                                                 'cid': entry[RT.NEXT_CID],
                                                 'cone': entry[RT.CONE]}
            
            # Log the ocom route for statistics and measures
            self._net_manager.net_stats.log_ocom_route(self, ocomid, end_sender=ongoing_ocom['ocom_end_sender'],
                                                   helper=self.id,
                                                   end_rcvr=ongoing_ocom['ocom_end_rcvr'],
                                                   first_hop_helper=dict((k, v) for k, v in ongoing_ocom['route_to_receiver'].items() if k in ['node', 'cid']))
            
            data_msg_buffer = ongoing_ocom.get('data_msg_buffer', [])
            for waiting_seq_index, waiting_data in data_msg_buffer:
                self._helper_relay_payload(ocomid, ongoing_ocom, waiting_seq_index, waiting_data)
        else:
            # An actual message carrying a payload, and with a seq_index > 7.
            # Note that the helper may have received such a message BEFORE the
            # message with seq_index == 7, i.e. the one that contains the pseudo. 
            if 'route_to_receiver' not in ongoing_ocom:
                # If the helper does not yet hnows the route towards the end-receiver,
                # insert the message in a waiting list
                if 'data_msg_buffer' not in ongoing_ocom:
                    ongoing_ocom['data_msg_buffer'] = []
                ongoing_ocom['data_msg_buffer'].append((seq_index-8, data))
            else:
                # Else, simply relay the message
                self._helper_relay_payload(ocomid, ongoing_ocom, seq_index-8, data)
    
    
    def _helper_process_ocom_close_msg(self, payload_msg, inner_header):
        """Processes the message received by the helper, that indicates the end of an oriented communication session.
        
        For the helper to know when it is ok to delete the temporary storage
        related to a communication session, the end-sender sends a
        :attr:`~apart.core.messages.MsgInnerFlag.OCOM_CLOSE` message
        (encapsulated in a :attr:`~apart.core.messages.MsgFlag.PAYLOAD`
        message). It tells how many messages the helper needs to forward in
        total.
        
        Note that, by the way forwarding works, this "close" message may arrive
        before the helper even gets all the messages containing the data to
        relay. Thus, this function does not automatically deletes the temporary
        storage. In some cases, it only raises a flag, saying *when* it should
        be deleted.
        
        Arguments:
            * payload_msg (:obj:`~apart.core.messages.LinkMsg`): the received 
                        :attr:`~apart.core.messages.MsgFlag.PAYLOAD` message.
            * inner_header (:obj:`~apart.core.message.MsgInnerHeader`): the
                        (decrypted) inner header,containing information on the
                        oriented communication session
        """
        
        ocomid = inner_header.ocomid
        ongoing_ocom = self._ongoing_ocom['helper'][ocomid]
        nb_msgs_total = Elgamal_dec(self._sk, payload_msg.c2)
        
        # Delete the temporary storage only if all messages have 
        # been forwarded to the receiver. Not always the case, since 
        # messages arrive in unpredictable order  
        if 'nb_msgs_relayed' in ongoing_ocom and ongoing_ocom['nb_msgs_relayed'] == nb_msgs_total:
            del self._ongoing_ocom['helper'][ocomid]
        else:
            ongoing_ocom['nb_msgs_total'] = nb_msgs_total
            
    def _helper_relay_payload(self, ocomid, ongoing_ocom, seq_index, data):
        """The relaying by the helper of a payload message in an oriented communication
        
        This function simply crafts a payload message, and sends it on the pre-
        selectionned route to the receiver (the description of the route is in
        the ``ongoing_ocom`` argument).
        
        Arguments:
            * ocomid (int): the id of the communication session
            * ongoing_ocom (:obj:dict`): temporary stored info on the 
                            communication session
            * seq_index(int): the sequence number of the payload 
            * data (object): the data to send
        """
        next_node = ongoing_ocom['route_to_receiver']['node']
        next_cid = ongoing_ocom['route_to_receiver']['cid']
        cone = ongoing_ocom['route_to_receiver']['cone']
        additional_info = {'is_ocom': True, 'is_ocom_payload': True, 
                           'end_sender': self.id, 'end_rcvr': ongoing_ocom['ocom_end_rcvr'], 
                           'ocom_helper': self.id, 'ocom_end_sender': ongoing_ocom['ocom_end_sender'], 'ocom_end_rcvr': ongoing_ocom['ocom_end_rcvr']}
        
        if next_node == F_NULL:
            # Special case: the helper is actually the end-receiver !
            additional_info['end_sender'] = additional_info['ocom_end_sender']
            self._receiver_process_ocom_rcv_msg(ocomid, seq_index, data, additional_info=additional_info)
        else:
            # General case
            c1 = Elgamal_enc_nopk(cone, MsgInnerHeader(MsgInnerFlag.OCOM_RCV, ocomid, seq_index=seq_index))
            c2 = Elgamal_enc_nopk(cone, data)
            m = LinkMsg(sent_by=self.id, sent_to=next_node, 
                        c1=c1, c2=c2, 
                        flag=MsgFlag.PAYLOAD, cid=next_cid, 
                        additional_info=additional_info)           
            
            self._place_msg_in_pool(next_node, m)      
        
        # Update the number of messages sent towards te receiver
        ongoing_ocom['nb_msgs_relayed'] += 1 
        
        # Verify if this was the last message to relay. If so, 
        # delete the temporary storage
        if ongoing_ocom['nb_msgs_relayed'] == ongoing_ocom.get('nb_msgs_total', -1):
            del self._ongoing_ocom['helper'][ocomid]
    
    def _sender_ocom_init_finalise(self, ocomid, rt_prop_relay_bwd_msg):
        """Second step for the sender in an oriented communication.
        
        After starting the ocom init (by sending the first 4
        :attr:`~apart.core.messages.MsgFlag.PAYLOAD` messages and 2
        :attr:`~apart.core.messages.MsgFlag.RT_PROP_RELAY_FWD` ones to the
        helper), the sender expects to receive two
        :attr:`~apart.core.messages.MsgFlag.RT_PROP_RELAY_BWD` messages, each
        containing a piece of the ciphertext encrypting dstR**srcI.
        
        This allows the end-sender to compute the helper's pseudonym towards the
        end-receiver, to send it back, and to start sending the pieces of data.
        
        At the end of this function, the temporary storage of the end-sender
        concerning the ocom session is deleted.
        
        Arguments:
            * ocomid (int): the id of the communication session
            * rt_prop_relay_bwd_msg (:obj:dict`): the received 
                        :attr:`~apart.core.messages.MsgFlag.RT_PROP_RELAY_BWD`
                        message
        """
        
        
        ongoing_ocom = self._ongoing_ocom['sender'][ocomid]
        
        
        cpseudo_part = Elgamal_dec(ongoing_ocom['sk_tmp'], rt_prop_relay_bwd_msg.c1)
        if rt_prop_relay_bwd_msg.seq_index == 1:
            ongoing_ocom['cpseudo_0'] = cpseudo_part
        else:
            ongoing_ocom['cpseudo_1'] = cpseudo_part
        
        # When both rt prop relay bwd messages have been receiver,
        # process them, and get back to the helper
        if 'cpseudo_0' in ongoing_ocom and 'cpseudo_1' in ongoing_ocom:
            cpseudo = (ongoing_ocom['cpseudo_0'], ongoing_ocom['cpseudo_1'])
            pseudo = SHA3_hash(Elgamal_dec(ongoing_ocom['sk_ocom'], cpseudo))
            cone = ongoing_ocom['route_to_helper']['cone']
            next_node = ongoing_ocom['route_to_helper']['node']
            next_cid = ongoing_ocom['route_to_helper']['cid']            
            additional_info = copy.deepcopy(rt_prop_relay_bwd_msg.additional_info)
            additional_info.update({'end_sender': self.id, 'end_rcvr': ongoing_ocom['helper']}) 
            
            # Prepare the message containing the helper's pseudonym towards the receiver
            c1 = Elgamal_enc_nopk(cone, MsgInnerHeader(MsgInnerFlag.OCOM_INIT, ocomid, seq_index=7))
            c2 = Elgamal_enc_nopk(cone, pseudo)
            m_pseudo = LinkMsg(sent_by=self.id, sent_to=next_node, 
                               c1=c1, c2=c2, 
                               flag=MsgFlag.PAYLOAD, cid=next_cid,
                               additional_info=additional_info)
            self._place_msg_in_pool(next_node, m_pseudo)
            
            # Prepare one payload message for each piece of data to send
            # to the end receiver
            additional_info = copy.deepcopy(additional_info)
            additional_info['is_ocom_payload'] = True 
            for i, data in enumerate(ongoing_ocom['data_list']):
                c1 = Elgamal_enc_nopk(cone, MsgInnerHeader(MsgInnerFlag.OCOM_INIT, ocomid, seq_index=8+i))
                c2 = Elgamal_enc_nopk(cone, data) # If AES encryption is implemented, "data" should be encrypted here
                m = LinkMsg(sent_by=self.id, sent_to=next_node, 
                            c1=c1, c2=c2, 
                            flag=MsgFlag.PAYLOAD, cid=next_cid,
                            additional_info=additional_info) 
                self._place_msg_in_pool(next_node, m)
                
            # Send the special message that closes the session (i.e. 
            # that lets the helper know the session is over)
            additional_info = copy.deepcopy(additional_info)
            del additional_info['is_ocom_payload']  
            c1 = Elgamal_enc_nopk(cone, MsgInnerHeader(MsgInnerFlag.OCOM_CLOSE, ocomid))
            c2 = Elgamal_enc_nopk(cone, len(ongoing_ocom['data_list'])) 
            m = LinkMsg(sent_by=self.id, sent_to=next_node, 
                        c1=c1, c2=c2, 
                        flag=MsgFlag.PAYLOAD, cid=next_cid,
                        additional_info=additional_info) 
            self._place_msg_in_pool(next_node, m)     
            
            # For debug purposes, inform the receiver of when the actual ocom starts
            self._net.nodes[ongoing_ocom['end_rcvr']]._ongoing_ocom['to_receive'][ocomid].update({'first_payload_sent_at_round': self.batching_round})
        
            # Delete all temporary stored information
            del self._ongoing_ocom['sender'][('rt_prop_relay_bwd_shortcut', rt_prop_relay_bwd_msg.sent_by, rt_prop_relay_bwd_msg.cid, ongoing_ocom['route_to_helper']['rcid'])]
            del self._ongoing_ocom['sender'][ocomid]
            
    
             
    
    def _handle_payload_ocom_rcv_msg(self, payload_msg, inner_header):
        """Handler of :attr:`~apart.core.messages.MsgInnerFlag.OCOM_RCV` message, for the end-receiver of oriented communications
        
        Arguments:
            * payload_msg (:obj:`~apart.core.messages.LinkMsg`): the received 
                        :attr:`~apart.core.messages.MsgFlag.PAYLOAD` message.
            * inner_header (:obj:`~apart.core.message.MsgInnerHeader`): the
                        (decrypted) inner header,containing information on the
                        oriented communication session
        """
        ocomid = inner_header.ocomid
        seq_index = inner_header;seq_index
        data = Elgamal_dec(self._sk, payload_msg.c2)
        self._receiver_process_ocom_rcv_msg(ocomid, seq_index, data, payload_msg.additional_info)
        

    def _receiver_process_ocom_rcv_msg(self, ocomid, seq_index, data, additional_info):
        """Receiving of actual data from an oriented communication.
        
        This is the function where an end-receiver of an oriented communication
        processes the data it receives. Recall that data is contained (only) in
        the second ciphertext of :attr:`~apart.core.messages.MsgFlag.PAYLOAD`,
        the first one containing an inner header allowing to re-order messages.
        
        In an acutal implementation, the data received should be passed on to
        the *application layer*. Here, all this function does is logging the
        received message.
        
        Arguments:
            * ocomid (int): the identifier of the communication session
            * seq_index (int): sequence number of the piece of data 
            * data (object): the piece of data to receive
            * additional_info (:obj:`dict` or None): optional additional 
                        information on the communication session 
        """
        logging.debug("Node {} end-received '{}' as part of an oriented communication ocomid({}) with end-sender {}".format(self.id, data, ocomid, additional_info['end_sender']))
        
#         self._net_manager.net_stats.log_rcvd_ocom_msg(self, data)
        
        assert self.id == additional_info['ocom_end_rcvr']
        ongoing_ocom = self._ongoing_ocom['to_receive'][ocomid]
        ongoing_ocom['data_list'].remove(data)
        if len(ongoing_ocom['data_list']) == 0:
            self._net_manager.net_stats.log_ocom_latency(self, additional_info['end_sender'], 
                                                         ongoing_ocom['started_at_round'],
                                                         ongoing_ocom['first_payload_sent_at_round'])
            del self._ongoing_ocom['to_receive'][ocomid]
        
        




    #===========================================================================
    # Misc. functions
    #===========================================================================
    def display_node_table(self, table="rt+prt", constraints={}):
        """Print one or sevral of the node's routing tables. Used for debug purposes"""
        print("I am node {}, at round {} (ime is {}), and here is my routing table ({}):".format(self.id, self.batching_round, self._net.timestamp, table))
        

        if table == "rt+prt":
            fields = [PRT.PREV_NODE, PRT.PREV_CID] + \
                      [f for f in RT.fields if f != RT.NODE]
            join_on = (PRT.RT_ROWID, RT.ROWID)
            contents = self.rt.joint_lookup(PRT, fields=fields, join_on=join_on, order_by=[RT.ACTUAL_RCVR, PRT.PREV_NODE])
            pprinttable(fields, contents)
            
        elif table == "rt":
            pprinttable(RT.fields,  self.rt.lookup(constraints=constraints, order_by=[RT.ACTUAL_RCVR]))
        elif table == "prt":
            pprinttable(PRT.fields, self.prt.lookup(constraints=constraints, order_by=[PRT.RT_ROWID]))

        print()
    
    
    # Dict mapping message type to node function to process them. Put as static
    # member to avoid re-constructing it at each Node instance (or even each
    # call of function)
    __msg_handlers = {MsgFlag.RTPROP: _handle_rt_prop_msg,
                      MsgFlag.RTPROP_ANSWER: _handle_rt_prop_answer_msg,
                      MsgFlag.RTPROP_FINAL: _handle_rt_prop_final_msg,
                      MsgFlag.RTPROP_RELAY_FWD: _handle_rt_prop_relay_fwd_msg,
                      MsgFlag.RTPROP_RELAY_BWD: _handle_rt_prop_relay_bwd_msg,
                      MsgFlag.PAYLOAD: _handle_payload_msg}

    __payload_msgs_handler = {MsgInnerFlag.OCOM_INIT: _helper_process_ocom_init_msg,
                              MsgInnerFlag.OCOM_CLOSE: _helper_process_ocom_close_msg,
                              MsgInnerFlag.OCOM_RCV: _handle_payload_ocom_rcv_msg}



class ProtocolError(Exception):
    """Exception raised when an error occurs in the protocol"""
    pass