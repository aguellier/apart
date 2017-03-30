# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
This module contains classes managing the network. 

The class :class:`~apart.core.network.Network` represents a network, and class
:class:`~apart.core.network.NetworkManager` the manager of the network.The
manager is responsible for triggering the different *phases* of the network
(coded in :class:`~apart.core.network.NetPhase`), and starting the nodes
processes. The network itself contains the topology graph and the list of nodes
that compose the network.

This module does not contain code properly relatin to the protocol. All the code
of the protocol, and the behavior of nodes, is coded in module
:mod:`~apart.core.node`.

This module also defines the :exc:`.SimulationException` exception.
'''

from collections import defaultdict
from enum import IntEnum
import graphviz
import itertools
import math
import os
import random
import simpy
from threading import Thread
import time

from common.custom_logging import *
from apart.core.node import Node
from apart.core.tables import SqliteRoutingTable, RoutingTable as RT, PrevRoutingTable as PRT
import collections as c
import matplotlib.pyplot as plt
from measures.network_statistics import NetworkStats
import networkx as nx


class Network(object):

    
    __next_network_uid = 1
    # A counter to keep track of network unique identifiers. Used for managing
    # the weird routing table implementation (see apart.core.tables)

    def __init__(self, net_manager, simpy_env, sim_params, net_params):
        """The network which nodes are part of.

        This class is repsonsible for:
            * Managing the set of nodes
            * Creating and managing the topology graph
        
        The constructor creates the topology graph, creates the nodes 
        (:obj:`~apart.core.node.Node` objects), generates a subset of 
        corrupted nodes,
        
        Arguments:
            net_manager (:obj:`~apart.core.network.NetworManager`): the manager of 
                        the network
            simpy_env (:obj:`simpy.Environment`): the simpy environment
            sim_params (:obj:`~simulation.simulation.SimulationParams`): set of 
                        parameters for the simulation
            net_params (:obj:`~apart.core.network.NetworkParams`): set of 
                        parameters for the network (nb nodes, etc.)
        """
        self._network_uid = str(round(time.time()))+"_"+str(Network.__next_network_uid)
        """The network identity, uniquely identifying the network run (and the set of nodes therein)"""
        Network.__next_network_uid += 1
        
        self._env = simpy_env
        """The SimPy environment"""
        
        self._sim_params = sim_params
        """Parameters of the simulation"""
        
        self._params = net_params
        """Network parameters"""
        
        self._net_manager = net_manager
        """The network manager""" 
        
        if isinstance(self.params.topology_graph, nx.classes.graph.Graph):
            if len(self.params.topology_graph) == self.params.nb_nodes:
                self._topology_graph = self.params.topology_graph
            else:
                raise SimulationException("The provided topology graph has {} nodes, while {} were expected.".format(len(self.params.topology_graph), self.params.nb_nodes))
        else:
            self._topology_graph = Network.gen_topology_graph(self.params.nb_nodes)     
            """The topology graph, i.e. graph connecting the nodes"""
        logging.info("Topology graph has {0} nodes with {1} neighbors each".format(self.params.nb_nodes, self._topology_graph.degree(0)))
        
        # Set the maximum route length, if not specified by the parameters
        if not self.params.rtprop_policy_max_hop_count:
            all_shortest_paths_lengths = nx.shortest_path_length(self._topology_graph)
            self.params.rtprop_policy_max_hop_count = 1+max(itertools.chain(*([l for l in d.values()]
                                                                            for d in all_shortest_paths_lengths.values())))
        
        # Draw the network graph at startup if asked to
        if self._sim_params.draw_topology_graph:
            self.draw_topology_graph()
        
        
        self._nodes = [Node(self._env, self, self._net_manager, n, self._topology_graph.neighbors(n)) for n in range(self.params.nb_nodes)]
        """The list of :obj:`~apart.core.node.Node` objects"""
        
        # Among those nodes, mark some of them as corrupted
        self._corrupted_nodes_indexes = Network.gen_corruption_states(self.nb_nodes, self.params.corruption_ratio, self._topology_graph)
        """The indexes of the corrupted nodes"""
        
        # Inform node of their corruption
        for n in self._corrupted_nodes_indexes: self._nodes[n].set_corruption_state(corrupted=True)
        
        assert all(node.is_corrupted or any(not self._nodes[neigh].is_corrupted for neigh in node.neighbors) for node in self._nodes)

         
    @property
    def network_uid(self):
        """int: The network unique id"""
        return self._network_uid
    
    @property
    def nodes(self):
        """list of :obj:`~apart.core.node.Node`: The list of nodes in the network. The list is passed by reference, but should not be modified. Nodes inside the list can be modified."""
        return self._nodes
    
    @property
    def corrupted_nodes(self):
        """list of :obj:`~apart.core.node.Node`: The set of corrupted nodes"""
        return [self._nodes[i] for i in self._corrupted_nodes_indexes]
    
    @property
    def nb_nodes(self):
        """int: Number of nodes in the networks"""
        return len(self._nodes)
    
    @property
    def params(self):
        """:obj:`apart.core.network.NetworkParams`: The network parameters"""
        return self._params
    
    @property
    def topology_graph(self):
        """:obj:`networkx.Graph`: The network's underlying topology graph"""
        return self._topology_graph
    
    @property
    def routing_tables_sqlite_db_cursor(self):
        """:obj:`sqlite3.Cursor`: The sqlite3 database cursor internally used by the node's routing table implementations
        
        For statistics computing, and inspection of routing tables after the run
        of the network, it is sometimes useful to directly access the sqlite3 cursor.
        """
        return SqliteRoutingTable.sqlite_db_cursor(self.network_uid)
    
        
    @property
    def timestamp(self):
        """int: Returns the SimPy simulation time, rounded to the next integer"""
        return round(self._env.now)
    
    @property
    def batching_round(self):
        """int: Returns the current network batching round
        
        In a real-world network, each node would  manage its own batching
        counter. Although all nodes have the same batching interval, skewing in
        their clocks imply that different nodes can be in different rounds (all
        nodes are not perfectly in sync). This function returns the maximum
        batching round over all nodes.
        
        Note that in this "implementation" with SimPy, all node clocks are
        perfectly in sync, and thus all nodes are exactly in the same batching round.
        """ 
        return max(n.batching_round for n in self.nodes)
        
    @staticmethod
    def gen_topology_graph(nb_nodes):
        """Makes a random (but fully connected) physical graph of the appropriate number of nodes. 
        
        Uses the networkx graph library.
        
        Returns:
            :obj:`networkx.Graph`: a topology graph (random connected graph)
        """
        num_neighbors = max(3, math.ceil(math.log(nb_nodes, 2)))

        # Generate a random graph (networkx package requires n*d to be even)
        if (num_neighbors * nb_nodes)%2 != 0: num_neighbors += 1
        physical_graph = nx.random_regular_graph(num_neighbors, nb_nodes)
        #physical_graph = nx.connected_watts_strogatz_graph(self.params.nb_nodes, num_neighbors, 0.8)
        
        # Check connectivity
        if not nx.is_connected(physical_graph):
                logging.warning("The physical graph is not connected!")
        
        return physical_graph
    
    @staticmethod
    def gen_corruption_states(nb_nodes, corruption_ratio, topology_graph):
        """Generates a random set of corrupted nodes. 
        
        Args:
            nb_nodes (int): total number of nodes in the network
            corruption_ratio (float): float between 0 and 1, representing
                    the percent of corrupted nodes in the network
            topology_graph (:obj:`networkx.Graph`): the topology graph of the 
                    network
        """
        for i in range(100):
            corrupted_nodes_indexes = random.sample(range(nb_nodes), k=math.floor(nb_nodes*corruption_ratio))
            
            # Check the base assumption: that each node has at least one honest
            # neighbor
            for n in range(nb_nodes):
                if (n not in corrupted_nodes_indexes 
                    and all(neigh in corrupted_nodes_indexes for neigh in topology_graph.neighbors(n))):
                    break
            else:
                return corrupted_nodes_indexes
            
            
        else:
            raise SimulationException("Impossible to find a network corruption"
                                      " state with a ratio of {} and such that"
                                      " each honest node has at least one honest"
                                      " neighbor".format(corruption_ratio))
        
        
    
    
    def draw_topology_graph(self, blocking=True):
        """Draw the topology graph of the network
        
        Creates a new gtk window, and displays the graph in it. This function
        can be called in a *blocking* fashion or not, in which case the program
        (and the network simulation) are stopped while the gtk window is open.2
        
        Arguments:
            * blocking (bool, optional): whether or not the python program
                            should be stopped until the gtk window is closed, or
                            should continue to run in the background
        """
        def do_draw():
            nx.draw_networkx(self.physical_graph)#, pos=nx.get_node_attributes(self.physical_graph, "coords"))
            plt.show()

        if blocking:
            do_draw()
        else:
            class PlotThread(Thread):
                def __init__(self, physical_graph):
                    Thread.__init__(self)
                    #---------------------- self.physical_graph = physical_graph
                def run(self):
                    do_draw()
            t = PlotThread(self.physical_graph)
            t.start()
            
            

    def draw_routes(self, *args, blocking=False):
        """Draw the routes between one or several pairs of nodes
        
        For each sender-receiver pair in argument, the function will draw all
        the routes that the sender knows towards the receiver.
        
        
        Arguments:
            args(list of 2-tuple):  a list of 2-tuple, each specifying a source and a destination
            blocking (bool, optional): whether or not the python program
                            should be stopped until the gtk window is closed, or
                            should continue to run in the background
        """
        
        for from_node, to_node in args:
            if (not isinstance(from_node, int) or from_node < 0 or from_node > self._sim_params.nb_nodes or 
                not isinstance(to_node, int) or to_node < 0 or to_node >= self._sim_params.nb_nodes) or from_node == to_node:
                continue


            # Create a directed graph with no edges, using the physical graph
            routes_graph = nx.MultiGraph()
            for n in self.nodes:
                routes_graph.add_edges_from(((n.id, neighbor) for neighbor in n.neighbors if neighbor > n.id))
            edge_labels = c.defaultdict(lambda: "")
            
            # Recursive function crawling through the nodes' tables
            def trace_route(current_node, next_node, next_cid, previous_node=None):
                # Update the edge list and edge labels
                if previous_node is None:
                    edge_labels[current_node, next_node] = str(current_node) + " "
                else:
                    edge_labels[current_node, next_node] = edge_labels[previous_node, current_node] + str(current_node) + " "
                routes_graph.add_edge(current_node, next_node, label=edge_labels[current_node, next_node], penwidth=2)

                if next_node == to_node:
                    return

                # Retrieve, using the RT+PRTof the "next_node", the next hop.
                # VERY WRONG : this table lookup triggers a sql query. SQL
                # queries inside loops are wrooooong, and can often be avoided.
                # But this is much simpler, and this function's efficiency is
                # not critical, since called only for debug
                entries = self.nodes[next_node].rt.joint_lookup(PRT, fields=[RT.NEXT_NODE,RT.NEXT_CID], join_on=(RT.ROWID, PRT.RT_ROWID), 
                                                   constraints={PRT.PREV_NODE: current_node, PRT.PREV_CID: next_cid})
                assert len(entries) == 1, "{} entries returned: {}\nWhile tracing from node {} to node {}, with current_node = {}, next_hop = {}. See tables".format(len(entries), [[v for v in r] for r in entries], from_node, to_node, current_node, (next_node, next_cid), self.nodes[next_node].display_node_table())
                next_hop = entries[0]
                

                # There may be several possible next hops that lead to the destination
                # For each possible next hop, call trace_route again (unless the next _node is the destination)
                trace_route(next_node, next_hop[RT.NEXT_NODE], next_hop[RT.NEXT_CID], previous_node=current_node)


            # Add edges according to the RT of nodes in the network
            starts_of_routes = self.nodes[from_node].rt.lookup(fields=[RT.NEXT_NODE, RT.NEXT_CID], constraints={RT.ACTUAL_RCVR: to_node})
#             self._db.execute("SELECT rowid, * FROM drt WHERE _node=? AND actualdest=? AND istemporary=0 AND actualnexthop<>-1", (from_node, to_node,))
#             drt_entries = self._db.fetchall()
            logging.info("Drawing {} routes from {} to {}".format(len(starts_of_routes), from_node, to_node))

            for route in starts_of_routes:
                trace_route(from_node, route[RT.NEXT_NODE], route[RT.NEXT_CID])

            nx.drawing.nx_pydot.write_dot(routes_graph, '/tmp/routes_graph_{}-{}.dot'.format(from_node, to_node))
            graphviz.render('dot', 'png', '/tmp/routes_graph_{}-{}.dot'.format(from_node, to_node))
            time.sleep(1)
            graphviz.view('/tmp/routes_graph_{}-{}.dot.png'.format(from_node, to_node))
            os.remove('/tmp/routes_graph_{}-{}.dot'.format(from_node, to_node))


class NetPhase(IntEnum):
    """Constants coding the different phases of the network"""
    INIT = 0
    TOPO_DISS = 1
    TOPO_DISS_FINISHED = 2
    OCOM = 3
    OCOM_FINISHED = 4


class NetworkManager(object):
        

    def __init__(self, sim_params, net_params):
        """Class modeling a network manager.
    
        The network manager is responsible for:
        
            * Creating and managing a :obj:`.Network` instance
            * Checking the termination of the topology dissemination and oriented communication phases
            * Managing the (uniqueness of) the cid, rcid, and ocomid
            * Gathering the statistics on the network
        
        Note that after a simulation of the network,
        :func:`~apart.simulation.run_simulation` returns an instance of this
        class. That is, it returns the network manager, containing the network
        in its final state, and the attribute :attr:`.net_params`. The latter is
        particularly useful to extract information on the network and the
        protocol (such as the total number of routes proposed, or the latency of
        communications).
            
        Args:
            sim_params (:obj:`~simulation.simulation.SimulationParams`): set of 
                        parameters for the simulation
            net_params (:obj:`~apart.core.network.NetworkParams`): set of 
                        parameters for the network (nb nodes, etc.)
        """
        
        
        self._env = simpy.Environment()
        """The SimPy environment"""
        
        self._sim_params = sim_params
        """Simulation parameters"""
        
        self._net_params = net_params
        """Simulation parameters"""
        
        self._net_stats = NetworkStats(self, self._net_params.nb_nodes, self._sim_params)
        """The :obj:`~measure.network_statistics.NetworkStats` instance where statistics on the network run (e.g. number of rt props, etc.)"""
        
        self._net = Network(self, self._env, sim_params, net_params)
        """The network this class instance manages"""
        
        self._network_is_running = False
        """The state of the network: running or stopped"""
        
        self._current_network_phase = NetPhase.INIT
        """Current phase of the network. Mainly: topo dissemination, or oriented communication"""
        
        # Normally, no two cid value between two nodes  should be equal. but for
        # simplicity, we cheat by using a global network counter. Same thing for
        # rcids and ocomids
        self._cids = defaultdict(lambda: 0)
        """Dict indexed by pairs of neighboring node, giving the next cid those should use"""
        self._rcids = defaultdict(lambda: 0)
        """Dict indexed by pairs of neighboring node, giving the next rcid those should use"""
        self._ocomids = 0
        """Integer giving the next ocomid nodes should use (common for all nodes)"""

    @property
    def network(self):
        """:obj:`~apart.core.network.Network`: The network instance"""
        return self._net
    
    @property
    def network_phase(self):
        """:obj:`.NetPhase`: The current phase the network is in"""
        return self._current_network_phase
        
    @property
    def is_running(self):
        """bool: True is the network is running (in the sense of SimPy)"""
        return self._network_is_running
    
    @property
    def net_stats(self):
        """:obj:`~measures.network_statistics.NetworkStats`: The network statistics instance"""
        return self._net_stats
    
    @property
    def sim_params(self):
        """:obj:`~simulation.SimulationParams`: The simulation parameters"""
        return self._sim_params
    
    def start(self):
        """Starts the network. Calls :meth:`.start_topology_dissemination`"""
        self.start_topology_dissemination()
    
    def start_topology_dissemination(self):
        """Starts the network.
        
        Basically registers the nodes as SimPy processes, and starts the SimPy
        environment
        
        Raises:
            SimulationException: if the network is not in its initiation phase, e.g. if the topology dissemination already took place
        """
        if self._current_network_phase is not NetPhase.INIT:
            raise SimulationException("Impossible to start the topology dissemination phase.")
        
        self.net_stats.network_start_time = time.time()
        
        self._current_network_phase = NetPhase.TOPO_DISS
        if not self.is_running:
            self._start_nodes()
    
        
    def start_oriented_communication_phase(self, ocom_sessions_per_node=None):
        """Start the oriented communication phase in the network.
        
        Using the `ocom_session_per_node` argument if one is provided, or the
        :attr:`~apart.simulation.simulation.SimulationParams.oriented_communication_sessions`
        attribute of the simulation parameters, provides each node with a set of
        sessions to run as end-sender.
        
        Raises:
            SimulationException: if the oriented communication parameter is not well formed
        """
        if self._current_network_phase is not NetPhase.TOPO_DISS_FINISHED:
            raise SimulationException("Impossible to start the oriented communication phase.")
        
        # By default, use the simulation params regarding ocom sessions. 
        # But allow manual override by argument
        if ocom_sessions_per_node is None:
            ocom_sessions_per_node = self._sim_params.oriented_communication_sessions
        
        
        logging.info("Starting oriented communication phase.")
        self._current_network_phase = NetPhase.OCOM
        
        if isinstance(ocom_sessions_per_node, dict):
            for n in self._net.nodes: 
                try:
                    ocom_sessions = ocom_sessions_per_node[n.id]
                except KeyError:
                    ocom_sessions = {}
                n.start_oriented_communications(ocom_sessions=ocom_sessions)
        elif ocom_sessions_per_node is None:
            for n in self._net.nodes: 
                n.start_oriented_communications()
        else:
            raise SimulationException('Network manager: oriented communication sessions parameter is not well formed. Dict excepted, indexed by sender nodes.')
        
        if not self.is_running:
            self._start_nodes()
    
            
    def _start_nodes(self):
        """Starts the nodes' SimPy processes"""
        self._network_is_running = True
        for n in self._net.nodes:
            self._env.process(n.run())
        
        self._env.run()
        
    def _stop_nodes(self):
        """Stops the process of each node in the network.
        
        Note that the function's name may be misleading, because there is no
        properly speaking "network process", only a collection of node processes.
        """
        self.net_stats.network_end_time = time.time()
        
        self._network_is_running = False
    
    def check_network_idleness(self):
        """Check of all nodes in the network are idle, and takes the adequate course of  actions.
        
        The network becomes idle a first time at the end of topology
        dissemination. At this point, if the simulation asks to perform the
        oriented communication phase, the network manager moves to this phase,
        and notifies the nodes. Otherwise, it stops all nodes.
        """ 
        if all((n.is_idle for n in self._net.nodes)):
            logging.info("Network has come to a stop at batching round: {}.".format(self._net.batching_round))
            
            if self._current_network_phase is NetPhase.TOPO_DISS:
                self._current_network_phase = NetPhase.TOPO_DISS_FINISHED
                self._net_stats.log_end_topo_diss(self._net.batching_round)
                logging.info("End of topology dissemination at batching round: {} (i.e. {} hours).".format(self._net.batching_round, round(self._net.batching_round*self._net.params.batching_t_interval/(1000*60*60), 1)))
                
                # debug : check that all node know all others
                ok = True
                for n in self.network.nodes:
                    nodes_set = set(range(self.network.nb_nodes))
                    rows = n.rt.lookup(fields=[RT.ACTUAL_RCVR])
                    for r in rows:
                        nodes_set -= set([r[RT.ACTUAL_RCVR]])
                    if nodes_set:
                        ok = False
                        logging.error("After topo. diss, node {} is missing routes towards nodes {}".format(n, nodes_set))
                assert ok
                
                if self._sim_params.print_nodes_tables:
                    for n in self._net.nodes: n.display_node_table(table="rt")
                
                if self._sim_params.automatic_oriented_comm_phase:
                    self.start_oriented_communication_phase()
                else:
                    self._stop_nodes()
            else:
                logging.info("End of oriented communications at batching round: {} (i.e. {} hours).".format(self._net.batching_round, round(self._net.batching_round*self._net.params.batching_t_interval/(1000*60*60), 1)))
                self.net_stats.log_end_ocom_phase(self._net.batching_round)
                self._stop_nodes()
    
                
            
    def get_next_cid(self, n1, n2):
        """Get the next circuit identifier between nodes `n1` and `n2`
        
        Args:
            n1 (int): index of the first node
            n2 (int): index of the second node
        
        Returns:
            int: a new cid for use between node n1 and node n2, a cid never used by this pair"""
        self._cids[(n1,n2)] += 1
        return self._cids[(n1,n2)]
    
    def get_next_rcid(self, n1, n2):
        """Get the next reverse circuit identifier between nodes `n1` and `n2`
        
        Args:
            n1 (int): index of the first node
            n2 (int): index of the second node
        
        Returns:
            int: a new rcid for use between node n1 and node n2, a rcid never used by this pair
        """
        self._rcids[(n1,n2)] += 1
        return self._rcids[(n1,n2)]
    
    def get_next_ocomid(self):
        """Get the next oriented communication identifier 
        
        Returns:
            int: a network-wide unique oriented communication identifier
        """
        self._ocomids += 1
        return self._ocomids
    


class NetworkParams(object):
    def __init__(self, **kwargs):
        """Class grouping all the parameters of the network
        
        Accepts keyword arguments, corresponding to the network parameters, to
        replace the default values
        """
        
        self.nb_nodes = 7
        """int: Number of nodes in the network.

        Default: 7.
        """
        
        self.corruption_ratio = 0.3
        """float between 0 and 1: The percentage of corrupted nodes in the network.

        Default: 0.3.
        """
        
        # How long should a node "wait" before sending a message. This parameters
        # basically models the time it takes for a node to forge and send a message
        # Assuming a message takes 1024 bits = 1 Mb, and assuming a the nodes send data at a bit rate of 50 Mbps, a message takes 20 ms to send
        self.communication_latency = 20  # in milliseconds.
        """int: Latency of message delivery in the network (simulated with SimPy).

        Default: 20.
        """
        
        # The security parameter for crypto
        self.secparam = 128
        """int: The cryptographic security parameter of the network. Typically, 80, 128, or 256.

        Default: 128.
        """
        
        self.topology_graph = None
        """:obj:`networkx.Graph`: The topology graph for the simulation. If left to None, one is randomly generated.

        Default: None.
        """
        
        # Maximum length of a route (nodes refuse the route proposition for routes
        # longer than this)
        self.rtprop_policy_max_hop_count = None
        """int: Route proposal policy, maximum length of a route. 
        
        Default: None. 
        
        If set to 0 or None, this parameter is automatically set to the 
        smallest integer so that any node is ensured to obtain a route 
        towards any other (i.e. the longest shortest path in the topo graph)
        """
    
        # Maximum number of routes a node can have towards a given destination. For
        # example, if this value is 5, and some node already have 5 routes towards
        # destination D, upon receiving a route prop towards D, it will refuse it
        self.rtprop_policy_max_routes = 3
        """int: Route proposal policy, maximum number of routes per receiver.

        Default: 3.
        """
    
        # When a node gets a route proposal for a destination it already knows, it
        # has a probability to refuse it straight away (independently from the other
        # decision elements such as routing loops, route length, etc). The proba.
        # below gives is the proba. of NOT refusing straight away
        self.rtprop_policy_p_reaccept = 0.5
        """float between 0 and 1: Route proposal policy, probability of accepting a second route towards an already known receiver.

        Default: 0.5.
        """
        
        # When a node already has enough routes towards a receiver, but receives one
        # more, it has a chance of replacing one of its already known route with the
        # new one
        self.rtprop_policy_p_replace = 0.25
        """float between 0 and 1: Route proposal policy, probability to "replace" a known route with a newly learned one.

        Default: 0.25.
        """
            
        # When a node re-discovers a destination (i.e. obtains a second or more
        # route towards a given destination already known beforehand), this node has
        # a probability to propose this route to its neighbors. This proba is
        # computed from the value below, put to the power of the number of routes
        # already proposed by the node (towards the concerned destination).
#         self.rtprop_policy_p_reprop = 0.1
        #"""Route proposal policy: probability of re-proposing a route"""
    
        # Parameters of the pool-based message re-ordering mechanism
        self.batching_t_interval = 1*60*1000 # 1 minute in milliseconds
        """int: message re-ordering, batching interval, in milliseconds (i.e. in SimPy simulation time).

        Default: 1 minute.
        """
        self.batching_nmin = 5
        """int: Message re-ordering, minimum of message that must always be in pools.

        Default: 5.
        """
        self.batching_f = 0.5
        """float between 0 and 1: Message re-ordering, maximum fraction of pools that can be sent in one round.

        Default: 0.5.
        """
        
        # Parameters of the dummy message and traffic rates policy
        self.dummypol_fdum = 0.8
        """float between 0 and 1: Dummy messages and Controlled Traffic rates, fraction of neighbors pools in which to insert a dummy at each round.

        Default: 0.8.
        """
        self.dummypol_deltar = 8
        """int: Dummy messages and Controlled Traffic rates, number of rounds on which the controlled traffic rate equation is relaxed.

        Default: 8.
        """
        
        self.update_params(**kwargs)
        
    def update_params(self, **kwargs):
        """Updates the parameters after instanciation of the object.
        
        Accepts keyword arguments, corresponding to the network parameters. Only
        keyword arguments that match a valid attribute of the class are taken
        into accounts. Others are simply ignored silently.
        """
        # Above are the default argument values. If kwargs is not empty,
        # override these values
        for attr_name, attr_value in kwargs.items():
            if attr_name in self.__dict__:
                setattr(self, attr_name, attr_value)
                
    def __str__(self):
        p_list = []
        for k, v in self.__dict__.items():
            if k == 'upgrade_params': continue
            p_list.append(k+"="+str(v))
        return '{}({})'.format(self.__class__.__name__, ", ".join(sorted(p_list)))


class SimulationException(Exception):
    """Raised when an error occurs due to the handling SimPy simulation, or during the lifetime of the network"""
    pass