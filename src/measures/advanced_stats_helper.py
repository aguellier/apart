# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html



'''
This module provides a common basis for computing *advanced* statistics on the network.

From a network state after a run, this class provides easy means of computing
common metrics that are likely to be uses by many measure module. E.g. get the
average route length from the node's routing table.
'''
from collections import defaultdict, OrderedDict
import functools
import re
from statistics import mean

from apart.core.network import NetworkStats
from apart.core.protocol_constants import RtPolicyReason, F_NULL
from apart.core.tables import RoutingTable as RT, PrevRoutingTable as PRT
from measures.network_statistics import RegularStat, RunningStat
import networkx as nx


class AdvancedNetworkStatsHelper(object):
    """Helper class to compute advanced stats, given the :obj:`~apart.core.network.NetworkManager` instance of a network run.
     
    The class implements a form of singleton, but one singleton is created per
    network manager instance provided
    """
    _instances = {}
    
    def __new__(self, net_manager):
        if net_manager in AdvancedNetworkStatsHelper._instances:
            return AdvancedNetworkStatsHelper._instances[net_manager]
        else:
            new_inst = AdvancedNetworkStatsHelper.AdvancedNetworkStatsHelperSingleton(net_manager)
            AdvancedNetworkStatsHelper._instances[net_manager] = new_inst
            return new_inst
            
    
    
    class AdvancedNetworkStatsHelperSingleton(object):

        def __init__(self, net_manager):
            """The singleton class, containing the actual code.
            
            Given a network manager instance, for a network which run is over,
            initilises the class.
            
            Args:
                net_manager (:obj:`~apart.core.network.NetworkManager`): the network manager instance after the network run.
            """
            
            self._net_manager = net_manager
            self._net = net_manager.network
            self._sqlite_cursor = self._net.routing_tables_sqlite_db_cursor

        #----------------------------
        # Helper functions
        #----------------------------
        @property
        def nb_routes_btw_pairs(self):
            """:obj:`~measures.network_statistics.RegularStat`: The number of routes between each pairs of nodes at the end of the network run.
            
            This property gives the average, minimum, and maximum number of
            routes between each pair of node (along with a standard deviation).
            """
            try:
                return self._nb_routes_by_pairs
            except AttributeError:
                self._sqlite_cursor.execute("SELECT COUNT(*) FROM "+RT.table_name+" WHERE "+RT.ACTUAL_RCVR+" <> "+RT.NODE+" AND "+RT.IN_USE+"=1 GROUP BY "+RT.NODE+", "+RT.ACTUAL_RCVR)
                self._nb_routes_by_pairs = [x[0] for x in self._sqlite_cursor.fetchall()]
                
                # Complete the list: some pairs of node may not have any route
                # between them, but by the way the sql query is made, they will not
                # be represented. Thus, add a zero value to fill thi gap
                nb_pairs = self._net.nb_nodes * (self._net.nb_nodes - 1)
                for _ in range(nb_pairs - len(self._nb_routes_by_pairs)):
                    self._nb_routes_by_pairs.append(0)
                    
                self._nb_routes_by_pairs = RegularStat(self._nb_routes_by_pairs)
    
                return self._nb_routes_by_pairs
        
        @property
        def routes_length(self):
            """list: The length of all routes at the end of the network run.
            
            This property is a list of dict, each dict containing keys
            `'sender'`, `'receiver'`, and `'length'`. Each element of the list
            represents one route.
            """
            try:
                return self._routes_length
            except AttributeError:
                self._sqlite_cursor.execute("SELECT "+RT.NODE+", "+RT.ACTUAL_RCVR+", "+RT.ACTUAL_LENGTH+" FROM "+RT.table_name+" WHERE "+RT.ACTUAL_RCVR+" <> "+RT.NODE+" AND "+RT.IN_USE+"=1")
                rows = self._sqlite_cursor.fetchall()
                self._routes_length = [{'sender': r[RT.NODE], 'receiver': r[RT.ACTUAL_RCVR], 'length': r[RT.ACTUAL_LENGTH]} for r in rows]
                return self._routes_length
    
        @property
        def nb_neighbors(self):
            """:obj:`~measures.network_statistics.RegularStat`: The number of neighbors a node has.
            
            This property gives the average, minimum, and maximum number of
            neighbors that any node in the network has (along with a standard deviation).
            """
            try:
                return self._nb_neighbors
            except AttributeError:
                self._nb_neighbors = RegularStat([len(n.neighbors) for n in self._net.nodes])
                return self._nb_neighbors
    
        @property
        def nb_rt_props(self):
            """dict: The total number of routes proposal at the end of the network run.
            
            This property gives a dict which values are
            :obj:`~measures.network_statistics.RegularStat`
            instances. It is indexed by `'received'`, `'accepted'`, and
            `'refused'`, respectively giving the number of received route
            proposal, the number of accepted ones, and the number of refused
            ones.
            """
            try:
                return self._nb_rt_props
            except AttributeError:
                self._nb_rt_props = dict(map(lambda t: (t[0], RegularStat(t[1].values())), 
                                             self._net_manager.net_stats.nb_route_props_per_node.items()))
                self._nb_rt_props['received'] = RegularStat([sum(self._net_manager.net_stats.nb_route_props_per_node[reason][n] for reason in RtPolicyReason) 
                                                             for n in range(self._net.nb_nodes)])
                self._nb_rt_props['accepted'] = RegularStat([sum(self._net_manager.net_stats.nb_route_props_per_node[reason][n] for reason in RtPolicyReason if reason.name.upper().startswith('ACCEPT')) 
                                                             for n in range(self._net.nb_nodes)])
                self._nb_rt_props['refused'] = RegularStat([sum(self._net_manager.net_stats.nb_route_props_per_node[reason][n] for reason in RtPolicyReason if reason.name.upper().startswith('REFUSED')) 
                                                             for n in range(self._net.nb_nodes)])
                return self._nb_rt_props

        def histogram_routes_length(self):
            """dict: Histogram of the routes length at the end of the network run.
            
            The dict is indexed by values of routes length, and its values are
            the number of routes of such length.
            """
            res = defaultdict(lambda: 0)
            for r in self.routes_length:
                res[r['length']] += 1
            return res

        # The the ratio between the length of the routes in the network and the
        # length of the shortest corresponding path
        def routes_length_vs_shortest_path(self):
            """:obj:`~measures.network_statistics.RunningStat`: Ratio of the routes length over their corresponding shortest paths
            
            This property computes the ratio `length(rt)/length(sp)` for each
            route `rt` existing in the networt, where `sp` is the shortest path
            corresponding to `rt`, *i.e.* the shortest path in the network graph
            that goes from the end-sender of the route to its end receiver. What
            is returned is
            :obj:`~measures.network_statistics.RunningStat`
            built from these ratios.
            """
            shortest_paths_lengths = nx.shortest_path_length(self._net.topology_graph)
             
            res = RunningStat()
            for r in self.routes_length:
                res.push_value(r['length'] / shortest_paths_lengths[r['sender']][r['receiver']])
 
                 
            return res

        @property
        def complete_routes_descriptions(self):
            """dict: Returns the description of all routes between all pairs of nodes.
            
            Returns a dict, indexed by (end-sender, end-receiver), which values
            are lists of lists. That is, d[s,r] = l, and each element of l is a
            list l' such that (l'[0] = index of end-sender, indexes of relay
            nodes (and indirection nodes), l'[-1] = index of end-receiver)
            """
 
            routes = {}
            
            for from_node in range(self._net.nb_nodes):
                for to_node in range(self._net.nb_nodes):
                    if from_node == to_node:
                        continue
                    
                    def trace_route(current_node, next_node, next_cid, previous_node=None, route_accumulator=[]):
                        route_accumulator.append((current_node, next_cid))
                        
                        if next_node == F_NULL:
                            return route_accumulator
                        
                        # Retrieve, using the RT+PRTof the "next_node", the next hop.
                        # VERY WRONG : this table lookup triggers a sql query. SQL
                        # queries inside loops are wrooooong, and can often be avoided.
                        # But this is much simpler, and this function's efficiency is
                        # not critical, since called only for debug
                        entries = self._net.nodes[next_node].rt.joint_lookup(PRT, fields=[RT.NEXT_NODE,RT.NEXT_CID], join_on=(RT.ROWID, PRT.RT_ROWID), 
                                                           constraints={PRT.PREV_NODE: current_node, PRT.PREV_CID: next_cid})
                        assert len(entries) == 1, "{} entries returned: {}\nWhile tracing from node {} to node {}, with current_node = {}, next_hop = {}. See tables".format(len(entries), [[v for v in r] for r in entries], from_node, to_node, current_node, (next_node, next_cid), self.nodes[next_node].display_node_table())
                        next_hop = entries[0]
                        
                        # There may be several possible next hops that lead to the destination
                        # For each possible next hop, call trace_route again (unless the next _node is the destination)
                        return trace_route(next_node, next_hop[RT.NEXT_NODE], next_hop[RT.NEXT_CID], previous_node=current_node, route_accumulator=route_accumulator)
    
                    
                    routes[(from_node, to_node)] = []
                    starts_of_routes = self._net.nodes[from_node].rt.lookup(fields=[RT.NEXT_NODE, RT.NEXT_CID], constraints={RT.ACTUAL_RCVR: to_node, RT.IN_USE: True})
                    for route in starts_of_routes:
                        one_route = trace_route(from_node, route[RT.NEXT_NODE], route[RT.NEXT_CID], route_accumulator=[])
                        routes[(from_node, to_node)].append(one_route)
            return OrderedDict(sorted(routes.items(), key=lambda x: x[0]))
    