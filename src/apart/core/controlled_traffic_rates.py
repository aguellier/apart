'''
This module contains material to manage the traffic counters of the controlled traffic rates mechanism of the protocol.
'''

from _collections import defaultdict
from collections import deque
import copy

from common.utilities import range1


class TrafficController(object):
    def __init__(self, delta_r, neighbors):
        """
        This class stores and manages the traffic counters of a node. More
        exactly, it manages the counters n_i_real, n_i_real over the last
        ``delta_r`` rounds, and manages one counter n_i_dum, n_o_dum per
        neighbor, for the last ``delta_r`` rounds.
        
        Args:
            delta_r (int): the number of rounds over which the traffic rates contraints are relaxed.
            neighbors (list of int): the indexes (in the network) of the neighbors of the node to
                 which a particular :obj:`.TrafficController` instance belongs to.
        """
        
        self.__delta_r = delta_r
        self._neighbors = copy.deepcopy(neighbors)
        
        
        #=======================================================================
        # Counters n_i_real and n_o_real are actually composed of ``delta_r+1``
        # elements, representing the last few batching rounds. Counters n_i_dum
        # and n_i_dum are actually dict containing each ``detla_r+1`` elements:
        # one dict entry per neighbor.
        #
        # The *vectors* are implemented with a ``collections.deque``, that are
        # rotated as rounds pass by. For each  counter c, c[delta_r] represents
        # the oldest round, and c[0] is the value for the current round
        #=======================================================================
        #=======================================================================
        deque_size = self.__delta_r+1
        self.__n_i_dum_budget = deque([defaultdict(lambda : 0) for _ in range(deque_size)], maxlen=deque_size)
        self.__n_o_dum_budget = deque([defaultdict(lambda : 0) for _ in range(deque_size)], maxlen=deque_size)
        
        self.__n_i_real_unresolved = deque([0]*deque_size, maxlen=deque_size)
        self.__n_o_real_unresolved = deque([0]*deque_size, maxlen=deque_size)
                
    def __index_from_relative_round(self, relative_round):
        # Si relative_round = -delta_r, return delta_r
        # Si relative_round = 0, return 0
        return -relative_round


    def next_round(self):
        """Move the counters to the next round
        
        Depending on the implementation, this function may involve a rotation of
        queues for instance
        """
        
        self.__n_i_dum_budget.rotate(1)
        self.__n_i_dum_budget[0] = defaultdict(lambda : 0)
        
        self.__n_o_dum_budget.rotate(1)
        self.__n_o_dum_budget[0] = defaultdict(lambda : 0)
            
        self.__n_i_real_unresolved.rotate(1)
        self.__n_i_real_unresolved[0] = 0
        
        self.__n_o_real_unresolved.rotate(1)
        self.__n_o_real_unresolved[0] = 0
        
        
    def equilibrate_n_o_real_and_n_i_real(self):
        """Called at the end of a round, updates n_o_real and n_i_real.
        
        This function effectively cancels out values of n_i_real with values of
        n_o_real, or vice-versa, depending on which one is the greatest of the two2
        """
        # Start from the (sum of) values of any of the two counter n_i_real or
        # n_o_real indifferently. Here, we choose n_i_real
        amount_resolvable = self.total_unresolved_n_i_real
        
        # Resolve the OTHER counter (n_o_real) by the said amount. If it turns
        #out that sum(n_o_real) was LOWER than sum(n_i_real), the below function
        #returns the amount NOT consumed from the n_i_real counter
        amount_left = self._consume_real_counter(self.__n_o_real_unresolved, amount_resolvable)
        
        # Accordingly resolve n_i_real by the SAME amount as n_o_real, i.e.
        # ``amount_resolvable-amount_left``
        self._consume_real_counter(self.__n_i_real_unresolved, amount_resolvable-amount_left)
        
    def resolve_n_o_real_with_dummies(self):
        """Called at the end of a round, to decrease n_o_real according to the dummy budget
        
        This function effectively consumes incoming dummy budget to resolve
        potentially unresolved n_o_real
        """ 
        amount_resolvable = self.total_n_i_dum_budget
        amount_left = self._consume_real_counter(self.__n_o_real_unresolved, amount_resolvable)
        self._consume_dummy_counter(self.__n_i_dum_budget, amount_resolvable - amount_left)
        
    def resolve_n_i_real_with_dummies(self):
        """Called at the end of a round, to decrease n_i_real according to the dummy budget
        
        This function effectively consumes outgoing dummy budget to resolve
        potentially unresolved n_i_real
        """
        amount_resolvable = self.total_n_o_dum_budget
        amount_left = self._consume_real_counter(self.__n_i_real_unresolved, amount_resolvable)
        self._consume_dummy_counter(self.__n_o_dum_budget, amount_resolvable-amount_left)
        
        
         
        
    def _consume_real_counter(self, counter, amount):
        """Decrease a real message counter by the specified amount, starting fro mthe oldest round"""
        # Start from the counter (i.e. from the oldest round), and decrement each value when possible
        i = self.__delta_r
        while amount > 0 and i >= 0:
            if counter[i] <= amount:
                amount, counter[i] = amount-counter[i], 0
            else:
                amount, counter[i] = 0, counter[i]-amount
            i -= 1
            
        return amount         
        
    def _consume_dummy_counter(self, counter, amount):
        """Decrease a dummy message counter by the specified amount, starting from the value in the oldest round""" 
        amount_per_neighbor = dict((n, amount) for n in self._neighbors)
        i = self.__delta_r
        while any(amount_per_neighbor.values()) and i >= 0:
            for n in amount_per_neighbor:
                if counter[i][n] <= amount_per_neighbor[n]:
                    amount_per_neighbor[n], counter[i][n] = amount_per_neighbor[n]-counter[i][n], 0
                else:
                    amount_per_neighbor[n], counter[i][n] = 0, counter[i][n]-amount_per_neighbor[n]
            i -= 1
        
    
    #===========================================================================
    # Information getters 
    #===========================================================================
    @property
    def current_equilibrium(self):
        """int: The equilibrium of the counters, i.e. the sum over the last delta_r rounds of n_o_real-n_i_real"""
        return sum([n_o-n_i for n_o,n_i in zip(self.__n_o_real_unresolved, self.__n_i_real_unresolved)]) 
    
    @property
    def last_unresolved_n_i_real(self):
        """int: the value of n_i_real for the oldest rounds, that **must** to be resolved in the current round"""
        return self.__n_i_real_unresolved[self.__delta_r]
    
    @property
    def total_unresolved_n_i_real(self):
        """int: the total excess (compared to sent ones) of received real messages"""
        return sum(self.__n_i_real_unresolved)
    
    @property
    def last_unresolved_n_o_real(self):
        """int: the value of n_o_real for the oldest rounds, that **must** to be resolved in the current round"""
        return self.__n_o_real_unresolved[self.__delta_r]
    
    @property
    def total_unresolved_n_o_real(self):
        """int: the total excess (compared to received ones) of sent real messages"""
        return sum(self.__n_o_real_unresolved)
    
    @property
    def total_n_o_dum_budget(self):
        """int: total outgoing dummy budget"""
        return min((sum(self.__n_o_dum_budget[r][n] for r in range1(0, self.__delta_r)) for n in self._neighbors))
    
    @property
    def total_n_i_dum_budget(self):
        """int: total incoming dummy budget"""
        return min((sum(self.__n_i_dum_budget[r][n] for r in range1(0, self.__delta_r)) for n in self._neighbors))
        
        
    
    
    
    #===========================================================================
    # Setters of the counters, under the form of incrementation of the current
    # round counter
    #===========================================================================
    def inc_current_n_i_real(self, inc=1):
        """Increments the n_i_real counter of the current round by the specified value"""
        self.__n_i_real_unresolved[0] += inc
        
    def inc_current_n_o_real(self, inc=1):
        """Increments the n_o_real counter of the current round by the specified value"""
        self.__n_o_real_unresolved[0] += inc

    def inc_current_n_o_dum(self, neighbor, inc=1):
        """Increments the n_o_dum counter of the current round by the specified value"""
        self.__n_o_dum_budget[0][neighbor] += inc
        
    def inc_current_n_i_dum(self, neighbor, inc=1):
        """Increments the n_i_dum counter of the current round by the specified value"""
        self.__n_i_dum_budget[0][neighbor] += inc
        
    def __str__(self):
        s = "n_i_dum_budget = {} = {}\n".format([[ddict[n] for n in self._neighbors] for ddict in self.__n_i_dum_budget], [sum(self.__n_i_dum_budget[r][n] for r in range1(0, self.__delta_r)) for n in self._neighbors])
        s += "n_o_dum_budget = {} = {}\n".format([[ddict[n] for n in self._neighbors] for ddict in self.__n_o_dum_budget], [sum(self.__n_o_dum_budget[r][n] for r in range1(0, self.__delta_r)) for n in self._neighbors])
        s += "last_unresolved_n_i_real = {}\n".format(list(self.__n_i_real_unresolved))
        s += "last_unresolved_n_o_real = {}".format(list(self.__n_o_real_unresolved))
        return s
    
# class TrafficRateError(Exception):
#     pass    
        
        
        
# Some basic tests
if __name__ == '__main__':    
    delta_r = 2
    neighbors = [12,5,42]
    tc = TrafficController(delta_r, neighbors)
    
    # First round
    for n in neighbors: tc.inc_current_n_i_dum(n, inc=1)
    for n in neighbors: tc.inc_current_n_o_dum(n, inc=1)
    tc.inc_current_n_i_real(inc=3)
    tc.inc_current_n_o_real(inc=0)
    
    #Second round
    tc.next_round()
    for n in neighbors: tc.inc_current_n_i_dum(n, inc=1+n%2)
    for n in neighbors: tc.inc_current_n_o_dum(n, inc=(1-n%2))
    tc.inc_current_n_i_real(inc=2)
    tc.inc_current_n_o_real(inc=0)
    
    # Third round
    tc.next_round()
    for n in neighbors: tc.inc_current_n_i_dum(n, inc=1+n%3)
    for n in neighbors: tc.inc_current_n_o_dum(n, inc=1+(2-n%3))
    tc.inc_current_n_i_real(inc=10)
    tc.inc_current_n_o_real(inc=0)
    
    
    print("CONSTRAINTS")
    current_equilibrium = tc.current_equilibrium
    o_dum_budget = -tc.total_n_o_dum_budget
    i_dum_budget = tc.total_n_i_dum_budget
    
    soft_lbound_real_msgs = o_dum_budget - current_equilibrium
    strict_lbound_real_msgs = o_dum_budget + tc.last_unresolved_n_i_real
    
    strict_ubound_real_msgs = i_dum_budget - current_equilibrium
    
    print(tc)
    print("{} (strict) <= {} (soft) <= nb_reals_to_send_in_this_round <= {}".format(strict_lbound_real_msgs, soft_lbound_real_msgs, strict_ubound_real_msgs))
    
    
    
    print("\n\nBeginning resolving")
    print(tc)
    print("Equilibrium : {}".format(tc.current_equilibrium))
    
    print("\nAfter equilibration")
    tc.equilibrate_n_o_real_and_n_i_real()
    print(tc)
    print("Equilibrium : {}".format(tc.current_equilibrium))
    
    print("\nAfter 'resolve n_i_real_with_dummies'")
    tc.resolve_n_i_real_with_dummies()
    print(tc)
    print("Equilibrium : {}".format(tc.current_equilibrium))
    
    print("\nAfter 'resolve n_o_real_with_dummies'")
    tc.resolve_n_o_real_with_dummies()
    print(tc)
    print("Equilibrium : {}".format(tc.current_equilibrium))
    