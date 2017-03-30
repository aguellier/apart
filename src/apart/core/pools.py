# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html



'''
This module gathers the components for the management of pools. 

It defines a class :class:`.NeighborMsgPool` which models on message pool, and a
class  :class:`.NeighborMsgPoolSet` modeling a set of pools containing one
:obj:`.NeighborMsgPool` instance for each neighbor.  
'''
from collections import defaultdict

import itertools
import math
import numpy
import random

from common.custom_logging import *
from apart.core.messages import LinkMsg, MsgFlag


class NeighborMsgPool(object):
    def __init__(self, n_min, f, node, neighbor):
        """This class represents one message pool, following the design of our variant of *cottrell mix*. It is paramterd by a value nmin and f.
        
        Args:
            n_min (int): the parameter nmin of the pool
            f (float): the paramter f of the pool (between 0 and 1)
            node (:obj:`~apart.core.node.Node`): the node to which this pool belongs to
            neighbor (int): the index (in the network) of the neighbor of ``node`` to which this pool relates. 
        """
        
        self._real_msgs_pool = set()
        self._dummy_msgs_pool = set()
        self.__n_min = n_min
        self.__f = f
        self.__node = node
        self.__neighbor = neighbor
        self.__nb_dummy_msgs = 0

    @property
    def n(self):
        """int: The number of messages (dummy and reals counted together) currently in the pool"""
        return self.n_real + self.n_dummy
    
    @property
    def n_real(self):
        """int: The number of real messages currently in the pool"""
        return len(self._real_msgs_pool)
    
    @property
    def n_dummy(self):
        """int: The number of dummy messages currently in the pool"""
        return len(self._dummy_msgs_pool)
    
    @property
    def n_available(self):
        """int: The number of *available* messages that can be taken out of the pool
        
        That is : min(:attr:`.n` - nmin, f * :attr:`.n`)
        """
        return max(0, min(self.n - self.__n_min, math.floor(self.n*self.__f)))
    
    def add_msg(self, m):
        """Add a message in the pool
        
        Args:
            m (:obj:`~apart.core.messages.LinkMsg`): the message to add
        """
        if m.flag is MsgFlag.DUMMY:
            pass
            self._dummy_msgs_pool.add(m)
        else:
            self._real_msgs_pool.add(m)
            
    def add_msgs(self, *args):
        """Add several messages to the pool
        
        Args:
            *args (list of obj:`~apart.core.messages.LinkMsg`): the messages to add in the pool
        """
        for m in args: self.add_msg(m)
    
    # Shortcut 
    def add_dummy_msg(self):
        """Add a dummy message in the pool. 
        
        Function provided for convenience
        """
        self.add_msg(LinkMsg.create_dummy(sent_by=self.__node.id, sent_to=self.__neighbor))

        
    def remove_batch(self, batch):
        """Remove a set of messages from the pool
        
        Args:
            batch (dict of string->list :obj:`~apart.core.messages.LinkMsg`): a
                dict with two keys ('reals' and 'dummies'), each respectively 
                containing a list of real and dummy messages
        """
        self._real_msgs_pool -= set(batch['reals'])
        self._dummy_msgs_pool -= set(batch['dummies'])
    
    # sampling bias is in [0,1[
    def sample_batch(self, k, sampling_bias=0.0):
        """Randomly sample messages from the pool
        
        Args:
            k (int): the number of messages to sample
            sampling_bias (float, optional): a float between 0 and 1, biasing the sampling towards real messages as it gets closer to 1 (Default: 0.0)
            
        Returns:
            dict of string->list :obj:`~apart.core.messages.LinkMsg`: a
                dict with two keys ('reals' and 'dummies'), each respectively 
                containing a list of real and dummy messages
            
        
        Raises:
            MsgPoolError: if ``k`` is greater than :attr:`.n_available`
        """ 
        if not (0 <= sampling_bias < 1):
            logging.error("The sampling bias must be a floating number between 0 (included) and 1 (excluded). Received {}".format(sampling_bias))
            sampling_bias = round(sampling_bias)
            if sampling_bias >= 1:
                sampling_bias = 0.99999
            else:
                sampling_bias = 0
         
        if k > self.n_available:
            raise MsgPoolError("Not enough messages in pool to sample {} messages ({} available)".format(k, self.n_available))
         
        separate_batches = {'reals': [], 'dummies': []}
        if k == 0:
            return separate_batches
         
        # We have to resort to numpy for the *weighted* sampling *without replacement*
        if self.n_real:
            p_base = (1/self.n)*(1-sampling_bias)
            p_real_msgs = (1/self.n_real)*sampling_bias
            prob_distrib_real_msgs = [p_base+p_real_msgs]*self.n_real
            prob_distrib_dummy_msgs = [p_base]*self.n_dummy
            prob_distrib = prob_distrib_real_msgs+prob_distrib_dummy_msgs
        else:
            prob_distrib = None
        
        population = list(self._real_msgs_pool)+list(self._dummy_msgs_pool)
         
        batch = numpy.random.choice(population, size=k, replace=False, p=prob_distrib)
         
        for m in batch:
            if m.flag is MsgFlag.DUMMY:
                separate_batches['dummies'].append(m)
            else:
                separate_batches['reals'].append(m)
         
        return separate_batches
        
        
    def __iter__(self):
        return itertools.chain(self._real_msgs_pool, self._dummy_msgs_pool)
        
    def __str__(self):
        return str(self._real_msgs_pool | self._dummy_msgs_pool)
    
    def __repr__(self):
        return self.__str__()
        
class NeighborMsgPoolSet(object):
    def __init__(self, n_min, f, node, neighbors):
        """Class abstracting the set of pools (one per neighbor) managed by any given node.
        
        An instance of this class contains as many instances of
        :class:`.NeighborMsgPool` as there node ``node`` has neighbors.
        
        Args:
            n_min (int): the parameter nmin of the pool
            f (float): the paramter f of the pool (between 0 and 1)
            node (:obj:`~apart.core.node.Node`): the node to which this pool belongs to
            neighbors (list int): the indexes (in the network) of all neighbors of ``node``
        """
        self.__pools = dict((n, NeighborMsgPool(n_min, f, node, n)) for n in neighbors)
        
        
    @property
    def n_available(self):
        """int: The overall number of available messages in the pool set.
        
        Defined as the mininum of the value
        :attr:`~.NeighborMsgPool.n_available` over all :obj:`.NeighborMsgPool`
        instances.
        """
        return min((p.n_available for p in self.__pools.values()))
    
    @property
    def n_real(self):
        """int: The total number of real messages in the collection of pools""" 
        return sum((p.n_real for p in self.__pools.values()))
 
    def sample_neighbor_batches(self, sampling_bias=0):
        """Randomly sample a set of batches, one from each neighbor pool
        
        The number of messages sampled is automatically set to the maximum
        possible, i.e. to :attr:`.n_available`.
        
        Args:
            sampling_bias (float, optional): a float between 0 and 1, biasing the sampling towards real messages as it gets closer to 1 (Default: 0.0)
            
        Returns:
            dict of dict of string->list :obj:`~apart.core.messages.LinkMsg`: a
                dict indexed by neighbors, with as values 
                dict with two keys ('reals' and 'dummies'), each respectively 
                containing a list of real and dummy messages
        """
        k = self.n_available
         
        return dict((n, p.sample_batch(k, sampling_bias)) for (n, p) in self.__pools.items())

    def remove_batches(self, batches):
        """Remove messages from the pools
        
        Args:
            batches (dict of dict of string->list :obj:`~apart.core.messages.LinkMsg`): a
                dict indexed by neighbors, with as values 
                dict with two keys ('reals' and 'dummies'), each respectively 
                containing a list of real and dummy messages 
        """
        for (n, b) in batches.items():
            self.__pools[n].remove_batch(b)
            
    def __str__(self, *args, **kwargs):
        s = ""
        for (n, p) in self.__pools.items():
            s += str(n) + ": "+str(p)+"\n"
        return s
        
    def __getitem__(self, key):
        return self.__pools[key]
        
    def __iter__(self):
        return iter(self.__pools.values())
    
    
class MsgPoolError(RuntimeError):
    """Raised when a bad request for sampling messages in pools is issued."""
    pass




# Misc. very basic unit tests

def test_removal():
    neighbor = 1
    pool = NeighborMsgPool(2, 0.8, 0, 1)
    
    msgs = []
    for _ in range(10):
        msgs.append(LinkMsg.create_dummy(sent_by=0, sent_to=1))
    for i in range(10):    
        msgs.append(LinkMsg(sent_by=0, sent_to=1, c1=i, c2=0, flag=MsgFlag.PAYLOAD, cid=i*12))
    
    for m in msgs:
        pool.add_msg(m)
    
    msgs_to_remove = random.sample(msgs, k=5)
    msgs_to_remove.append(LinkMsg.create_dummy(sent_by=0, sent_to=1))
    msgs_to_remove.append(LinkMsg.create_dummy(sent_by=0, sent_to=1))
    msgs_to_remove.append(LinkMsg(sent_by=0, sent_to=1, c1=42, c2=0, flag=MsgFlag.PAYLOAD, cid=42))
    msgs_to_remove.append(LinkMsg(sent_by=0, sent_to=1, c1=0, c2=0, flag=MsgFlag.PAYLOAD, cid=0))
    
    print("Pool before remova = {}".format(pool))
    batch = {'reals': [m for m in msgs_to_remove if m.flag is not MsgFlag.DUMMY], 'dummies': [m for m in msgs_to_remove if m.flag is MsgFlag.DUMMY]}
    pool.remove_batch(batch) 
    print("Pool after remova = {}".format(pool))
    
    for m in msgs:
        if m in msgs_to_remove:
            # Should NOT be present in pool
            found = False
            for m_pool in pool:
                if m_pool is m:
                    found = True
                    break
            if found is True:
                print("Error !")
        else:
            # Should be present in pool
            found = False
            for m_pool in pool:
                if m_pool in pool:
                    found = True
                    break
            if found is False:
                print("Error 2 !")
 
    
def test_bias():
    neighbors = [1,2,3]
    pools = NeighborMsgPoolSet(2, 0.8, 0, neighbors)
    
    filling_mode = 'random'
    filling_nb_real_msgs = 10
    filling_nb_dummy_msgs = 10
    
    if filling_mode == 'random':
        for _ in range(len(neighbors)*filling_nb_dummy_msgs):
            n = random.choice(neighbors)
            pools[n].add_dummy_msg()
        for  i in range(len(neighbors)*filling_nb_real_msgs):
            n = random.choice(neighbors)
            pools[n].add_msg(LinkMsg(sent_by=0, sent_to=n, c1=i, c2=i, flag=MsgFlag.PAYLOAD, cid=42*i))
    else:
        for _ in range(filling_nb_dummy_msgs):
            for n in neighbors:
                pools[n].add_dummy_msg()
        for  i in range(filling_nb_real_msgs):
            for n in neighbors:
                pools[n].add_msg(LinkMsg(sent_by=0, sent_to=n, c1=i, c2=i, flag=MsgFlag.PAYLOAD, cid=42*i))
            
            
    
    
    print("{} msgs available".format(pools.n_available))
    
    for bias in range(0, 100, 10):
        bias = bias/100
        nb_tries = 50
        avg_nb_reals_overall = 0
        avg_nb_dummies_overall = 0
        avg_nb_reals_per_neighbor = defaultdict(lambda: 0)
        avg_nb_dummies_per_neighbor = defaultdict(lambda: 0)
        for _ in range(nb_tries):
            batches = pools.sample_neighbor_batches(sampling_bias=bias)
            avg_nb_reals_overall += sum((len(b['reals']) for b in batches.values()))
            avg_nb_dummies_overall += sum((len(b['dummies']) for b in batches.values()))

            
            
        print("For bias = {}, {} real messages and {} dummy messages selected in average".format(bias, avg_nb_reals_overall/nb_tries, avg_nb_dummies_overall/nb_tries))
        print()