'''
This package contains the core of the protocol implementation.

The most important module is the :mod:`~apart.core.node` module, that codes the
behavior of nodes, and thus the protocol. The :mod:`~apart.core.network` module
is also important, since it manages the nodes, the topology graph, and more
generally, the entire network.c

Other modules (:mod:`~apart.core.tables`, :mod:`~apart.core.messages`,
:mod:`~apart.core.pools`:mod:`~apart.core.controlled_traffic_rates`) are
secondary, and provide are used by the :mod:`~apart.core.node` module
(respectively, they code the node's tables,  the messages exchanged, the
neighbor batching pools, and the traffic counters).
'''