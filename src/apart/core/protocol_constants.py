# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


'''
This module simply contains constants used throughout the protocol.

The module defines in particular: 
    * GROUP_P and GROUP_G, constants for group operations
      (those are hardcoded to 2 and and 100103).
    * A null value (F_NULL = -1).
    * An enum class :class:`.RtPolicyReason` that codes
      the reasons of acceptation/refusal of route proposals.
'''
from enum import Enum

# The parameters of the ElGamal
GROUP_G = 2  # Group generator g (that does not need to be a group generator
GROUP_P = 100103  # The characteristic of the group (in our case, we simplify : it is also the order)

# A flag used during the route proposition process, when the destination
# gets the RtPropAns message, "decrypts" the MaxHopCount-HopCount value,
# and it happens that this value is 0. Instead of sending back the value of
# the pseudo, the destination sends back this flag
F_RT_TOO_LONG = -1

# A flag representing a null value (localid, nexthop, etc)
F_NULL = -1


# Reasons to accept/refuse a rt proposal
class RtPolicyReason(Enum):
    '''Reasons to accept or refuse a route proposal.
    
    Refusal reasons **must** begin with *REFUSE_*, while accept reasons **must**
    begin with *ACCEPT_*.
    '''
    REFUSE_ITSELF = 0
    """Route refused because the route is towards the node itself"""
    REFUSE_ENC_DEC = 1
    """Route refused according the the *encrypted decision* process"""
    REFUSE_REACCEPT = 2
    """Route refused by random choice, and because the node already knows at least one route towards the end-receiver"""
    REFUSE_TOO_MANY_ROUTES_NO_REPLACEMENT = 3
    """Route refused because the node already knows enough route towards the related end-receiver, and no replacement is performed."""
#     REFUSE_TOO_MANY_ROUTES = "Too many routes towards pseudo"
#     REFUSE_NO_REPLACEMENT = "Enough routes towards pseudo AND no replacement"
#     REFUSE_RANDOM = "Random choice: refuse"
    
    
    ACCEPT_FIRST_KNOWN_ROUTE = 4
    """Route accepted because the route is the first that the node lears towars the given end-receiver."""
    ACCEPT_REACCEPT_NO_REPLACEMENT = 5
    """Route accepted by random choice, with no replacement of an old entry."""
    ACCEPT_REACCEPT_REPLACEMENT = 6
    """Route accepted by random choice, with replacement of an old entry."""
#     ACCEPT_REPLACEMENT = "Enough routes towards pseudo BUT replacement"
#     ACCEPT_RANDOM = "Random choice: accept"
    
    @staticmethod
    def to_human_readable(reason):
        """Transforms an instance of :class:`.RtPolicyReason` intro a human-readable string"""
        if reason is RtPolicyReason.REFUSE_ITSELF:
            return "Receiver is itself"
        elif reason is RtPolicyReason.REFUSE_ENC_DEC:
            return "Route too long or presence of loop"
        elif reason is RtPolicyReason.REFUSE_REACCEPT:
            return "Random choice of refusing a route towards a receiver already known"
        elif reason is RtPolicyReason.REFUSE_TOO_MANY_ROUTES_NO_REPLACEMENT:
            return "Route was going to be re-accepted, but not by replacement, and max number of route reached"
        elif reason is RtPolicyReason.ACCEPT_FIRST_KNOWN_ROUTE:
            return "First route learned towards a given proposer"
        elif reason is RtPolicyReason.ACCEPT_REACCEPT_NO_REPLACEMENT:
            return "Accepting a route towards an already known receiver, without replacement"
        elif reason is RtPolicyReason.ACCEPT_REACCEPT_REPLACEMENT:
            return "Accepting a route towards an already known receiver, with replacement"
        else:
            return "NotARtPolocyReasons"
    
    def __lt__(self, other):
        return str(self) < str(other)
