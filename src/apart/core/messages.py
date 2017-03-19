'''
This module provides classes to manage messages exchanged by nodes in the protocol.

The main class is :class:`.LinkMsg`, which models a link message. The enum
classes :class:`.MsgFlag` and :class:`.MsgInnerFlag` respectively provides
constant coding the types of link and end-to-end messages.
'''
from enum import IntEnum

from apart.crypto import Ctxt


class MsgFlag(IntEnum):
    """An enumeration class giving the types of link messages in the network"""
    
    RTPROP = 1
    """The first message in a route proposition"""
    RTPROP_ANSWER = 2
    """Message sent by the proposee after receiving the :attr:`~apart.core.messages.MsgFlag.RTPROP` and :attr:`~apart.core.messages.MsgFlag.RTPROP_INFO` messages"""
    RTPROP_FINAL = 3
    """Message sent by the proposer, that terminates the route proposal"""
    RTPROP_RELAY_FWD = 4
    """The message to relay from proposer to receiver in a relayed proposal"""
    RTPROP_RELAY_BWD = 5
    """The message to relay from receiver back to proposer in a relayed proposal"""
    PAYLOAD = 6
    """A regular payload message"""
    DUMMY = 7
    """A dummy link message"""

class MsgInnerFlag(IntEnum):
    """An enumeration class giving the different flags an encapsulated message can have.
    
    These flags usually go into the first ciphertext of the link messages), and
    are in particular used to signal end-to-end dummy messages and oriented
    communication messages
    """ 
    
    DUMMY = 1
    """A dummy en-to-end message"""
    OCOM_INIT = 2
    """A message part of the oriented communication initialization, exchanged between sender and helper"""
    OCOM_RCV = 3
    """A message part of an oriented communication, for the end-receiver"""
    OCOM_CLOSE = 4
    """Indicates to the helper the closing of the session"""

    
    


class LinkMsg(object):

    def __init__(self, sent_by, sent_to, c1, c2, flag, cid=None, rcid=None, seq_index=None, additional_info={}) :
        """This is the main class of this module. It basically
        contains a header (an instance of :class:`.LinkMsgHeader`), two
        ciphertexts  ``c1`` and ``c2``, the identities of the link-sender and
        the link-receiver, plus additional info for debugging and computing
        statistics.
        
        """
        self._sent_by = sent_by
        self._sent_to = sent_to
        
        self._c1 = c1
        self._c2 = c2
        
        self._header = LinkMsgHeader(flag, cid, rcid, seq_index)
        
        self._additional_info = additional_info

        assert not (flag is MsgFlag.PAYLOAD or flag is MsgFlag.RTPROP_RELAY_FWD or flag is MsgFlag.RTPROP_RELAY_BWD) or ('end_sender' in additional_info and 'end_rcvr' in additional_info)
        assert not ('end_sender' in additional_info and 'end_rcvr' in additional_info) or (additional_info['end_sender'] != additional_info['end_rcvr']), str(flag)+", "+str(additional_info)
        assert not (isinstance(c1[0], MsgInnerHeader) and (c1[0].flag is MsgInnerFlag.OCOM_INIT or c1[0].flag is MsgInnerFlag.OCOM_RCV or c1[0].flag is MsgInnerFlag.OCOM_CLOSE) )or ('is_ocom' in additional_info)
        
    @staticmethod
    def create_dummy(sent_by, sent_to, link_key=None):
        """Convenience function providing a shortcut to create a link dummy message.
        
        Args:
            sent_by (int): index of the link-sender node
            sent_to (int): index of the link-receiver node
            link_key (any, optional): not used in this implementation
            
        Returns:
            :obj:`.LinkMsg`: a dummy link message (with a *dummy* flag in its header)
        """
        # For a fully fledged implementation, here, the DUMMY flag should be
        # encrypted with the link key
        return LinkMsg(sent_by, sent_to, c1=Ctxt(0,0), c2=Ctxt(0,0), flag=MsgFlag.DUMMY)
    
    @property
    def sent_by(self):
        """int: the index of the link-sender node"""
        return self._sent_by
    
    @property
    def sent_to(self):
        """int: the index of the link-receiver node"""
        return self._sent_to
    
    @property
    def header(self):
        """:obj:`.LinkMsgHeader`: the full message header"""
        return self._header
    
    @property
    def flag(self):
        """:obj:`.MsgFlag`: the flag contained in  the message header (see :attr:`.LinkMsgHeader.flag`)"""
        return self.header.flag
    
    @property
    def cid(self):
        """int: the cid value contained in  the message header (see :attr:`.LinkMsgHeader.cid`)"""
        return self.header.cid
    
    @property
    def rcid(self):
        """int: the rcid contained in the message header (see :attr:`.LinkMsgHeader.rcid`)"""
        return self.header.rcid
    
    @property
    def seq_index(self):
        """int: the sequence index contained in the message header  (see :attr:`.LinkMsgHeader.seq_index`)"""
        return self.header.seq_index
    
    @property
    def c1(self):
        """:obj:`~apart.crypto.Ctxt`: the first Elgamal ciphertext of the message"""
        return self._c1
    
    @property
    def c2(self):
        """:obj:`~apart.crypto.Ctxt`: the second Elgamal ciphertext of the message"""
        return self._c2
    
    @property
    def additional_info(self):
        """dict str->any: Adidtional information transported along with the message. For debug and measures purposes"""
        # This returns the dict by reference. The dict _additional_info can thus
        # be modified, which is what is needed
        return self._additional_info

    def __str__(self):
        memory_location = super().__repr__()[-11:-1]
        return "LinkMsg<{}>({}, {}, {}, from={}, to={})".format(memory_location, self.header, self.c1, self.c2, self.sent_by, self.sent_to)
      
    def __repr__(self):
        return self.__str__()
    
#     def __eq__(self, other):
#         return self.lala == other.lala
    


class LinkMsgHeader(object):
    def __init__(self, flag, cid=None, rcid=None, seq_index=None):
        """A link message header, containing at least a flag (type), and possibly, a cid, a rcid, and a sequence index.
    
        Args:
            flag (:obj:`.MsgFlag`): the flag of the message, specifying its type 
            cid (int, optional): the circuit identifier value cid (Default: None)
            rcid (int, optional): the reverse circuit identifier value rcid (Default: None)
            seq_index (int, optional): the sequence index of the message, used in route proposals (Default: None)   
        """
        
        # Note that a named tuple could have been used to represent link message headers. The
        # only reason a class was created is to overload the __str__ function to a
        # nice string for debug
        
        self.__flag = flag
        self.__cid = cid
        self.__rcid = rcid
        self.__seq_index = seq_index
        
    @property
    def flag(self): 
        """:obj:`.MsgFlag`: the flag specifying the type of the message"""
        return self.__flag
    
    @property
    def cid(self): 
        """int: the circuit identifier.
        
        This value can be None, for messages with flag :attr:`.MsgFlag.DUMMY`
        """
        return self.__cid
    
    @property
    def rcid(self): 
        """int: the reverse circuit identifier. 
        
        The only type of messages that should have a reverse circuit identifier
        are those with flags :attr:`.MsgFlag.RTPROP_RELAY_FWD` and
        :attr:`.MsgFlag.RTPROP_RELAY_BWD`. For other messages, this value is None.
        """
        return self.__rcid
    
    @property
    def seq_index(self): 
        """int: the sequence index of the message. 
        
        Sequence indexesUsed are used to differentiate the first and second
        message in a route proposal. Indeed, every step of the route proposals
        require two messages. Having this sequence index makes it easier to
        differentiate them.<
        
        The only type of messages that should have a sequence index
        are those relating to route proposals. For other messages, this value is None.
        """
        return self.__seq_index
    
    def __str__(self):
        s = str(self.flag)
        if self.cid is not None:
            s += " || cid({})".format(self.cid)
        if self.rcid is not None:
            s += " || rcid({})".format(self.rcid)
        if self.seq_index is not None:
            s += " || seq({})".format(self.seq_index)
        return s
    
    def __repr__(self):
        return self.__str__()

class MsgInnerHeader(object):
    def __init__(self, flag, ocomid=None, seq_index=None):
        """Inner header of a message of type :attr:`.MsgFlag.PAYLOAD`.
        
        An inner header contains data regarding the management of  oriented
        communications and their initializations, and regarding end-to-end dummy
        messages 
        
        Args:
            flag (:obj:`.MsgInnerFlag`): the inner flag, specifying whether the message is an end-to-end dummy, 
                    or one pertaining to an oriented communication
            ocomid (int, optional): the oriented communication identifier (Default: None)
            seq_index (int, optional): the sequence index of messages in an oriented communication (Default: None)
        
        """
        
        self.__flag = flag 
        self.__ocomid = ocomid 
        self.__seq_index = seq_index
        
    @property
    def flag(self): 
        """:obj:`.MsgInnerFlag`: the inner flag.
        
        This flag specifies whether the message is an end-to-end dummy, or one
        pertaining to an oriented communication. In th elatter case, it also
        specifies which message it is as part of the initialisation of the communication.
        """
        return self.__flag
    
    @property
    def ocomid(self): 
        """int: the oriented communication identifier.
        
        Present in all payload and rtproprelay messages of oriented communications.
        
        This value is None for end-to-end dummy messages (with inner flag :attr:`.MsgInnerFlag.DUMMY`).
        """
        return self.__ocomid
    
    @property
    def seq_index(self): 
        """int: the sequence index of messages in an oriented communication
        
        This allows the actors of an oriented communication to know which
        message contains which information. Without this sequence index, the
        end-sender and indirectio nnode would not be able to run the oriented
        communicatio ninitlaisation properly.
        
        This value is None for end-to-end dummy messages (with inner flag :attr:`.MsgInnerFlag.DUMMY`).
        """
         
        return self.__seq_index
    
    def __str__(self):
        s = str(self.flag)
        if self.ocomid is not None:
            s += " || ocomid({})".format(self.ocomid)
        if self.seq_index is not None:
            s += " || seq({})".format(self.seq_index)
        return s
    
    def __repr__(self):
        return self.__str__()


class EndToEndDummyError(Exception):
    """Raised when a node is not able to produce an end-to-end dummy"""
    pass
        