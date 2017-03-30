# Author: Antoine Guellier
# Copyright (c) 2017 Université de Rennes 1
# License: CeCILL. The full license text is available at:
#  - http://www.cecill.info/licences/Licence_CeCILL_V2.1-fr.html


"""
This module models the routing tables used by nodes.

It provides an abstract base class, :class:`.SqliteRoutingTable`, and two
concrete classes: :class:`.RoutingTable` and :class:`.PrevRoutingTable`.
These two table together form the routing table of the node.

This module also defines the :exc:`RoutingTableError` exception.
"""
from collections import defaultdict
from sqlite3 import connect
import sqlite3

from apart.crypto import Ctxt




# The implementatio nof the routing table is quite complex. It is based on an
# in-memory sqlite3 database. There is one table for the RoutingTable, and one
# for the PrevRoutingTable.

# A crucial point to understand is that **all nodes in the network share the
# same sqlite3 database, and the same sql table**. Hence, we resort to a lot of
# *static* variables and methods, and even to metaclasses

# We also wanted to allow an access to table fields by diong e.g.
# RoutingTable.PSEUDO. This resulted in quite a mess: the fields are coded in a
# list of strings, and stored in a static attribute (of RoutingTable for
# instance), and the metaclass __MetaRoutingTable overrides the __new__ function
# of RoutingTable, to augment the class with other static attributes, and an
# overriden __getattr__ function in __MetaRoutingTable that intercepts calls
# such as RoutinTable.PSEUDO and answers adequately. Note that, RoutingTable
# never has a PSEUDO static member.

# On top of that, we wanted to be able to serialise (pickle) the routing tables,
# and thus, we had to find a way to extract and re-write the database from and
# to memory without loosing anything.



class __MetaRoutingTable(type):
        def __new__(cls, name, parent, attributes):
            # Here, we override the __new__ function of nodes of type
            # "__MetaRoutingTable". In this overriden __new__, we check that:
            # the class defines static _fields and _table_name attributes, and
            # that certain key fields are present in _fields.
            
             
            cls_declared_new = attributes.get('__new__', None)
            def cls___new__(cls2, *args, **kwargs):
                missing_static_attributes = set(["_fields", "_table_name"]).difference(cls2.__dict__.keys())
                if len(missing_static_attributes) > 0:
                    raise TypeError("Can't instantiate abstract class {} without static attributes {}".format(name, ", ".join(missing_static_attributes)))
                if not isinstance(cls2.__dict__['_fields'], list):
                    raise TypeError("Can't instantiate abstract class {} with static attributes _fields = {}".format(name, cls2.__dict__['_fields']))
                if not isinstance(cls2.__dict__['_table_name'], str):
                    raise TypeError("Can't instantiate abstract class {} with static attributes _table_name = {}".format(name, cls2.__dict__['_table_name']))
                if cls2.__dict__['_fields'][:2] != ['ROWID', 'NODE']:
                    raise TypeError("Can't instantiate class {}: it does not exhibit 'rowid' and 'node' as its two first fields".format(name))
                
                if cls_declared_new is not None:
                    return cls_declared_new(cls2, *args, **kwargs)
                else:
                    return parent[0].__new__(cls2)
            
            attributes["__new__"] = cls___new__ 
            
            
            # In a second time, when we create the a class of type
            # __MetaRoutingTable, we augment the class with other static
            # members. Namely, it processes the field list, and produces
            # derivatives of this list. This is done once and for all here.
            if "_fields" in attributes and "_table_name" in attributes \
            and attributes['_fields'] is not None and attributes['_table_name'] is not None:
                attributes['_prefixed_fields'] = [attributes['_table_name'] + "_" + f.lower() for f in attributes['_fields']]
                attributes['_fields_to_prefixed_fields'] = dict(zip(attributes['_fields'], attributes['_prefixed_fields']))
                                      
            return type.__new__(cls, name, parent, attributes)


        def __getattr__(self, *args, **kwargs):
            # When accessing e.g. RoutingTable.PSEUDO, this function is called.
            # If arg[0] is a valid field of the routing table (stored in the
            # static attribute "_fields_to_prefixed_fields" of the table), this
            # function returns the string representing the SQl string associated
            # to this field.
            try:
                if len(args) == 1 and args[0] in self._fields:
                    return self._fields_to_prefixed_fields[args[0]]
            except:
                pass
            
            return type.__getattribute__(self, *args, **kwargs)
        
 
        @property
        def table_name(self):
            """str: the name of the table"""
            return self._table_name
          
        @property
        def fields(self):
            """list of str: the list of fields of the table"""
            return self._prefixed_fields

        


class SqliteRoutingTable(object, metaclass=__MetaRoutingTable):
    """Abstract base class for routing tables.
        
    Manages the tables in an in-memory sqlite3 database. Creates one table
    per type of routing table, but all nodes use the same sql database and
    tables.
    
    """

    # The three following functions are there to manage the instances of routing
    # tables. It is actally quite complex, notably if several networks are run
    # in the same python execution. Indeed, there can only be one in-memory sql
    # database in the version of sqlite3 that we use...
    __routing_tables_db_conn = {}
    __nb_instances = {}
    __is_table_initialised = {}
    
    # Static memebers that **must** be redefined in derived classes
    _fields = None
    _table_name = None


    def __init__(self, node, network_uid):
        # Here, a lot of ugly hacks are done to ensure the good functioning of
        # our tables implementation in the case of multiple network runs in-
        # sequence (in the same python execution).
        
        
        if network_uid not in type(self).__routing_tables_db_conn: 
            # If the network is new (according to its uid), then create the in-
            # memory sqlite2 db for that particular network run IMPORTANT :
            # there can not be several network runs in parallel in the same
            # python execution, because the version of sqlite3 used allows only one in-memory db 
            sqlite3.register_adapter(Ctxt, lambda x: str(x)) 
            sqlite3.register_converter("CTXT", lambda x: eval(x))
            type(self).__routing_tables_db_conn[network_uid] = connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
            
            # Set the row factory
            type(self).__routing_tables_db_conn[network_uid].row_factory = sqlite3.Row
        
        if network_uid not in type(self).__is_table_initialised:
            type(self).__is_table_initialised[network_uid] = defaultdict(lambda: False)
        
        # Then, if the network is new or not, for each new instance of
        # RoutingTable or PrevRoutingTable (or any derived clases from
        # SqliteRoutingTable), keep track of the number of instances. (this will
        # be used in __del__)
        if network_uid not in type(self).__nb_instances: 
            type(self).__nb_instances[network_uid] = 1
        else:
            type(self).__nb_instances[network_uid] += 1
            
        
        
        self._network_uid = network_uid
        self._node = node
        self._db = type(self).__routing_tables_db_conn[network_uid].cursor()
        
        # If this is the first instantiation on that particular subclass of
        # SqliteRoutingTable in that particular network run, initialise the
        # table (i.e. create the SQl table)
        if not type(self).__is_table_initialised[network_uid][type(self)]:

            type(self)._init_table(type(self).__routing_tables_db_conn[network_uid].cursor())
            type(self).__is_table_initialised[network_uid][type(self)] = True


    def __del__(self):
        # For each instance of subclass of SqliteRoutingTable deleted, decrease
        # the counter of instances for that subclass.
        type(self).__nb_instances[self._network_uid] -= 1
        
        # When the counter reaches zero, close and delete the sqlite3 database
        if type(self).__nb_instances[self._network_uid] == 0:
            type(self).__routing_tables_db_conn[self._network_uid].commit()
            type(self).__routing_tables_db_conn[self._network_uid].close()
            del type(self).__nb_instances[self._network_uid]
            del type(self).__routing_tables_db_conn[self._network_uid]
    
    @classmethod
    def sqlite_db_cursor(cls, network_uid=None):
        """:obj:`sqlite3.Cursor`: the sqlite3 cursor for the database. 
        
        Used for measure and debug purposes to access the routing tables
        directly, and make more efficient sql queries
        """ 
        if network_uid is None:
            print(cls.__routing_tables_db_conn)
            network_uid = list(cls.__routing_tables_db_conn.keys())[-1]
            
        if cls.__routing_tables_db_conn[network_uid] is not None:
            return cls.__routing_tables_db_conn[network_uid].cursor()
        else:
            return None

    def insert_entry(self, **kwargs):
        """Insert a routing table entry
        
        The arguments of this function depend on the concrete routing table that
        on which this function is called. This function checks that the
        ``**kwargs`` in argument specifies all the fields of the said routing
        table (in accordance with the static :attr:`_fields` member of the class
        that defines the routing table, see :class:`.RoutingTable` and
        :class:`.PrevRoutingTable` for a description of those fields)).
        
        Args:
            **kwargs (dict str->any): the dict specifying the entry to insert. 
                This dict is indexed by the fields of the routing table, and
                **must** specify a value for all of them.
            
        Returns:
            int: the (unique) id of the new entry
        
        Raises:
            RoutingTableError: if the entry specified in input does not
                match the fields of the considered table
        """
        self.insert_entries([kwargs])
        
        return self._db.lastrowid
    
    def insert_entries(self, entries):
        """Insert several routing table entries.
        
        Args:
            entries (list of dict str->any): a list of dict adequate for input to :meth:`.insert_entry`
            
        Raises:
            RoutingTableError: if at least one of the entries specified in argument does not
                match the fields of the considered table
        """
        table_name = type(self).table_name
        table_fields = type(self).fields[2:]
        
        if not isinstance(entries, list) or len(entries) == 0:
            return 
        
        values = [] 
        try:
            for entry in entries: 
                values.append(self._node)
                for f in table_fields: values.append(entry[f])
        except KeyError as e:
            for entry in entries:
                missing_fields = set(table_fields) - set(entry.keys())
                if missing_fields:
                    raise RoutingTableError("Impossible to insert entry in table {}. The following fields are missing: {}".format(table_name, ", ".join(missing_fields)))
            else:
                raise e
            
        query = "INSERT INTO "+table_name+" VALUES "+(",".join(["("+(",".join("?"*(len(table_fields)+1)))+")"]*len(entries)))
#         print(query, values)
        self._db.execute(query, values)
        
    def lookup(self, fields=None, constraints=None, constraints_bindings=None, order_by=None):
        """Lookup one or several routing table entries
        
        Args:
            fields (list of string, optional): the table fields that must 
                be returned by the lookup (Default: None)
            
            constraints (string or dict, optional): the constraints of the lookup. Can
                either be a string containing text suitable for a WHERE SQL
                clause, with '?' in place of values; either a dict that maps a
                field to a value. In the latter case, the WHERE clause is
                constructed as a conjunction of the constrain on each field in
                the dict. (Default: None)
                
            consraints_bindings (list of any): if ``constraints`` is provided as
                a string (a SQL-like WHERE clause), then the values in this
                argument are used as bindings for the '?' place holders in ``constraints``
            
            order_by (list of str): list of fields, specifying the order in
            which results should be provided
        
        Returns:
            (list of :obj:`sqlite3.Row`): the list of routing table entries
            returned according to the constraints of the lookup
        """
        table_name = type(self).table_name
        try:
            fields = [table_name+"_rowid"] + fields
        except:
            fields = type(self).fields
        fields_str = " "+", ".join(fields)
        fields_str = fields_str.replace(" "+table_name+"_rowid", " "+table_name+".rowid AS "+table_name+"_rowid")
        
        
        where_str = " WHERE "+table_name+"_node=?"
        bindings = [self._node]
        where_str_rest, bindings_rest = self._constraints_to_where_statement(constraints, constraints_bindings)
        where_str += where_str_rest
        bindings += bindings_rest
            
        if order_by is None or not isinstance(order_by, list):
            order_by = []
        order_by = [table_name+"_node"] + order_by
            
        query = "SELECT {} FROM "+ table_name + where_str + " ORDER BY " + (", ".join(order_by))
        query = query.replace(" "+table_name+"_rowid", " "+table_name+".rowid").format(fields_str)
        bindings = tuple(bindings)
#         print(query, constraints_bindings)
        self._db.execute(query, bindings)
        
        return self._db.fetchall() 

    
    def joint_lookup(self, other_table, fields=None, join_on=None, constraints=None, constraints_bindings=None, order_by=None):
        """Perform mode advanced lookups, involving a SQL JOIN
        
        This function is often used by nodes to perform a lookup on a previous
        hop (in table :class:`.PrevRoutingTable`), joined on the corresponding
        entry of the table :class:`.RoutingTable`. This avoids doing two SQL
        queries. See :class:`.PrevRoutingTable` for more details.
        
        Args:
            other_table (:obj:`.RoutingTable` or :obj:`.PrevRoutingTable`): the 
                table to join on the first (the first being the one on which 
                this function is called)
            fields (list of string, optional): see :meth:`.lookup` (Default: None)
            join_on (2-tuple of str, optional): the field of each of the rwo routing tables on which to perform the join (Default: None)
            constraints (string or dict of str->any, optional): see :meth:`.lookup` (Default: None)
            constraints_bindings (list of any): see :meth:`.lookup` (Default: None)
            order_by (list of str): see :meth:`.lookup` (Default: None)
        
        Returns:
            (list of :obj:`sqlite3.Row`): the list of routing table entries
            returned according to the constraints of the lookup
        """
        table_name = type(self).table_name
        
        try:
            fields_str = " "+", ".join(fields)
        except:
            fields_str = " "+", ".join(type(self).fields + other_table.fields)
        
        fields_str = fields_str.replace(" "+table_name+"_rowid", " "+table_name+".rowid AS "+table_name+"_rowid") \
                                .replace(" "+other_table.table_name+"_rowid", " "+other_table.table_name+".rowid AS "+other_table.table_name+"_rowid")
                    
        try:    
            join_str = join_on[0] + " = " + join_on[1]
        except:
            return []
        
        where_str = " WHERE "+table_name+"_node=? AND "+other_table.table_name+"_node=?"
        bindings = [self._node, self._node]
        where_str_rest, bindings_rest = self._constraints_to_where_statement(constraints, constraints_bindings)
        where_str += where_str_rest
        bindings += bindings_rest
        
        if order_by is None or not isinstance(order_by, list):
            order_by = []
        order_by = [table_name+"_node"] + order_by
        
        query = "SELECT {} FROM "+table_name+" LEFT JOIN "+other_table.table_name+" ON "+join_str \
                + where_str + " ORDER BY "+ (", ".join(order_by))
        query = query.replace(" "+table_name+'_rowid', " "+table_name+".rowid")
        query = query.replace(" "+other_table.table_name+'_rowid', " "+other_table.table_name+".rowid").format(fields_str)
        bindings = tuple(bindings)
#         print(query)
        self._db.execute(query, bindings)
            
        return self._db.fetchall()
    
    
    def update_entries(self, update=None, constraints=None, constraints_bindings=None):
        table_name = type(self).table_name
        
        set_str = ""
        bindings = []
        for (f, v) in update.items():
            set_str += f+"=?, "
            bindings.append(v)
        set_str = set_str[:-2]
        
    
        where_str = " WHERE "+table_name+"_node=?"
        bindings.append(self._node)
        where_str_rest, bindings_rest = self._constraints_to_where_statement(constraints, constraints_bindings)
        where_str += where_str_rest
        bindings += bindings_rest

        
        query = "UPDATE "+ table_name +" SET " + set_str + where_str
        query = query.replace(" "+table_name+"_rowid", " "+table_name+".rowid")
        bindings = tuple(bindings)
#         print(query, bindings)
        self._db.execute(query, bindings)

    
    def remove_entries(self, constraints=None, constraints_bindings=None):
        """Remove entries from a routing table that fit into the specified constraints.
        
        Args:
            constraints (string or dict of str->any, optional): see :meth:`.lookup` (Default: None)
            constraints_bindings (list of any): see :meth:`.lookup` (Default: None)
        """
        
        table_name = type(self).table_name
        
        where_str = " WHERE "+table_name+"_node=?"
        bindings = [self._node]
        where_str_rest, bindings_rest = self._constraints_to_where_statement(constraints, constraints_bindings)
        where_str += where_str_rest
        bindings += bindings_rest
            
        query = "DELETE FROM "+table_name+ where_str
        bindings = tuple(bindings)
        #print(query, constraints_bindings)
        self._db.execute(query, bindings)
    
    def _constraints_to_where_statement(self, constraints, constraints_bindings):
        where_str = ""
        where_bindings = []
        try:
            where_str += " AND ("+constraints+")"
            where_bindings += constraints_bindings
        except:
            try:
                for (f, v) in constraints.items():
                    where_str += " AND "+f+"=?"
                    where_bindings.append(v)
            except:
                where_str = ""
                where_bindings = []
        return where_str, where_bindings
    
    
    # Special function for serialising (pickling) this weird routing table
    # implementation based on a memroy sqlite db
    def __getstate__(self):
        # The whole in-memory database is dumped into a string
        self.database_str = "".join(line for line in type(self).__routing_tables_db_conn[self._network_uid].iterdump())
        return self.__dict__.copy()
     
    def __setstate__(self, state):
        database_str = state.pop('database_str')
        init_db = (state['_network_uid'] not in type(self).__routing_tables_db_conn)
        # Put the boolean to True, so that the __init__ function of e.g. the
        # RoutingTable class does not try to create the tables: they will be
        # created and populated afterward
        # Note that, still, the below __ini__ will initialize
        # type(self).__routing_tables_db_conn if it is None
        type(self).__is_table_initialised[state['_network_uid']] = defaultdict(lambda: True)
        self.__init__(state['_node'], state['_network_uid'])
        
        
        # If this is the first table object that we unpickle, the following
        # condition is true. Then, reconstruct the full db from the script str
        if init_db:
            type(self).__routing_tables_db_conn[state['_network_uid']].executescript(database_str)
            
        
    
    
                

class RoutingTable(SqliteRoutingTable):
    """The main routing table of nodes. 
    
    Each node has a different instance of this class.
    
    This table features rows that describe routes that a given node knows. Each
    entry in the table exhibits the following fields:
        * ROWID (int): the (unique) identifier of the table entry
        * NODE (int): used internally, to differentiate which SQL table entry belongs to which node 
        * PSEUDO (int): pseudonym of the end-receiver of the route 
        * CONE (:obj:`~apart.crypto.Ctxt`): encryption of one used to encrypt/re-encrypt messages on the route
        * NEXT_NODE (int): index of the next node on the route (can be considered as the next hop's IP address) 
        * NEXT_CID (int): circuit identifier of the next hop 
        * REPROPOSED (bool): whether the node relayed this route to its neighbor. Used in the route proposal policy.  
        * IN_USE (bool): True if the node uses this route as end-sender, False if it uses it only as relay. 
        * TIMESTAMP (int): time at which the routing table entry was created 
        * ACTUAL_RCVR (int): index of the end-receiver of the route. used only for debug and statistics purposes. 
        * ACTUAL_LENGTH (int): number of hops between the node and the end-receiver of the route. Used only for debug and statistics purposes
        
    Each of these fields can be accessed by :obj:`.RoutingTable.FIELD`.
    """

    # The table's name and fields
    _table_name = "rt"
    _fields = ['ROWID', 
               'NODE', 
               'PSEUDO', 
               'CONE', 
               'NEXT_NODE', 
               'NEXT_CID', 
               'REPROPOSED', 
               'IN_USE', 
               'TIMESTAMP', 
               'ACTUAL_RCVR', 
               'ACTUAL_LENGTH']
    
    @classmethod
    def _init_table(cls, db_cursor):
        # Upon creating the first RoutingTable instance (in a given
        # network/simulation run), this function is called to create the SQL
        # table.
        rt_fields = [field_name.lower() for field_name in RoutingTable.fields[1:]]
        rt_spec = """CREATE TABLE IF NOT EXISTS {0} (
        {1} SMALLINT NOT NULL,
        {2} BIGINT(64)  NOT NULL,
        {3} CTXT NOT NULL,
        {4} SMALLINT NOT NULL,
        {5} BIGINT(64) NOT NULL,
        {6} BOOLEAN NOT NULL,
        {7} BOOLEAN NOT NULL,
        {8} BIGINT NOT NULL,
        {9} SMALLINT NOT NULL,
        {10} SMALLINT NOT NULL);
        CREATE INDEX rt_node_key ON {0}(rt_node);
        CREATE INDEX rt_localid_key ON {0}(rt_pseudo);""".format(RoutingTable._table_name, *rt_fields)
        
        db_cursor.executescript(rt_spec)
        
#     def insert_entry(self, **kwargs):
#         # Override the method only to 
#         return super().insert_entry(**kwargs)

class PrevRoutingTable(SqliteRoutingTable):
    """The routing table of nodes that specifies the *previous hops* associated to each entry of their main routing table.
    
    Each node has a different instance of this class. 
    
    This table features rows that describe (possibly multiple) previous hops
    associated to each route known by a given node. That is, each row in this
    table refers to a row in the main routing table :obj:`.RoutingTable` of the
    node. 
    
    To illustrate the use of this table: when a node receives a message from a
    neighbor node Y and with circuit identifier cid, it makes a lookup in this
    table for (PREV_NODE = Y, PREV_CID = cid), which allows to retrieve the
    ROWID of the associated entry in the main routing table. Consequently, the
    node makes a lookup in the main routing table (and then possibly relays the
    message on the specified next hop).
    
    Each entry in the table exhibits the following fields:
        * ROWID (int): the (unique) identifier of the table entry
        * NODE (int): used internally, to differentiate which SQL table entry belongs to which node 
        * PREV_NODE (int): the index of the previous node on the route  (can be considered as its IP address)
        * PREV_CID (int): the circuit identifier of the previous hop
        * RT_ROWID (int): the ROWID of the :obj:`.RoutingTable` entry that this previous hop relates to 
        
        Each of these fields can be accessed by :obj:`.PrevRoutingTable.FIELD`.
    """


    # The table's name and fields
    _table_name = "prevrt"
    _fields = ['ROWID', 'NODE', 'PREV_NODE', 'PREV_CID', 'RT_ROWID']

    @classmethod
    def _init_table(cls, db_cursor):
        # Upon creating the first PrevRoutingTable instance (in a given
        # network/simulation run), this function is called to create the SQL
        # table.
        
        prevrt_fields = [field_name.lower() for field_name in PrevRoutingTable.fields[1:]]
        prevrt_spec = """CREATE TABLE IF NOT EXISTS {0} (
        {1} SMALLINT NOT NULL,
        {2} SMALLINT NOT NULL,
        {3} BIGINT(64) NOT NULL,
        {4} INT  NOT NULL);
        CREATE INDEX prevrt_node_key ON {0}(prevrt_node);""".format(PrevRoutingTable._table_name, *prevrt_fields)
        
        db_cursor.executescript(prevrt_spec)
                
  
        
class RoutingTableError(Exception):
    """Raised when an invalid operation is performed on the routing tables""" 
    pass
    
if __name__ == '__main__':
    # Some basic tests

    print("Fields of PrevRoutingTable: {}".format(PrevRoutingTable.fields))
    
    print("Sqlite3 cursor : {}".format(SqliteRoutingTable.sqlite_db_cursor()))
    
    print("Instanciating RoutingTable")
    rt = RoutingTable(1)
    rt.insert_entry(**{RoutingTable.PSEUDO: -1,  RoutingTable.CONE: Ctxt(52, set([8899])), 
                        RoutingTable.NEXT_NODE: 3,  RoutingTable.NEXT_CID: 5588, 
                        RoutingTable.TIMESTAMP:  78936543, RoutingTable.REPROPOSED: True,
                        RoutingTable.IN_USE: True, RoutingTable.ACTUAL_RCVR: 1,
                        RoutingTable.ACTUAL_LENGTH: 3})
    print([v for v in rt.lookup()[0]])
    
    print("Sqlite3 cursor : {}".format(SqliteRoutingTable.sqlite_db_cursor()))
    
    print("Instanciating PrevRoutingTable")
    prt = PrevRoutingTable(1)
    
    print("Instanciating SqliteRoutingTable")
    shouldnotbepossible = SqliteRoutingTable(5)
        
    #===========================================================================
    # rt = RoutingTable(1)
    # 
    # # Testing insert and lookup
    # rt_entry = (0,2,3,5,6,'7',8,9,10,11)
    # rt._insert_entry(*rt_entry)
    # 
    # res = rt.lookup()
    # if res is None or len(res) != 1 or res[0] != rt_entry:
    #     print("Error 1")
    #     print(res)
    # 
    # # Testing lookup for particular _fields
    # res = rt.lookup([RoutingTable.PSEUDO, RoutingTable.NEXT_NODE])
    # if res is None or len(res) != 1 or res[0] != (2,8):
    #     print("Error 2")
    #     print(res)
    #     
    #===========================================================================
