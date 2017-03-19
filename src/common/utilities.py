'''
This module contains miscelaneous functions such as prime number generation, etc.
'''

from collections import defaultdict
from fractions import Fraction
import functools
import math
import operator


# def primes(n):
#     if n == 2:
#         return [2]
#     elif n < 2:
#         return []
# 
#     res = []
#     p = 2
#     while p <= n:
#         for i in range(2, p):
#             if p % i == 0:
#                 p = p + 1
#         res.append(p)
#         p = p + 1
#     return res
# 
# def euclidean_dist(p1, p2):
#     """Returns the euclidean distance between two points in 2D space.
#     
#     Each point should contain a list with two values, the first 
#     for the coordinate x, the second for y
#     
#     """
# 
#     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def pprinttable(headers, rows):
    """Prints a table with headers.
    
    Used to print nodes' routing tables, mainly.
    
    Args:
        headers (list of str): the column titles
        rows (list of list of any): the rows and data they contain
         (must contain lists all of the same size, and of the same size as `header`)
        
    """
    if len(rows) > 0:
        if len(rows[0]) != len(headers):
            print("Error in printing table: number of headers ({}) does not match the number of values ({}):".format(len(headers), len(rows[0])))
            print("Headers: {}".format(headers))
            print("First row: {}".format(rows[0]))
            return
        
        values_lens = []
        for i in range(len(rows[0])):
            values_lens.append(len(str(max([x[i] for x in rows] + [headers[i]], key=lambda x:len(str(x))))))

        formats = []
        hformats = []
        for i in range(len(rows[0])):
            if isinstance(rows[0][i], int) and False:
                formats.append("%%%dd" % values_lens[i])
            else:
                formats.append("%%%ds" % values_lens[i])
            hformats.append("%%-%ds" % values_lens[i])

        pattern = " | ".join(formats)
        hpattern = " | ".join(hformats)
        separator = "-+-".join(['-' * n for n in values_lens])
        print(hpattern % tuple(headers))
        print(separator)
        for line in rows:
            print(pattern % tuple(line))
    else:
        print(" | ".join(headers))
        print("Empty Table")


def print_recursive_structure(data):
    """Prints a recusrsive structure.
    
    Adds indentations, and handles dict, list, sets, and defaultdict. 
    
    Used for debug purposes.
    """
    def print_with_indent_level(str, indent_level):
        print(('   ' * indent_level) + str)

    queue = [data]
    indent_level = 0
    while len(queue) > 0:
        data = queue.pop()
        if isinstance(data, defaultdict):
            print_with_indent_level('defaultdict('+str(data.default_factory())+'){', indent_level)
            indent_level += 1
            queue.append((None, '}'))
            for t in data.items():
                queue.append(t)
        elif isinstance(data, dict):
            print_with_indent_level('{', indent_level)
            indent_level += 1
            queue.append((None, '}'))
            for t in data.items():
                queue.append(t)
        elif isinstance(data, list):
            if len(data) <= 5 and indent_level > 0:
                print_with_indent_level(str(data), indent_level)
            else:
                indent_level += 1
                queue.append((None, ''))
                data.reverse()
                queue.extend(data)
        elif isinstance(data, tuple) and data[0] is not None:
            print_with_indent_level(str(data[0]) + ':', indent_level)
            indent_level += 1
            queue.append((None, ''))
            queue.append(data[1])
        elif isinstance(data, tuple) and data[0] is None:
            indent_level -= 1
            if data[1] != '':
                print_with_indent_level(data[1], indent_level)
        else:
            print_with_indent_level(str(data), indent_level)

def range1(*args):
    """Simple alias for range, with an inclusive upper bound.
    
    For instance, ``range1(0, 10)`` is equivalent to ``range(0, 11)``
    """
    if len(args) == 1:
        return range(args[0]+1)
    elif len(args) == 2:
        return range(args[0], args[1]+1)
    elif len(args) == 3:
        return range(args[0], args[1]+1, args[2])
    
    return range(*args)

def make_hashable(x):
    """Makes a recursive structure hashable.
    
    This function transforms the argument `x` from list to tuples, and from dict
    to tuples of tuples, and set to frozenset
    
    The function is applied recursively is `x` is a list that contains lists for instance.
    
    Args:
        x (*): the structure to make hashable
        
    Returns:
        any: the argument `x` transformed to be hashable by python
    """
    try:
        hash(x)
        return x
    except TypeError:
        if isinstance(x, list):
            res = tuple((make_hashable(y) for y in x))
        elif isinstance(x, defaultdict):
            res = make_hashable(("defaultdict("+str(x.default_factory())+")", tuple((make_hashable(k), make_hashable(v)) for k, v in x.items())))
        elif isinstance(x, dict):
            res = tuple((make_hashable(k), make_hashable(v)) for k, v in x.items())
        elif isinstance(x, set):
            res = frozenset((make_hashable(y) for y in x))
        else:
            try:
                res = type(x)((make_hashable(y) for y in iter(x)))
            except TypeError as e:
                raise TypeError("Can not make value '{}' of type {} hashable".format(x, type(x).__name__))
        # Check good hasing
        try:
            hash(res)
        except Exception as e:
            assert False, "Failed to make '{}' hashable: {}: '{}'".format(x, type(e).__name__, e)
        return res

def comb(n,k): 
    """Computes the combination of k elements in n"""
    return int( functools.reduce(operator.mul, (Fraction(n-i, i+1) for i in range(k)), 1) )
