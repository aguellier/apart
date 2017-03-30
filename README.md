# REPOSITORY IN CONSTRUCTION

* Commit second part of code

# apart
Proof-of-concept implementation of a privacy-preserving Internet overlay 

This repository contains the academic implementation of the protocol proposed as part of my PhD thesis. This code is made available for the community to reproduce the results.

[1] Antoine Guellier. "Strongly Private Communicaitons in a Homogeneous Network". CentraleSupélec (to be published)

This thesis was funded by the University of Rennes 1.

## Description 
 
The protocol is meant to provide ironclad privacy in Internet communications. It is meant to be run as a peer-to-peer Internet overlay, providing privacy to individuals around the world. The design is in particular inspired from [Tor](https://torproject.org/), and builds on the scientific literature about *mixnets*. It however provides much stronger privacy guarantees. In particular, the protocol functions on a *homogeneous network*, in which all nodes simultaneously act as client, and as relay servers for other nodes. This allows each node to conceal its own traffic in the traffic it relays for other peers. 

This repository only contains a basic, insecure implementation, for purposes of measuring the privacy and performances that can be expected from the protocol. More accurately, it is a *simulation* of the protocol that is implemented, using the [Simpy](http://simpy.readthedocs.io/en/latest/) discrete-event simulator. The code also uses the [Networkx](http://networkx.readthedocs.io/en/stable/) library to generate and manipulate the network graph, and [matplotlib.pyplot](http://matplotlib.org/) to draw various graphs.

## Dependencies

The project is written for Python 3.4, with the following packages and versions:
* [Simpy](http://simpy.readthedocs.io/en/latest/) 3.0.10
* [Networkx](http://networkx.readthedocs.io/en/stable/) 1.11
* [Matplotlib](http://matplotlib.org/) 1.3.1, for its pyplot subpackage
* dill 0.2.5
* Numpy 1.8.2
* Scipy 0.13.3

Sphinx 1.5.3 is used to generate the documentation, along with the ReadTheDocs theme (v. 0.2.4)

## License

This software is licensed under the [CeCILL license](http://www.cecill.info), created by the Inria, CNRS, and CEA French research institutes. It is inspired from and compatible with the GNU GPL license.

Copyright (c) 2017 Université de Rennes 1



