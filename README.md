# APART

This repository contains the academic implementation of the protocol proposed as part of my PhD thesis. This code is made available for the community to reproduce the results.

[1] Antoine Guellier. "Strongly Private Communications in a Homogeneous Network". CentraleSupélec (to be published)

This thesis was funded by the University of Rennes 1.

## Description 
 
The protocol is meant to provide ironclad privacy in Internet communications. It is meant to be run as a peer-to-peer Internet overlay, providing privacy to individuals around the world. The design is in particular inspired from [Tor](https://torproject.org/), and builds on the scientific literature about *mixnets*. It however provides much stronger privacy guarantees. In particular, the protocol functions on a *homogeneous network*, in which all nodes simultaneously act as client, and as relay servers for other nodes. This allows each node to conceal its own traffic in the traffic it relays for other peers. 

This repository only contains a basic, insecure implementation, for purposes of measuring the privacy and performances that can be expected from the protocol. More accurately, it is a *simulation* of the protocol that is implemented, using the [Simpy](http://simpy.readthedocs.io/en/latest/) discrete-event simulator. The code also uses the [Networkx](http://networkx.readthedocs.io/en/stable/) library to generate and manipulate the network graph, and [matplotlib.pyplot](http://matplotlib.org/) to draw various graphs.

## Getting Started: Running Simulations

The documentation provides details on how to run simulations, and gathering statistics on network runs. 

To run a simulation, one possibility is to create a `main.py` file that imports the `apart.simulation` module. The latter provides a `run_simulation()` function, that accepts parameters for the network (*e.g.* the number of nodes), and for the simulation (*e.g* logging level, pieces of information that should be recorded during the network run). The execution of the function may take some time for a number of node greater than 10. It also makes log outputs to the standard output, letting the user follow the stages of the simulation. Ultimately, the function returns a `NetworkManager` instance, which in particular contains a `net_stat` attribute, itself containing all gathered statistics and pieces of information on the network and the simulation. 

The user can then exploit these pieces of information in the way she sees fit. However, it is more interesting to run fully-fledged measure campaign, in which many simulations are run with different network parameters. This is possible with the material provided in the `measures` package. It provides ways to run many network simulations, and compute statistics on them. This allows *e.g.* to obtain an average over many runs and over many possible network topologies of the latency in communications. The `measures` package further provides ways to draw graphs from the results.

The `examples` folder includes several modules that make use of the benchmark package `measures`: 
* `measure_general_efficiency`, allowing in particular to measure communication latency, 
* `measure_route_prop`, allowing to measure various metrics on the route proposal mechanism proposed in the protocol,
* `measure_privacy`, allowing to measure the expected privacy that users get from the protocol.
* `measure_all`, that simultaneously performs the measures of all three modules at once.  

Note that running these modules may take some time. Parameters for simulations can be easily changed in the respective *main part* of each module. The user is also free to create its own modules and construct custom measure campains (*e.g.*  based on the same model as the above modules)

## Documentation

The sources of the documentation are located in `docs/`. Documentation is generated using Sphinx (v. 1.5.3), and the ReadTheDocs theme (v. 0.2.4). It is thus required to have those packages installed (*via* the system package manager, pip, or any virtual environment).

To build the documentation (in html format), run `make html` while in the `docs/` directory. This creates a `docs/build/` folder. You can then open the file `docs/build/html/index.html` in your favorite browser.

## Dependencies

The project is written for Python 3.4, with the following packages and versions:
* [Simpy](http://simpy.readthedocs.io/en/latest/) 3.0.10
* [Networkx](http://networkx.readthedocs.io/en/stable/) 1.11
* [Matplotlib](http://matplotlib.org/) 1.3.1, for its pyplot sub-package
* dill 0.2.5

Sphinx 1.5.3 is used to generate the documentation, along with the ReadTheDocs theme (v. 0.2.4)

## License

This software is licensed under the [CeCILL license](http://www.cecill.info), created by the Inria, CNRS, and CEA French research institutes. It is inspired from and compatible with the GNU GPL license.

Copyright (c) 2017 University of Rennes 1



