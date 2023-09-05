# -*- coding: utf-8 -*-
"""
 File name: hamiltonian-cycle-QUBO.py
 
 File type: Python file
 
 Author: Jorge Luis Hita
 
 Institution: Private use
              
 Date: 25th January 2022

 Description: This program solves naïve instances of the Hamiltonian cycle 
     problem using D-Wave's Quantum Annealer.

     A Hamiltonian cycle problem is a problem of visiting every node in a 
     graph, subject to the restriction that every node should be visited once
     and only once.

     When the graph is complete, the solution is easier than in the general
     case. 

     In this file, the QUBO formalism is used.

Disclaimer: This code is an experimental, not-optimized, not-tested code.
     USE IT AT YOUR OWN RISK.

"""

import numpy as np
import pandas as pd

from dwave.system import DWaveSampler, EmbeddingComposite

def main():

    # Define the number of nodes to visit
    n_nodes = 5
    # Defines the order in which each node is visited
    n_order = n_nodes
    # Defines the lagrange multiplier for the constraints
    lag_mul = 1

    linear, quadratic = {}, {}

    # Define the constraint 1: Any node must be visited once and only once
    for node1 in range(n_nodes-1):
        for node2 in range(node1+1, n_nodes):
            for order in range(n_order):
                quadratic[(order + node1*n_order, order + node2*n_order)] = lag_mul

    # Define the constraint 1: Any node must be visited once and only once
    for node in range(n_nodes):
        for order1 in range(n_order-1):
            for order2 in range(order1+1, n_order):
                quadratic[(order1 + node*n_order, order2 + node*n_order)] = lag_mul

    # Define the constraint 1: Any node must be visited once and only once
    for node in range(n_nodes):
        for order in range(n_order):
            linear[(order + node*n_order, order + node*n_order)] = -lag_mul

    Q = {**linear, **quadratic}

    # Declare the sample to be used and solve the problem
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(Q, num_reads=100)

    # Take best solution
    first_sample = sampleset.first.sample

    # Reshapes the best solution
    data = np.zeros((n_nodes, n_order))
    for order in range(n_order):
        for node in range(n_nodes):
            data[node, order] = first_sample[x[node][order]]

    # Defines column and row names for the final solution
    columns = ['order_'+str(order) for order in range(n_order)]
    index = ['node_'+str(node) for node in range(n_nodes)]
    df_solution = pd.DataFrame(data=data, columns=columns, index=index)

    # Prints final solution
    print(df_solution)
    
	
if __name__=="__main__":
    main()
