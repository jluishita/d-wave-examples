# -*- coding: utf-8 -*-
"""
 File name: hamiltonian-cycle-CQM.py
 
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

     In this file, the Constrained Quadratic Model sampler (CQM) is used.

Disclaimer: This code is an experimental, not-optimized, not-tested code.
     USE IT AT YOUR OWN RISK.

"""

import numpy as np
import pandas as pd

from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, quicksum

def main():

    # Define the number of nodes to visit
    n_nodes = 5
    # Defines the order in which each node is visited
    n_order = n_nodes

    # Defines the array of binary variables
    x = [Binary(order + node*n_order) for order in range(n_order) for node in range(n_nodes)]

    # Declares the bqm object
    cqm = ConstrainedQuadraticModel()

    # Define the constraint 1: Any node must be visited once and only once
    for node in range(n_nodes):
        cqm.add_constraint(quicksum(x[order + node*n_order] for order in range(n_order)) == 1, label='node_once_'+str(node))

    # Define the constraint 2: Any order must be selected once and only once
    for order in range(n_order):
        cqm.add_constraint(quicksum(x[order + node*n_order] for node in range(n_nodes)) == 1, label='order_once_'+str(order))

    # Declare the sample to be used and solve the problem
    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm, time_limit=10)

    # Gets rid of the unfeasible solutions and gets the best feasible solution
    feasible_sols = np.where(sampleset.record.is_feasible==True)
    solution = sampleset.record[feasible_sols[0][0]][0]

    # Reshapes the best feasible solution
    data = np.zeros((n_nodes, n_order))
    for order in range(n_order):
        for node in range(n_nodes):
            data[node, order] = solution[order + node*n_order]
            
    # Defines column and row names for the final solution
    columns = ['order_'+str(order) for order in range(n_order)]
    index = ['node_'+str(node) for node in range(n_nodes)]
    df_solution = pd.DataFrame(data=data, columns=columns, index=index)

    # Prints final solution
    print(df_solution)
    
	
if __name__=="__main__":
    main()
