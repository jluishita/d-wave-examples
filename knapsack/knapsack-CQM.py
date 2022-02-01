# -*- coding: utf-8 -*-
"""
 File name: knapsack-CQM.py
 
 File type: Python file
 
 Author: Jorge Luis Hita
 
 Institution: Private use
              
 Date: 1st February 2022

 Description: This program solves naïve instances of the knapsack 
     problem.

     In a knapsack problem there is a set of objects. Each of them has a
     value and a weight. A knapsack problem is a problem of selecting a subset 
     of objects to maximize the total value of the selection, subject to the 
     restriction that the sum of weights must not exceed some upper value.

     In this file, the Constrained Quadratic Model sampler (CQM) is used.

 Disclaimer: This code is an experimental, not-optimized, not-tested code.
     USE IT AT YOUR OWN RISK.

"""

import numpy as np
import pandas as pd
from tabulate import tabulate

from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, Integer, quicksum

def test_unique_color(data, n_regions):
    """
    This function checks whether the first constraint is fulfilled
    """
    if np.array_equal(np.ones(n_regions), data.sum(axis=1)):
        print(' \nConstraint #1 is fulfilled.\n')
    else:
        print(' \nWarning: Constraint #1 is NOT fulfilled\.n')

def test_adjacent_regions(list_of_regions, color_to_region, list_of_borders):
    """
    This function checks whether the second constriant is fulfilled
    """
    for r1, region1 in enumerate(list_of_regions):
        for region2 in list_of_borders[region1]:
            r2 = list_of_regions.index(region2)
            if color_to_region[r1]==color_to_region[r2]:
                print(' \nWarning: Constraint #2 is NOT fulfilled.\n')
                return
    print(' \nConstraint #2 is fulfilled.\n')

def main():

    # Defines the objects
    list_of_objects = ["obj_1", "obj_2", "obj_3", "obj_4"]

    # Defines the number of elements for each object
    list_of_elements = [3, 5, 5, 4]

    # Defines the value for each object
    list_of_values = [1.3, 2.5, 0.8, 1.7]

    # Defines the nweights for each object
    list_of_weights = [2, 4, 1.8, 1.6]

    max_weight = 7.8

    # Number of colors and number of regions
    n_objects = len(list_of_objects)

    # Defines the array of binary variables
    x = [Integer(list_of_objects[object], upper_bound=list_of_elements[object]) for object in range(n_objects)]

    # Declares the cqm object
    cqm = ConstrainedQuadraticModel()

    cqm.set_objective(quicksum(-x[object]*list_of_values[object] for object in range(n_objects)))

    cqm.add_constraint(quicksum(x[object]*list_of_weights[object] for object in range(n_objects)) <= max_weight, label='max_weight')

    # Declare the sample to be used and solve the problem
    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm, time_limit=10)

    # Gets rid of the unfeasible solutions and gets the best feasible solution
    feasible_sols = np.where(sampleset.record.is_feasible==True)
    solution = sampleset.record[feasible_sols[0][0]][0]

    # Prints final solution
    print(solution)
    
	
if __name__=="__main__":
    main()
