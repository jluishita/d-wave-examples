# -*- coding: utf-8 -*-
"""
 File name: map-coloring-BQM.py
 
 File type: Python file
 
 Author: Jorge Luis Hita
 
 Institution: Private use
              
 Date: 26th January 2022

 Description: This program solves naïve instances of the Map Coloring 
     problem using D-Wave's Quantum Annealer.

     A Map Coloring problem is a problem of assigning colors to a set of 
     regions, subject to the restriction that adjacent regions can not be
     assigned to the same color.

     For planar graphs (maps) it was proven that the graph can be colored
     with four colors. Sometimes, three colors are enough. 

     In this file, the Binary Quadratic Model sampler (BQM) is used.

Disclaimer: This code is an experimental, not-optimized, not-tested code.
     USE IT AT YOUR OWN RISK.

"""

import numpy as np
import pandas as pd
from tabulate import tabulate

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

def test_unique_color(data, n_regions):
    if np.array_equal(np.ones(n_regions), data.sum(axis=1)):
        print(' The constraint 1 is fulfilled.')
    else:
        print(' Warning: the constraint 1 is NOT fulfilled')


def main():

    # Define the number of nodes to visit
    list_of_regions = ["Nouvelle Aquitaine", "Occitanie", "Auvergne Rhône-Alpes", "Bretagne", "Normandie", 
                       "Pays de la Loire", "Centre-Val de Loire", "Île-de-France", "Hauts-de-France",
                       "Grand-Est", "Bourgogne Franche-Comté", "Provence-Alpes-Côte dAzur"]
    # Defines the order in which each node is visited
    list_of_colors = ['red', 'blue', 'green', 'yellow']
    n_regions = len(list_of_regions)
    n_colors = len(list_of_colors)

    list_of_borders = {"Nouvelle Aquitaine": ["Occitanie", "Auvergne Rhône-Alpes", "Centre-Val de Loire", "Pays de la Loire"],
                       "Occitanie": ["Provence-Alpes-Côte dAzur", "Auvergne Rhône-Alpes", "Nouvelle Aquitaine"],
                       "Provence-Alpes-Côte dAzur": ["Auvergne Rhône-Alpes", "Occitanie"], 
                       "Auvergne Rhône-Alpes": ["Bourgogne Franche-Comté", "Centre-Val de Loire", "Nouvelle Aquitaine", 
                                                "Occitanie", "Provence-Alpes-Côte dAzur"],
                       "Bourgogne Franche-Comté": ["Grand-Est", "Île-de-France", "Centre-Val de Loire", "Auvergne Rhône-Alpes"],
                       "Centre-Val de Loire": ["Île-de-France", "Normandie", "Pays de la Loire", "Nouvelle Aquitaine", 
                                               "Auvergne Rhône-Alpes", "Bourgogne Franche-Comté"],
                       "Pays de la Loire": ["Nouvelle Aquitaine", "Centre-Val de Loire", "Normandie", "Bretagne"], 
                       "Bretagne": ["Pays de la Loire", "Normandie"],
                       "Normandie": ["Bretagne", "Pays de la Loire", "Centre-Val de Loire", "Île-de-France", "Hauts-de-France"],
                       "Île-de-France": ["Normandie", "Centre-Val de Loire", "Bourgogne Franche-Comté", 
                                         "Grand-Est", "Hauts-de-France"],
                       "Grand-Est": ["Hauts-de-France", "Île-de-France", "Bourgogne Franche-Comté"],
                       "Hauts-de-France": ["Normandie", "Île-de-France", "Grand-Est"]}
    # Defines the lagrange multipliers for the constraints
    lag_mul_1 = 1
    lag_mul_2 = 1

    # Defines the array of binary variables
    x = [[f'x_c_{region}_p_{color}' for color in range(n_colors)] for region in range(n_regions)]

    # Declares the bqm object
    bqm = BinaryQuadraticModel('BINARY')

    # Define the constraint 1: Any region must be assigned to one and only one color
    for region in range(n_regions):
        c_weights = [(x[region][color], 1) for color in range(n_colors)]
        bqm.add_linear_equality_constraint(c_weights, constant=-1, lagrange_multiplier=lag_mul_1)

    # Define the constraint 2: Any order must be selected once and only once
    visited = set()
    for region1 in list_of_regions:
        for region2 in list_of_borders[region1]:
            if (region1,region2) not in visited and (region2,region1) not in visited:
                visited.add((region1,region2))
                r1 = list_of_regions.index(region1)
                r2 = list_of_regions.index(region2)
                for color in range(n_colors):
                    bqm.add_quadratic(x[r1][color], x[r2][color], lag_mul_2)


    # Declare the sample to be used and solve the problem
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, num_reads=100)

    # Take best solution
    first_sample = sampleset.first.sample

    # Reshapes the best solution
    data = np.zeros((n_regions, n_colors))
    for color in range(n_colors):
        for region in range(n_regions):
            data[region, color] = first_sample[x[region][color]]

    test_unique_color(data, n_regions)

    list_of_colors_np = np.array(list_of_colors)
    color_to_region = list_of_colors_np[np.argmax(data, axis=1)]

    # Defines column and row names for the final solution
    columns = ['color']
    df_solution = pd.DataFrame(data=color_to_region, columns=columns, index=list_of_regions)

    # Prints final solution
    print(df_solution)
    
	
if __name__=="__main__":
    main()
