#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:37:58 2024

@author: research
"""

import FastRDF as frdf
import numpy as np

# %% Prepare the coordination files from GROMACS
# In this case, simulation files correspond to a mixture of ethanol at 70%mol.
gro_file = 'example/sim.gro'
traj_file = 'example/sim.trr'

# %% Generate the residues dictionary
# This dictionary is used to identify the names and numbers of residues.
d_residues = frdf.parse_gro(gro_file)

# %% Compute the pairs of atoms
# Pairs of oxygen of water (SOL-OW) and hydrogen of water (SOL-HW1 and SOL-HW2).
pairs1a = frdf.gen_pairs('SOL-OW', 'SOL-HW1', d_residues)
pairs1a = pairs1a['inter']
pairs1b = frdf.gen_pairs('SOL-OW', 'SOL-HW2', d_residues)
pairs1b = pairs1b['inter']

pairs1 = np.concatenate((pairs1a, pairs1b), axis=0)

# Pairs of oxygen of water (SOL-OW) and hydrogen of hydroxil group (ETH-H08)
pairs2 = frdf.gen_pairs('SOL-OW', 'ETH-H08', d_residues)
pairs2 = pairs2['inter']

# %% Compute the RDF
rdf1 = frdf.compute_rdf(traj_file, gro_file, pairs1)
rdf2 = frdf.compute_rdf(traj_file, gro_file, pairs2)

# %% Plot the results

import matplotlib.pyplot as plt

plt.plot(rdf1[:,0], rdf1[:,1], label='OW-OW')
plt.plot(rdf2[:,0], rdf2[:,1], label='OW-H(O)')
plt.xlabel('r [nm]')
plt.ylabel('g(r)')
plt.title('RDF mixture Water-Ethanol')
plt.legend()

