#access object's parameters: (iclamp in example below)
#print([item for item in dir(iclamp) if not item.startswith('__')])

from neuron import h,gui
import csv
import pandas as pd
import numpy as np
import re
import random
import igraph as ig # for analysis of geodesic distances
from sklearn.neighbors import NearestNeighbors

# changed to 0.008x scaled down version from Jamie's file
h.load_file('local5_s.hoc')
# connections from VA6

cell1 = h.local5_s()

### re-mesh the segments (spatial discretization)
for sec in h.allsec(): 
	sec.nseg = sec.n3d()

# show attributes of a section:
#from pprint import pprint
#pprint(cell1.axon[1].psection())

### create table of all segments (0 to section.n3d()-1), x, y, z coordinates, 
### and their associated section number (re.split('\[|\]', cell1.axon[192].name())[3] gives 192)

### add parent section in too, so that we can create an igraph representation

tree = [] # tree is initially a list, dict to DataFrame is fastest to create the pandas DataFrame
for sec in cell1.axon:
	num_segs = sec.n3d()
	sec_index = re.split('\[|\]', sec.name())[3]
	for i in range(num_segs):
		toAppend = {} 	# each row to add is a dictionary
		loc = sec.arc3d(i) / sec.L
		geodesic_dist = h.distance(cell1.axon[0](0.5), sec(loc))
		toAppend.update(sec=sec_index, seg=i, 
						x=sec.x3d(i), y=sec.y3d(i), z=sec.z3d(i), 
						arc=sec.arc3d(i), gd = geodesic_dist)
		tree.append(toAppend)
tree = pd.DataFrame(tree)
# print(tree.head())
# returns 8361 total rows with 6854 unique x,y,z coordinates: np.unique(tree.loc[:, 'x':'z'], axis = 0).shape

###
### add VA6 synapses
###
### import synaptic locations
conn_va6 = pd.read_csv('conn_va6.csv')
num_synapses = conn_va6.shape[0]
print("imported " + str(num_synapses) + " VA6 synapses")

### KNN to map each synapse x, y, z (scaled x0.008) to the closest segment
print("KNN to map synapses onto morphology")
tree_coords = tree.loc[:, 'x':'z']
syn_coords = conn_va6.loc[:, 'x':'z'] / 125
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tree_coords)
distances, indices_va6 = nbrs.kneighbors(syn_coords) 
# indices: index of closest section location to a synapse

### add synapses onto morphology
syns = h.List()
j = 0 # index in syns
#i = 0 # track number of indexes we've looked at
num_in_dendr = 0
for index in indices_va6:
	sec = int(tree.loc[index, 'sec'])
	seg = tree.loc[index, 'seg']
	loc = cell1.axon[sec].arc3d(seg) / cell1.axon[sec].L 
	# 0 to 1, length along section
	syns.append(h.Exp2Syn(cell1.axon[sec](loc)))
	syns.object(j).tau1 = 0.2 #ms
	syns.object(j).tau2 = 1.1 #ms
	syns.object(j).e = 0 #mV, synaptic reversal potential = 0 mV for acetylcholine? 
	syns.object(j).g = 0.0001 #uS default
	if (cell1.axon[sec] in cell1.dendrites):
		num_in_dendr = num_in_dendr + 1
	j = j + 1
print("proportion VA6 synapses in dendrites: " + str(num_in_dendr / num_synapses))

### use NetStim to activate NetCon
nc = h.NetStim()
nc.number = 1
nc.start = 100
nc.noise = 0

### peak synaptic conductance, microS
syn_strength = 0.0001 

ncs = h.List()
for i in range(len(list(syns))):
	ncs.append(h.NetCon(nc, syns.object(i)))
	ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

###
### prune VA6 synapses
### if prune_worst = True, prune proximal synapses
### otherwise, prune randomly until # VA6 synapses = # VL2a synapses
###
def prune_VA6(prune_worst = False):
	random.seed(420)
	to_prune = random.sample(range(0, 101-1), (101-85)) 
	# prune most proximal synapses (largest impact on EPSP)
	to_prune_worst = [1, 2, 74, 79, 33, 32, 78, 13, 
					  9, 5, 57, 4, 34, 8, 20, 10, 11] 
	for i in range(len(list(syns))):
		if prune_worst:
			if i in to_prune_worst:
				ncs.object(i).weight[0] = 0
		else: 
			if i in to_prune: 
				ncs.object(i).weight[0] = 0
	print("finished pruning VA6 synapses")

def reset_VA6():
	for i in range(len(list(syns))):
		ncs.object(i).weight[0] = 0.0001 # uS, peak conductance change
	print("VA6 synapses restored")

###
### add VL2a synapses
###

conn_vl2a = pd.read_csv('conn_vl2a.csv')
num_synapses = conn_vl2a.shape[0]
print("imported " + str(num_synapses) + " VL2a synapses")

### KNN to map each synapse x, y, z (scaled x0.008) to the closest segment
print("KNN to map synapses onto morphology")
tree_coords = tree.loc[:, 'x':'z']
syn_coords = conn_vl2a.loc[:, 'x':'z'] / 125
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tree_coords)
distances, indices_vl2a = nbrs.kneighbors(syn_coords) 
# indices: index of closest section location to a synapse

### add synapses onto morphology
syns2 = h.List()
j = 0
num_in_dendr = 0
for index in indices_vl2a:
	sec = int(tree.loc[index, 'sec'])
	seg = tree.loc[index, 'seg']
	loc = cell1.axon[sec].arc3d(seg) / cell1.axon[sec].L 
	# 0 to 1, length along section
	syns2.append(h.Exp2Syn(cell1.axon[sec](loc)))
	syns2.object(j).tau1 = 0.2 #ms
	syns2.object(j).tau2 = 1.1 #ms
	syns2.object(j).e = 0 #mV, synaptic reversal potential 
	syns2.object(j).g = 0.0001 #uS default
	if (cell1.axon[sec] in cell1.dendrites):
		num_in_dendr = num_in_dendr + 1
	j = j + 1
print("proportion VL2a synapses in dendrites: " + str(num_in_dendr / num_synapses))

### use NetStim to activate NetCon, starts after VA6 NetStim
nc2 = h.NetStim()
nc2.number = 1
nc2.start = 300
nc2.noise = 0

ncs2 = h.List()
for i in range(len(list(syns2))):
	ncs2.append(h.NetCon(nc2, syns2.object(i)))
	ncs2.object(i).weight[0] = syn_strength # uS, peak conductance change

###
### relocating VL2a synapses based on geodesic distances
###

def plot_geodesic():
	gd_VA6 = []
	for i in range(len(list(syns))):
		gd_VA6.append(h.distance(cell1.axon[1](0.5), syns.object(i).get_segment()))
# for i in range(len(list(syns))): print(h.distance(cell1.axon[1](0.5), syns.object(i).get_segment()))

### simulation runs for 300 ms
Tstop = 400

### record voltage at or near soma, tune parameters to achieve ~6 mV EPSPs
### (VL2a has ~2.5 mV EPSPs)

#v = h.Vector().record(local5_s[0].axon[92](0.25)._ref_v) # soma membrane potential
#v = h.Vector().record(soma(0.5)._ref_v)             # Membrane potential vector
#t = h.Vector().record(h._ref_t)                     # Time stamp vector
