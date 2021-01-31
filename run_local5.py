### 2020-07-12
### working with retraced local5 neuron (with 2x as many VA6 as VL2a synapses)

#access object's parameters: (iclamp in example below)
#print([item for item in dir(iclamp) if not item.startswith('__')])

from neuron import h, gui
import csv
import pandas as pd
import numpy as np
import re
import random
import shelve # really useful for saving data from simulation runs
import igraph as ig # for analysis of geodesic distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt # plot geodesic histograms
from matplotlib.pyplot import draw
from statistics import mean, median
from neuron.units import ms, mV # allows us to specify units
plt.rcParams["font.family"] = "Arial"
#plt.rcParams.update({'font.size': 22})		

# set to 1 for local5v1, 2 for local5v2, 3 if importing the functions to another file
# (i.e. for running pop_mc_model.py)
# TODO: use an if __name__ == '__main__' to prevent the local5 code from being run upon import!
which_local5 = 3

if which_local5 == 1:
	# load the first local5 that I worked with (5813105722); 209 VA6 synapses, 108 VL2a synapses
	# local5_r_s = retraced, scaled, 12486 segments; seems better for more accurate synapse placement
	# local5_r_s_d = retraced, scaled, d_lambda rule for meshing, 1970 segments
	h.load_file('local5_r_s.hoc')
	cell1 = h.local5_r_s()
	# synapse location csv files, output from a quick filter in R: LH_input_dist_distrs.Rmd
	file_loc_va6 = 'conn_r_va6.csv'
	file_loc_vl2a = 'conn_r_vl2a.csv'
	# soma and putative spike initiation zone
	p_soma = cell1.axon[135]
	p_siz = cell1.axon[225]
	p_dendr = cell1.axon[800]
	p_axon = cell1.axon[423]
	loc_on_siz = 0.5
	# neuron ID:
	ID = str(5813105722)
elif which_local5 == 2:
	# load the second local5 instance (696126258); 195 VA6 synapses, 147 VL2a synapses
	# NOTE: after CellBuilder, had to use the PowerShell command to rename "soma" to "axon":
	# 
	h.load_file('local5v2_r_s.hoc')
	cell1 = h.local5v2_r_s()
	# synapse location csv files
	file_loc_va6 = 'conn_v2_r_va6.csv'
	file_loc_vl2a = 'conn_v2_r_vl2a.csv'
	# soma and putative spike initiation zone
	p_soma = cell1.axon[1]
	p_siz = cell1.axon[36]
	p_dendr = cell1.axon[824]
	p_axon = cell1.axon[788]
	loc_on_siz = 0.1
	# neuron ID:
	ID = str(696126258)

# changed to 0.008x scaled down version from Jamie's file

### biophysical parameters for local5
### currently set as the defaults in the hoc file
### 7/19: changed to lowest total error from parameter fit
syn_strength = 0.000035 # uS, peak synaptic conductance
R_a = 125 # ohm-cm
g_pas = 4.4e-5 # S/cm^2
e_pas = -55 # mV
R_m = 1/g_pas # ohm
c_m = 1.2 #uF/cm^2
param_print = False # whether to print param values when changing them

if which_local5 == 1 or which_local5 == 2:
	### re-mesh the segments (spatial discretization)
	for sec in h.allsec(): 
		sec.nseg = sec.n3d()

	# show attributes of a section:
	#from pprint import pprint
	#pprint(cell1.axon[1].psection())

	### create table of all specified 3d points (0 to section.n3d()-1), x, y, z coordinates, 
	### (note, 3d point != segment, but arc3d(point i)/section length does give "x position" (0 to 1) of point)
	### and their associated section number (re.split('\[|\]', cell1.axon[192].name())[3] gives 192)

	### add parent section in too, so that we can create an igraph representation

	tree = [] # tree is initially a list, dict to DataFrame is fastest to create the pandas DataFrame
	for sec in cell1.axon:
		num_segs = sec.n3d()
		sec_index = re.split('\[|\]', sec.name())[3]
		for i in range(num_segs):
			toAppend = {} 	# each row to add is a dictionary
			loc = sec.arc3d(i) / sec.L
			geodesic_dist = h.distance(p_soma(0.5), sec(loc))
			toAppend.update(sec=sec_index, seg=i, 
							x=sec.x3d(i), y=sec.y3d(i), z=sec.z3d(i), 
							arc=sec.arc3d(i), gd = geodesic_dist)
			tree.append(toAppend)
	tree = pd.DataFrame(tree)
	# print(tree.head())
	# returns 12486 total rows with 10527 unique x,y,z coordinates: np.unique(tree.loc[:, 'x':'z'], axis = 0).shape

	###
	### add VA6 synapses
	###
	### import synaptic locations
	### conn_va6 = stores information about VA6 synapses, including:
	### gd, neurite diameter, closest adjacent, size of mini_EPSP at soma, at SIZ, whether synapse is active
	conn_va6 = pd.read_csv(file_loc_va6)
	conn_va6 = conn_va6.drop(columns = ['type', 'partner'])
	num_synapses_va6 = conn_va6.shape[0]
	print("imported " + str(num_synapses_va6) + " VA6 synapses")

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
		seg = tree.loc[index, 'seg']	### note again, this seg is more properly thought of as 3d point index
		#print("adding synapse " + str(j) + " to section " + str(sec))
		loc = cell1.axon[sec].arc3d(seg) / cell1.axon[sec].L 
		# 0 to 1, length along section
		#print("about to append")
		syns.append(h.Exp2Syn(cell1.axon[sec](loc)))
		syns.object(j).tau1 = 0.2 #ms
		syns.object(j).tau2 = 1.1 #ms
		syns.object(j).e = -10 #mV, synaptic reversal potential = 0 mV for acetylcholine? 
		# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125135/
		#syns.object(j).g = 0.0001 #uS default

		h.pop_section() # clear the section stack to avoid overflow (triggered by using ".L" above?)

		if (cell1.axon[sec] in cell1.dendrites):
			num_in_dendr = num_in_dendr + 1
		j = j + 1
	print("proportion VA6 synapses in dendrites: " + str(num_in_dendr / num_synapses_va6))

	### use NetStim to activate NetCon
	nc = h.NetStim()
	nc.number = 1
	nc.start = 100
	nc.noise = 0

	ncs = h.List()
	for i in range(len(list(syns))):
		ncs.append(h.NetCon(nc, syns.object(i)))
		ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

	###
	### add VL2a synapses
	###

	conn_vl2a = pd.read_csv(file_loc_vl2a)
	conn_vl2a = conn_vl2a.drop(columns = ['type', 'partner'])
	num_synapses_vl2a = conn_vl2a.shape[0]
	print("imported " + str(num_synapses_vl2a) + " VL2a synapses")

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
		syns2.object(j).e = -10 #mV, synaptic reversal potential 
		#syns2.object(j).g = 0.0001 #uS default

		h.pop_section() # clear the section stack to avoid overflow (triggered by using ".L" above?)

		if (cell1.axon[sec] in cell1.dendrites):
			num_in_dendr = num_in_dendr + 1
		j = j + 1
	print("proportion VL2a synapses in dendrites: " + str(num_in_dendr / num_synapses_vl2a))

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
### find geodesic distances
###
def find_geodesic(synapses):
	gd = []
	for i in range(len(list(synapses))):
		if (synapses == syns):
			if (ncs.object(i).weight[0] != 0):
				gd.append(h.distance(p_soma(0.5), synapses.object(i).get_segment()))
		elif (synapses == syns2):
			if (ncs2.object(i).weight[0] != 0):
				gd.append(h.distance(p_soma(0.5), synapses.object(i).get_segment()))
		else:
			gd.append(h.distance(p_soma(0.5), synapses.object(i).get_segment()))
	return gd

if which_local5 == 1 or which_local5 == 2:
	gd_va6 = find_geodesic(syns) # track geodesic distances of each synapse
	gd_vl2a = find_geodesic(syns2)

# plot the current geodesic histograms
def plot_geodesic():
	gd_va6 = find_geodesic(syns) # track geodesic distances of each synapse
	gd_vl2a = find_geodesic(syns2)

	plt.rcParams["figure.figsize"] = (10,8)

	# %matplotlib inline 
	bins = np.linspace(100, 300, 40)
	plt.hist(gd_va6, bins, alpha = 0.5, label = 'VA6', color = 'r')
	plt.axvline(mean(gd_va6), color='r', linestyle='dashed', linewidth=1)

	plt.hist(gd_vl2a, bins, alpha = 0.5, label = 'VL2a', color = 'b')
	plt.axvline(mean(gd_vl2a), color='b', linestyle='dashed', linewidth=1)
	plt.legend(loc = 'upper right')
	plt.xlabel("geodesic distance from soma (um)")
	plt.ylabel("# synapses")
	plt.xlim(100, 300)
	plt.ylim(0, 40)
	plt.show()

###
### prune VA6 synapses
### if prune_worst = True, prune proximal synapses
### otherwise, prune randomly until # VA6 synapses = # VL2a synapses
###

random.seed(420)
if which_local5 == 1 or which_local5 == 2:
	num_to_prune = num_synapses_va6 - num_synapses_vl2a
else:
	num_to_prune = 0
	gd_va6 = None
def prune_VA6(number_to_prune = num_to_prune, geodesic = gd_va6, prune_worst = False):
	global gd_va6

	if prune_worst:
		# prune MOST PROXIMAL synapses (largest impact on EPSP)
		to_prune_threshold = sorted(geodesic)[number_to_prune-1]
		to_prune_worst = [i for i in range(len(geodesic)) if geodesic[i] <= to_prune_threshold]
	else:
		to_prune = random.sample(range(0, num_synapses_va6-1), (number_to_prune))
	
	#VA6 synapses on axon = [98, 99, 165, 167, 179, 181], 179 and 181 are on the same section

	# prune by setting connection weight to 0
	for i in range(len(list(syns))):
		if prune_worst:
			if i in to_prune_worst:
				ncs.object(i).weight[0] = 0
		else: 
			if i in to_prune: 
				ncs.object(i).weight[0] = 0
	print("finished pruning " + str(number_to_prune) + " VA6 synapses")

	# recalculate VA6 geodesic distances:
	if prune_worst:
		gd_va6 = [val for i, val in enumerate(gd_va6) if i not in to_prune_worst]
		return to_prune_worst
	else:
		gd_va6 = [val for i, val in enumerate(gd_va6) if i not in to_prune]
		return to_prune

def reset_VA6():
	global gd_va6
	for i in range(len(list(syns))):
		ncs.object(i).weight[0] = syn_strength # uS, peak conductance change
	gd_va6 = find_geodesic(syns)
	print("VA6 synapses restored")

###
### find surface area of entire neuron (um^2)
### see NEURON help forum for reference
###
def surface_area():
	h('_sum=0')
	h('forall for (x,0) _sum += area(x)')
	area = h('_sum')
	return area

###
### remove a section and rejoin its parents and children
### default truncate axon initial segment (axon[231])
### 225 (1.0 um) (pSIZ)->227 (0.4 um) ->231 (61.9 um, 105 um^2) ->423 (3.9 um), 232 (2.7 um)
### 225 and 227 are 1 and 0.4 um collectively
###
def remove_section(sec_num):
	global gd_va6, gd_vl2a

	if which_local5 == 1:
		sec_num = 231
	elif which_local5 == 2:
		sec_num = 36

	par = cell1.axon[sec_num].parentseg()
	chil = cell1.axon[sec_num].children()
	h.delete_section(sec = cell1.axon[sec_num])
	# reconnect children to parent
	for c in chil:
		c.connect(par, 0)
	# recalculate geodesic distances that may have moved
	gd_va6 = find_geodesic(syns) # track geodesic distances of each synapse
	gd_vl2a = find_geodesic(syns2)

	#### this gd_va6 adds back the pruned synapses!!!

### remove sections downstream with overall area = area of specified section
def remove_equiv_area_downstream(sec_num = 231, reverse = False):
	global gd_va6, gd_vl2a

	target_area = 0
	for seg in cell1.axon[sec_num].allseg():
		target_area += seg.area()
	print("target_area=" + str(target_area))

	deleted_area = 0
	chil = cell1.axon[sec_num].children()
	if reverse:
		chil = reversed(chil)
	for c in chil:
		downstream = c.subtree()
		for sec in downstream:
			can_remove = True
			sec_area = 0
			sec_index = re.split('\[|\]', sec.name())[3]
			for seg in sec:
				sec_area += seg.area()
				# if there are NO point processes, can consider deleting
				if seg.point_processes(): # if there ARE point processes...
					can_remove = False	  # ...don't remove
			# if deleted section area won't exceed target, and no point processes...
			if ((sec_area + deleted_area < target_area) & can_remove):
				# ...delete section!
				remove_section(sec_num = int(sec_index))
				deleted_area += sec_area
				print("removed " + str(sec_index) + ", area=" + str(sec_area))
				print("deleted_area=" + str(deleted_area))

	gd_va6 = find_geodesic(syns) # track geodesic distances of each synapse
	gd_vl2a = find_geodesic(syns2)


### remove the entire tree downstream of specified section
### TODO: something pathological happens when a section with a point process is removed...
def remove_downstream(sec_num = 231):
	for sec in cell1.axon[sec_num].subtree(): 
		h.delete_section(sec = sec)

###
### lengthen section
### default first section of dendrite initial segment (axon[796])
### add_length scaled to equalize VA6 and VL2a gd means
def lengthen_section(sec_num = 796, add_length = 14.1):
	global gd_va6

	curr_nseg = cell1.axon[sec_num].nseg
	curr_length = cell1.axon[sec_num].L
	cell1.axon[sec_num].L = curr_length + np.abs(add_length)
	cell1.axon[sec_num].nseg = curr_nseg *  int(np.floor(cell1.axon[sec_num].L/curr_length))		
	### hard-coded to new length / old length
	gd_va6 = find_geodesic(syns)
'''
practice on a random branch of the axon
cell1.axon[785].parentseg()
cell1.axon[785].children()
h.delete_section(sec = cell1.axon[785])
# reconnect the children to the original parent
cell1.axon[787].connect(cell1.axon[783](1), 0)
cell1.axon[786].connect(cell1.axon[783](1), 0)
# downstream tree and synapses automatically move!
'''

###
### adjacency matrix of synapse locations
### entry i, j is geodesic distance between synapses i, j
###
def find_adj(synapses):
	num_syns = len(list(synapses))
	adj = np.zeros((num_syns, num_syns))
	for i in range(0, num_syns):
		for j in range(i, num_syns):
			adj[i][j] = h.distance(synapses.object(i).get_segment(), synapses.object(j).get_segment())
			adj[j][i] = adj[i][j]
	return adj

def plot_clustering(syn1, syn2 = None):
	plt.rcParams["figure.figsize"] = (20,15)

	adj = find_adj(syn1)

	avg_adj = [0] * len(adj)
	for i in range(len(adj)): 
		avg_adj += np.sort(adj[i,:])
		plt.plot(np.sort(adj[i,:]), color = '#ffcccb', alpha = 0.4)
	avg_adj1 = avg_adj / len(adj)
	
	if syn2 != None:
		adj = find_adj(syn2)

		avg_adj = [0] * len(adj)
		for i in range(len(adj)): 
			avg_adj += np.sort(adj[i,:])
			plt.plot(np.sort(adj[i,:]), color = '#add8e6', alpha = 0.4)
		avg_adj = avg_adj / len(adj)
		plt.plot(avg_adj, color = 'blue', lw = 2.5, label = "VL2a")

	plt.plot(avg_adj1, color = 'red', lw = 2.5, label = "VA6")
	plt.xlabel("n'th closest synapse")
	plt.ylabel("geodesic distance away (um)")
	plt.legend(loc = "upper right")

	plt.show()


# plot adjacency matrix on block diagonal (by sorting first row)
# plt.matshow(adj_va6[np.ix_(np.argsort(adj_va6[1,:]),np.argsort(adj_va6[1,:]))], cmap = plt.cm.viridis.reversed())
# plt.matshow(adj_vl2a[np.ix_(np.argsort(adj_vl2a[1,:]),np.argsort(adj_vl2a[1,:]))], cmap = plt.cm.viridis.reversed())
# plt.clim(0, 150)
# plt.colorbar()

###
### change params:
### max synaptic conductance (across all synapses)
### global Ra (axial resistance)
### membrane conductance / membrane resistance
### membrane capacitance
###
def change_syn_strength(strength):
	global syn_strength
	for i in range(len(list(syns))):
		ncs.object(i).weight[0] = strength # uS, peak conductance change
	for i in range(len(list(syns2))):
		ncs2.object(i).weight[0] = strength # uS, peak conductance change
	syn_strength = strength
	if param_print:
		print("new synapse conductance: " + str(strength) + " uS")

def change_R_a(ra):
	global R_a
	for sec in h.allsec():
		sec.Ra = ra
	R_a = ra
	if param_print:
		print("new R_a: " + str(ra) + " ohm-cm")

def change_g_pas(gpas):
	global g_pas, R_m
	for sec in h.allsec():
		for seg in sec.allseg():
			if hasattr(seg, 'pas'):
				seg.pas.g = gpas
	g_pas = gpas
	R_m = 1/gpas
	if param_print:
		print("new g_pas: " + str(gpas) + " S/cm^2, new R_m: " + str(round(R_m/1000,2)) + " kOhm-cm^2")

def change_e_pas(epas):
	global e_pas

	for sec in h.allsec():
		for seg in sec.allseg():
			if hasattr(seg, 'pas'):
				seg.pas.e = epas
	e_pas = epas
	if param_print:
		print("new e_pas: " + str(epas) + " mV")


def change_c_m(cm):
	global c_m
	for sec in h.allsec():
		sec.cm = cm
	c_m = cm
	if param_print:
		print("new c_m: "+ str(cm) + " uF/cm^2")

if which_local5 == 1 or which_local5 == 2:
	###
	### given the current global parameter state, trigger VA6 and VL2a synapses
	### return voltage traces measured at the soma and putative spike initation zone
	### we'll record from record_from_seg input, which by default is the "soma"
	###
	def run_VA6_VL2a(record_from_seg = p_soma(0.5), siz_loc = loc_on_siz):

		t_traces = []
		v_soma = []
		v_siz = []

		for i in range(0,2):
			if (i==0):
				# run VA6 first
				nc.number = 1
				nc.start = 0
				nc2.number = 0
				nc2.start = 0
			elif (i==1):
				# run VL2a second
				nc.number = 0
				nc.start = 0
				nc2.number = 1
				nc2.start = 0
			
			h.load_file('stdrun.hoc')
			x = h.cvode.active(True)
			v_s = h.Vector().record(record_from_seg._ref_v) 		# soma membrane potential
			v_z = h.Vector().record(p_siz(siz_loc)._ref_v)		# putative SIZ membrane potential
			t = h.Vector().record(h._ref_t)                     # Time stamp vector
			h.finitialize(-55 * mV)
			h.continuerun(60*ms)

			t_traces.append(list(t))
			v_soma.append(list(v_s))
			v_siz.append(list(v_z))
		
		return t_traces, v_soma, v_siz

	###
	### original use: unitary VA6 EPSP, record at VA6 (input_sites) and VL2a (record_sites) synapse locs
	###
	def uEPSP_record_locs(stimulate = 'VA6', input_sites = syns, record_sites = syns2):

		# prepare VA6 synapses for activation
		if stimulate=='VA6':
			nc.number = 1
			nc.start = 0
			nc2.number = 0
			nc2.start = 0
			
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)

		# establish recordings at all VA6 synapses
		v_input = []
		for loc in input_sites:
			record_loc = loc.get_segment()
			v_input.append(h.Vector().record(record_loc._ref_v))

		# establish recordings at all VL2a synapses
		v_record = []
		for loc in record_sites:
			record_loc = loc.get_segment()
			v_record.append(h.Vector().record(record_loc._ref_v))

		v_soma = h.Vector().record(p_soma(0.5)._ref_v) 		# soma membrane potential
		t = h.Vector().record(h._ref_t)                     # Time stamp vector

		h.finitialize(-55 * mV)
		h.continuerun(60*ms)
		
		return t, v_input, v_record, v_soma


###
### general purpose plotting function for basic one run VA6 and VL2a traces
###
def plot_run(titletext, t_sim, v_soma_sim, v_siz_sim):
	plt.rcParams["figure.figsize"] = (20,15)

	fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = False)
	fig.subplots_adjust(top=0.9)

	for i in range(0, 1):
		if (i==0):
			LS = '--' #dashed
			LW = 2.0   #line as wide as the experimental trace
		elif (i==1):
			LS = '-.' #dash dotted
			LW = 1.4
		elif (i==2):
			LS = (0, (3, 10, 1, 10, 1, 10)) #loosely dashdotdotted
			LW = 1.3
		elif (i==len(lowest_errors)-1):
			LS = (0, (1, 10)) #loosely dotted
			LW = 1
		else:
			LS = ':' #dotted
			LW = 1.4

		index = i

		#label = ("error = " + str(round(sim_params.loc[index]["error_t"],3)) + ", " +
		#		 "g_pas = " + str(round(sim_params.loc[index]["g_pas"], 6)) + ", " + 
		#		 "g_syn = " + str(round(sim_params.loc[index]["g_syn"]*1000, 3)) + ", " + 
		#		 "c_m = " + str(round(sim_params.loc[index]["c_m"], 2)) + ", " + 
		#		 "R_a = " + str(round(sim_params.loc[index]["R_a"])))

		# plot VA6
		ax1.plot(t_sim[0], v_soma_sim[0], label = "soma", color='r', ls = LS, lw = LW)
		ax1.plot(t_sim[0], v_siz_sim[0], label = "siz", color='#ffa500', ls = LS, lw = LW)

		# plot VL2a
		ax2.plot(t_sim[1], v_soma_sim[1], label = "soma", color='b', ls = LS, lw = LW)
		ax2.plot(t_sim[1], v_siz_sim[1], label = "siz", color='#ffa500', ls = LS, lw = LW)

	### 
	### read experimental traces from MATLAB export
	### trace from Jamie's edits, 2020-07-15
	###
	ax1.plot(t_va6_tofit, v_va6_tofit, label = "VA6 experimental trace", color = 'r', linewidth=2)
	ax2.plot(t_vl2a_tofit, v_vl2a_tofit, label = "VL2a experimental trace", color = 'b', linewidth=2)

	ax1.legend(loc = "upper right", fontsize = '16')
	ax1.set_xlim(-5, 60)
	ax1.set_xticks(ticks=np.arange(-5, 60, step=5))
	ax1.set_ylim(-55, -40)
	ax1.set_yticks(ticks=np.arange(-55, -40, step=2))
	ax1.set_ylabel("membrane potential (mV)", fontsize = '20')
	ax1.set_xlabel("time (ms), VA6 activates at 0 ms")
	ax1.set_title("VA6 fit")

	ax2.legend(loc = "upper right", fontsize = '16')
	ax2.set_xlim(-5, 60)
	ax2.set_xticks(ticks=np.arange(-5, 60, step=5))
	ax2.set_ylim(-55, -40)
	ax2.set_yticks(ticks=np.arange(-55, -40, step=2))
	ax2.set_xlabel("time (ms), VL2a activates at 0 ms")
	ax2.set_title("VL2a fit")

	props = ("Params.: g_pas (S/cm^2) = " + str(round(g_pas, 6)) + 
			 ", g_syn (nS) = " + str(round(syn_strength*1000, 3)) + 
			 ", c_m (uF/cm^2) = " + str(round(c_m, 2)) +
			 ", R_a (Ohm-cm) = " + str(round(R_a)))

	plt.suptitle(titletext + props + " [local5 (" + ID + ")]", 
				 y = 0.96)

	plt.show()

### 
### baseline run: simply plot VA6 and VL2a amplitudes against experiment
###
def baseline_run():
	t_traces, v_soma, v_siz = run_VA6_VL2a()
	plot_run(titletext = "Baseline run, ", t_sim = t_traces, 
			 v_soma_sim = v_soma, v_siz_sim = v_siz)

###
### t1, v1 the simulated trace, t2, v2 the experimental trace to fit to
### weight error from VL2a x3 to normalize amplitudes
###
def find_error(t1, v1, t2, v2):
	early_t = np.linspace(1,20,20) # "early" 20 ms, experimental trace most accurate
	later_t = np.linspace(21,30,10) # "later" 20-30 ms, experimental trace less accurate

	early_v = np.interp(early_t, t1, v1)
	early_v_e = np.interp(early_t, t2, v2) # experimental

	later_v = np.interp(later_t, t1, v1)
	later_v_e = np.interp(later_t, t2, v2)

	# sqrt of sum of squares of differences:
	early_trace_diff = np.mean(np.abs(early_v - early_v_e))
	later_trace_diff = np.mean(np.abs(later_v - later_v_e))

	# weight later error 25% as important as early error
	return early_trace_diff + later_trace_diff/4

###
### doesn't work currently
### prevent change param functions from printing
### https://stackoverflow.com/questions/8447185/to-prevent-a-function-from-printing-in-the-batch-console-in-python
### 
def silent(fn):
    """Decorator to silence functions."""
    def silent_fn(*args, **kwargs):
        saved_stdout = sys.stdout
        sys.stdout = NullIO()
        result = fn(*args, **kwargs)
        sys.stdout = saved_stdout
        return result
    return silent_fn

###
### shelve simulated run data
###
def shelve_data(filename, toshelve = ['t_sim', 'v_soma_sim', 'v_siz_sim', 'sim_params']):
	shelf = shelve.open(filename, 'n')
	for key in dir():
		try: 
			if key in toshelve: 
				shelf[key] = globals()[key]
		except TypeError:
			#
        	# __builtins__, my_shelf, and imported modules can not be shelved.
        	#
			print('ERROR shelving: {0}'.format(key))
	shelf.close()

def load_shelf(shelfname):
	# shelve2.out: first big param search from 7/16/2020
	my_shelf = shelve.open(shelfname)
	for key in my_shelf:
		globals()[key]=my_shelf[key]
	my_shelf.close()

###
### big automated parameter search
###
### track: g_syn (round to 3), g_pas, R_a, c_m
### size of pSIZ EPSP
### error of VA6, VL2a
### location in v_soma, v_siz, t_traces of the corresponding traces
#def param_search():

if which_local5 == 1 or which_local5 == 2:
	# load experimental traces to fit to
	trace_va6 = pd.read_csv('va6_traceUpdate.csv', header = None, dtype = np.float64)
	trace_vl2a = pd.read_csv('vl2a_traceUpdate.csv', header = None, dtype = np.float64)

	t_va6_tofit = trace_va6[0]+1.25 # slightly adjust VA6 to align with rise time of EPSP
	v_va6_tofit = trace_va6[1]-55
	t_vl2a_tofit = trace_vl2a[0]+1.5 # note VL2a is delayed 1-3 s relative to VA6 -Jamie
	v_vl2a_tofit = trace_vl2a[1]-55

def param_search():
	'''
	# "big": search ranges 7 x 9 x 11 x 11 = 8316 parameter combinations
	# previously 12,672 parameter combinations
	# np.arange(start, stop, step_size)
	g_syn_s = np.arange(0.00002, 0.00009, 0.00001) # uS, i.e. 0.03 to 0.07 nS in 0.005 nS increments
	c_m_s = np.arange(0.7, 1.51, 0.1) # uF/cm^2
	g_pas_s = np.arange(1.2e-5, 5.4e-5, 0.4e-5) # S/cm^2, round to 6 places
	R_a_s = np.arange(50, 601, 50) # ohm-cm
	'''
	# "big": search ranges 7 x 9 x 11 x 11 = 16,632 parameter combinations
	# previously 12,672 parameter combinations
	# np.arange(start, stop, step_size)
	g_syn_s = np.arange(0.0000275, 0.000041, 0.0000025) # uS, i.e. 0.03 to 0.07 nS in 0.005 nS increments
	c_m_s = np.arange(0.6, 1.61, 0.1) # uF/cm^2
	g_pas_s = np.arange(1.2e-5, 5.4e-5, 0.2e-5) # S/cm^2, round to 6 places
	R_a_s = np.arange(50, 350, 25) # ohm-cm

	'''
	# "focused": search range: (3 x 9 x 21 x 14)
	g_syn_s = np.arange(0.000025, 0.000036, 0.000005) # uS, i.e. 0.025 to 0.035
	# include 0.035 bc it's like a depressed ORN-PN synapse; include 0.025 bc error might keep dropping 
	c_m_s = np.arange(0.7, 1.51, 0.1) # uF/cm^2 -- expanding the range of capacitance values; 
	# Gouwens models three different DM1 PNs, and finds c_m values between 0.8 and 2.6 uF/cm^2
	g_pas_s = np.arange(1.2e-5, 5.4e-5, 0.2e-5) # S/cm^2, round to 6 places
	R_a_s = np.arange(50, 400, 25) # ohm-cm; in Gouwens, fits with R_a above 800 ohm-cm were discarded
	'''
	# easy search ranges: (~24 parameter sets)
	'''
	g_syn_s = np.arange(0.00003, 0.00007, 0.00002) # uS, i.e. 0.03 to 0.07 nS (round to 6 places)
	c_m_s = np.arange(0.6, 0.8, 0.2) # uF/cm^2
	g_pas_s = np.arange(2.0e-5, 4.4e-5, 0.8e-5) # S/cm^2, round to 6 places
	R_a_s = np.arange(200, 1000, 400) # ohm-cm
	'''
	# 
	sim_params = []

	# length = # of simulation runs, for each element, [0]: VA6 trace, [1]: VL2a trace
	t_sim = []			
	v_soma_sim = []
	v_siz_sim = []

	for g_syn_i in g_syn_s:
		change_syn_strength(g_syn_i)
		for c_m_i in c_m_s:
			change_c_m(c_m_i)
			for g_pas_i in g_pas_s:
				change_g_pas(g_pas_i)
				for R_a_i in R_a_s:
					change_R_a(R_a_i)

					# run VA6 and VL2a, append output traces to v_soma_sim, v_siz_sim
					curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
					t_sim.append(curr_t)
					v_soma_sim.append(curr_v_soma)
					v_siz_sim.append(curr_v_siz)
					# calculate fit error of these two traces
					error_va6 = find_error(t1 = curr_t[0], v1 = curr_v_soma[0], 
										   t2 = t_va6_tofit, v2 = v_va6_tofit)
					error_vl2a = find_error(t1 = curr_t[1], v1 = curr_v_soma[1], 
										    t2 = t_vl2a_tofit, v2 = v_vl2a_tofit)
					# weight VL2a error x3 to normalize for amplitude (6.12 vs 2.04) differences
					error_total = error_va6 + error_vl2a * 3

					# save parameter values, output trace indices, fit errors
					params_toAppend = {}
					params_toAppend.update(g_syn = g_syn_i, c_m = c_m_i, 
									g_pas = g_pas_i, R_a = R_a_i, index_loc = len(t_sim)-1,
									error_t = error_total, 
									error_va6 = error_va6, error_vl2a = error_vl2a)

					sim_params.append(params_toAppend)
				print("finished running " + str(str(round(g_pas_i, 6))) + " S/cm^2")
		print("g_syn: finished with " + str(str(round(g_syn_i*1000, 4))) + " nS")

	sim_params = pd.DataFrame(sim_params)

	return sim_params, t_sim, v_soma_sim, v_siz_sim
	
#load_shelf() # load sim_params, voltage and time traces
# sim_params, t_sim, v_soma_sim, v_siz_sim = param_search()

###
### code to help find best simulation runs after param_search()
###
'''
# calculate the top 6 (lowest fit errors) simulations, plot them
lowest_errors = np.argsort(sim_params["error_t"])[0:5]
# investigate the fits when g_syn = 0.065 nS
high_g_syn = sim_params[(sim_params["g_syn"]<0.000066) & (sim_params["g_syn"]>0.000064)]
high_g_syn = high_g_syn.reset_index() # reset index to aid in next sorting step
lowest_errors_high_g = np.argsort(high_g_syn["error_t"])[0:5]
# convert to indices of the sim_params table
lowest_errors_high_g = high_g_syn.loc[lowest_errors_high_g]["index_loc"]
# investigate low g_pas (closer to experimental values)
low_g_pas = sim_params[(sim_params["g_pas"]<0.000025)]
'''

###
### plot traces from the parameter search superimposed over experimental traces
### the simulation runs to plot are indicated by indices, i.e. we could plot the 
### sim runs with the lowest errors
###
def plot_param_search_top_traces(indices, ylim_va6 = -40, ylim_vl2a = -50, 
				titletext = "Top 5 parameter fit hits, "):

	plt.rcParams["figure.figsize"] = (27,17)

	fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = False)
	fig.subplots_adjust(top=0.9)

	#plt.hold(True)

	for i in range(0, len(indices)):
		if (i==0):
			LS = '--' #dashed
			LW = 2.0   #line as wide as the experimental trace
		elif (i==1):
			LS = '-.' #dash dotted
			LW = 1.4
		elif (i==2):
			LS = (0, (3, 10, 1, 10, 1, 10)) #loosely dashdotdotted
			LW = 1.3
		elif (i==len(lowest_errors)-1):
			LS = (0, (1, 10)) #loosely dotted
			LW = 1
		else:
			LS = ':' #dotted
			LW = 1.4

		#index = lowest_errors[i]
		index = indices.iloc[i]

		label = ("error = " + str(round(sim_params.loc[index]["error_t"],3)) + ", " +
				 "g_pas = " + str(round(sim_params.loc[index]["g_pas"], 6)) + ", " + 
				 "g_syn = " + str(round(sim_params.loc[index]["g_syn"]*1000, 3)) + ", " + 
				 "c_m = " + str(round(sim_params.loc[index]["c_m"], 2)) + ", " + 
				 "R_a = " + str(round(sim_params.loc[index]["R_a"])))

		# plot VA6
		ax1.plot(t_sim[index][0], v_soma_sim[index][0], label = "soma: " + label, color='r', ls = LS, lw = LW)
		ax1.plot(t_sim[index][0], v_siz_sim[index][0], color='#ffa500', ls = LS, lw = LW)

		# plot VL2a
		ax2.plot(t_sim[index][1], v_soma_sim[index][1], label = "soma: " + label, color='b', ls = LS, lw = LW)
		ax2.plot(t_sim[index][1], v_siz_sim[index][1], color='#ffa500', ls = LS, lw = LW)

	### 
	### read experimental traces from MATLAB export
	### trace from Jamie's edits, 2020-07-15
	###
	ax1.plot(t_va6_tofit, v_va6_tofit, label = "VA6 experimental trace", color = 'r', linewidth=2)
	ax2.plot(t_vl2a_tofit, v_vl2a_tofit, label = "VL2a experimental trace", color = 'b', linewidth=2)

	ax1.legend(loc = "upper right", fontsize = '16')
	ax1.set_xlim(-5, 60)
	ax1.set_xticks(ticks=np.arange(-5, 60, step=5))
	ax1.set_ylim(-55, ylim_va6)
	ax1.set_yticks(ticks=np.arange(-55, ylim_va6, step=2))
	ax1.set_ylabel("membrane potential (mV)", fontsize = '20')
	ax1.set_xlabel("time (ms), VA6 activates at 0 ms")
	ax1.set_title("VA6 fit")

	ax2.legend(loc = "upper right", fontsize = '16')
	ax2.set_xlim(-5, 60)
	ax2.set_xticks(ticks=np.arange(-5, 60, step=5))
	ax2.set_ylim(-55, ylim_vl2a)
	ax2.set_yticks(ticks=np.arange(-55, ylim_vl2a, step=0.5))
	ax2.set_xlabel("time (ms), VL2a activates at 0 ms")
	ax2.set_title("VL2a fit")

	props = ("Units: g_pas (S/cm^2), g_syn (nS), c_m (uF/cm^2), R_a (Ohm-cm)")
	plt.suptitle(titletext + props + " [local5 (" + ID + ")]", 
				 fontsize = 24, y = 0.96)

	draw()

	fig.savefig('top_5_hits.svg')
	#fig.savefig('top_5_hits.eps', format='eps')

	plt.show()

###
### once the param search is done, this plots run errors against 
### specific biophysical parameters
###
def plot_error_vs_params(error = "error_t"):
	fig, axs = plt.subplots(nrows = 2, ncols = 2)

	axs[0,0].scatter(sim_params["g_syn"], sim_params[error])
	axs[0,0].set_xlabel("g_syn (uS)")
	axs[0,0].set_ylabel("error")

	axs[0,1].scatter(sim_params["g_pas"], sim_params[error])
	axs[0,1].set_xlabel("g_pas (S/cm^2))")
	axs[0,1].set_ylabel("error")	

	axs[1,0].scatter(sim_params["c_m"], sim_params[error])
	axs[1,0].set_xlabel("c_m (uF/cm^2)")
	axs[1,0].set_ylabel("error")	

	axs[1,1].scatter(sim_params["R_a"], sim_params[error])
	axs[1,1].set_xlabel("R_a (ohm-cm)")
	axs[1,1].set_ylabel("error")	

	plt.suptitle("Search of " + str(len(sim_params)) + " parameter sets: " + error + " vs parameter values")
	plt.show()

###
### the initial implementation of an automated simulation run
### example manipulation: change R_a between each run, plot all results
###
def run_tests():

	t_traces = []
	v_soma = []
	v_siz = []
	
	# track the manipulations made between each run
	trace_labs = []

	for j in range(0,5):

		new_Ra = (j+1) * 266.1 #ohm-cm
		change_R_a(new_Ra)

		for i in range(0,2):
			if (i==0):
				# run VA6 first
				nc.number = 1
				nc.start = 0
				nc2.number = 0
				nc2.start = 0
			elif (i==1):
				# run VL2a second
				nc.number = 0
				nc.start = 0
				nc2.number = 1
				nc2.start = 0
			
			h.load_file('stdrun.hoc')
			x = h.cvode.active(True)
			v_s = h.Vector().record(p_soma(0.5)._ref_v) 		# soma membrane potential
			v_z = h.Vector().record(p_siz(0.5)._ref_v)		# putative SIZ membrane potential
			t = h.Vector().record(h._ref_t)                     # Time stamp vector
			h.finitialize(-55 * mV)
			h.continuerun(100*ms)

			t_traces.append(list(t))
			v_soma.append(list(v_s))
			v_siz.append(list(v_z))

			trace_labs.append("R_a = " + str(round(new_Ra)))

	#change_R_a(1300) # wtffffff somehow making this too low breaks the simulation

	fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = False)
	fig.subplots_adjust(top=0.9)

	plt.rcParams["figure.figsize"] = (10,12)
	
	#plt.hold(True)

	for i in range(0, len(t_traces), 2):
		if (i==0):
			LS = ':'
		elif (i==len(t_traces)-2):
			LS = '-.'
		else:
			LS = '--'

		ax1.plot(t_traces[i], v_soma[i], label = "soma: " + trace_labs[i], color='r', ls = LS)
		ax1.plot(t_traces[i], v_siz[i], label = "pSIZ: " + trace_labs[i], color='#ffa500', ls = LS)

		ax2.plot(t_traces[i+1], v_soma[i+1], label = "soma: " + trace_labs[i], color='b', ls = LS)
		ax2.plot(t_traces[i+1], v_siz[i+1], label = "pSIZ: " + trace_labs[i], color='#ffa500', ls = LS)
	
	### 
	### read experimental traces from MATLAB export
	### trace from Jamie's edits, 2020-07-15
	###
	ax1.plot(t_va6_tofit, v_va6_tofit, label = "VA6 trace", color = 'r', linewidth=2)
	ax2.plot(t_vl2a_tofit, v_vl2a_tofit, label = "VL2a trace", color = 'b', linewidth=2)

	ax1.legend(loc = "upper right")
	ax1.set_xlim(-10, 60)
	ax1.set_xticks(ticks=np.arange(-10, 60, step=10))
	ax1.set_ylim(-55, -30)
	ax1.set_yticks(ticks=np.arange(-55, -30, step=2))
	ax1.set_ylabel("membrane potential (mV)", fontsize = '20')
	ax1.set_xlabel("time (ms), VA6 activates at 0 ms")
	ax1.set_title("VA6 fit")

	ax2.legend(loc = "upper right")
	ax2.set_xlim(-10, 60)
	ax2.set_xticks(ticks=np.arange(-10, 60, step=10))
	ax2.set_ylim(-55, -30)
	ax2.set_yticks(ticks=np.arange(-55, -30, step=2))
	ax2.set_xlabel("time (ms), VL2a activates at 0 ms")
	ax2.set_title("VL2a fit")

	props = ("g_pas = " + str(g_pas) + " S/cm^2, g_syn = " + str(syn_strength*1000) + 
			" nS, c_m = " + str(c_m) + " uF/cm^2, R_a = " + str(R_a) + 
			" Ohm-cm")
	plt.suptitle(props + " [local5 (" + ID + ") current params]", 
				 fontsize = 24, y = 0.96)

	# save figure
	#manager = plt.get_current_fig_manager()
	#manager.window.showMaximized()
	#plt.savefig(props + ".png")

	plt.show()

###
### LEGACY CODE:
### 

'''
20-07-14: plotting the experimental traces against lower synaptic conductance
(to fit passive SIZ size)
t_traces = []
v_soma = []
v_siz = []
for i in range(0,1):
	if (i==0):
		for sec in h.allsec():
			sec.Ra = 266.1 #ohm-cm
	v_s = h.Vector().record(p_soma(0.5)._ref_v) 		# soma membrane potential
	v_z = h.Vector().record(p_siz(0.5)._ref_v)		# putative SIZ membrane potential
	t = h.Vector().record(h._ref_t)                     # Time stamp vector
	h.finitialize(-55 * mV)
	h.continuerun(500*ms)

	t_traces.append(list(t))
	v_soma.append(list(v_s))
	v_siz.append(list(v_z))

plt.rcParams["figure.figsize"] = (10,12)
plt.hold(True)
# plot soma voltage, red
plt.plot(t_traces[0], v_soma[0], label = "soma", color='r', ls = '--')
#plt.plot(t_traces[1], v_soma[1], label = "soma, high R_a", color='r')
# plot axon voltage, orange
plt.plot(t_traces[0], v_siz[0], label = "pSIZ", color='#ffa500', ls = '--')
#plt.plot(t_traces[1], v_siz[1], label = "pSIZ, high R_a", color='#ffa500')

### 
### read experimental traces from MATLAB export
### trace 1 = 2016_01_28
###
trace_va6 = pd.read_csv('va6_2016_01_28.csv', header = None, dtype = np.float64)
trace_va6 = trace_va6.values.T
trace_va6_t = np.linspace(90,190,1000)
plt.plot(trace_va6_t, trace_va6-55, label = "VA6 trace", color = 'r', linewidth=2)

trace_vl2a = pd.read_csv('vl2a_2016_01_28.csv', header = None, dtype = np.float64)
trace_vl2a = trace_vl2a.values.T
trace_vl2a_t = np.linspace(290,390,1000)
plt.plot(trace_vl2a_t, trace_vl2a-55, label = "VL2a trace", color = 'b', linewidth=2)

plt.legend(loc = "upper right")
plt.xlim(90, 400)
plt.ylim(-55, -30)
plt.xlabel("time (ms), VA6 activates at 100 ms, VL2a at 300 ms")
plt.ylabel("membrane potential (mV)")
plt.title("PN-like (Tobin et al) parameters, 60% lower synaptic conductance")
plt.show()
'''

###
### plot diameter of sections vs path length going from the primary neurite into 
### either the dendritic arbor (red) or the axonal arbor (black)
### goal is to see whether there's a significant difference in diameter at the branch point
###
def plot_stretch_of_diams(figname = 'branch_point_diams.svg', showfig = True):
	print("function only works for local5 v1") # bc the section path is hard coded

	plt.rcParams["figure.figsize"] = (17,12)

	sections_into_axon = [224, 225, 227, 231]
	len_into_axon = [0]
	diams_into_axon = []
	for sec_num in sections_into_axon:
		new_sec_index = len(len_into_axon)-1
		print("for sec num {} the index is {}".format(str(sec_num), str(new_sec_index)))
		for seg in cell1.axon[sec_num]:
			len_into_axon.append(len_into_axon[new_sec_index] + seg.x * cell1.axon[sec_num].L)
			diams_into_axon.append(seg.diam)
	plt.plot(len_into_axon[1:], diams_into_axon, color = 'black', label = 'axon path')

	sections_into_dendr = [224, 796, 798, 800]
	len_into_dendr = [0]
	diams_into_dendr = []
	for sec_num in sections_into_dendr:
		new_sec_index = len(len_into_dendr)-1
		print("for sec num {} the index is {}".format(str(sec_num), str(new_sec_index)))
		for seg in cell1.axon[sec_num]:
			len_into_dendr.append(len_into_dendr[new_sec_index] + seg.x * cell1.axon[sec_num].L)
			diams_into_dendr.append(seg.diam)
	plt.plot(len_into_dendr[1:], diams_into_dendr, color = 'red', label = 'dendrite path')
	plt.legend(loc= 'upper right')

	plt.xlabel("path length (starting from section of primary neurite) (\u03BCm)")
	plt.ylabel("neurite diameter (\u03BCm)")

	draw()

	plt.savefig(figname)

	if showfig:
		plt.show()

###
### set diameters of axon initial section to those of dendrite initial section
### note: these sections are PROXIMAL to the long AIS, axon[231]
###
def widen_axon_initial_sec():

	for seg in cell1.axon[225]:
		seg.diam += 0.085
	for seg in cell1.axon[227]:
		seg.diam += 0.23

###
### save mEPSP_vs_gd graph after multiple different diameter increases to the axon
### initial section
###
def thicken_AIS():

	# baseline measure
	#mEPSP_vs_gd(figname = 'mEPSP_vs_gd_AIS_thicken0.svg', showfig = False)
	plot_stretch_of_diams(figname = 'branch_point_diams_thicken0.svg', showfig = False)

	for seg in cell1.axon[231]:
		seg.diam += 0.6


	for i in range(7,12):
		for seg in cell1.axon[231]:
			seg.diam += 0.1

		mEPSP_vs_gd(figname = 'mEPSP_vs_gd_AIS_thicken{}.svg'.format(str(0.1*i)), showfig = False)
		plot_stretch_of_diams(figname = 'branch_point_diams_thicken{}.svg'.format(str(0.1*i)), showfig = False)

###
### plot size of miniEPSP vs distance from soma
###
### NOTE: d_lambda meshing produces different
### record_neurite: measure the EPSP at the location of the synapse rather than @ the soma
def mEPSP_vs_gd(record_neurite = False, figname = 'fig_mEPSP_vs_gd.svg', showfig = False):
	#change_R_a(125) #ohm-cm

	gd_va6 = find_geodesic(syns)
	gd_vl2a = find_geodesic(syns2)

	### first set VA6 synapses to weight 0
	for i in range(len(list(syns))):
		ncs.object(i).weight[0] = 0 # uS, peak conductance change
		#h.pop_section()

	# sequentially activate all VA6 synapses
	amp_va6 = []
	peak_t_va6 = []
	diams_va6 = []
	t_sim = []			
	v_soma_sim = []
	v_siz_sim = []
	# re-run simulation for each synapse
	for i in (range(len(list(syns)))):
		diams_va6.append(syns.object(i).get_segment().diam)

		# activate a single synapse
		ncs.object(i).weight[0] = syn_strength
		if i % 20 == 1:
			print("activating VA6 synapse " + str(i))

		if record_neurite:
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a(record_from_seg = syns.object(i).get_segment())
		else:
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
		t_sim.append(curr_t)
		v_soma_sim.append(curr_v_soma)
		v_siz_sim.append(curr_v_siz)
		#print(mean(curr_v_soma[0]), mean(curr_v_soma[1]))

		if not curr_v_soma[0]:
			print("run is empty")
		else:
			peak_index = curr_v_soma[0].index(max(curr_v_soma[0]))
			amp_va6.append(max(curr_v_soma[0]) + 55)
			peak_t_va6.append(curr_t[0][peak_index])
			if i % 30 == 1:
				print("VA6 synapse " + str(i) + ", max amp: " + str(max(curr_v_soma[0]) + 55))
				print("verify VL2a is exp value, max amp: " +  str(max(curr_v_soma[1]) + 55))

		h.pop_section() # clear the section stack to avoid overflow (triggered by using ".L" above?)

		# de-activate the synapse
		ncs.object(i).weight[0] = 0

	# reset VA6 synapses before other tests:
	for i in range(len(list(syns))):
		ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

	### set VL2a synapses to weight 0
	for i in range(len(list(syns2))):
		ncs2.object(i).weight[0] = 0 # uS, peak conductance change
		#h.pop_section()

	# sequentially activate all VL2a synapses
	amp_vl2a = []
	peak_t_vl2a = []
	diams_vl2a = []
	t_sim = []			
	v_soma_sim = []
	v_siz_sim = []
	# re-run simulation for each synapse
	for i in range(len(list(syns2))):
		diams_vl2a.append(syns2.object(i).get_segment().diam)

		# activate a single synapse
		ncs2.object(i).weight[0] = syn_strength

		if record_neurite:
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a(record_from_seg = syns2.object(i).get_segment())
		else:
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
		#t_sim.append(curr_t)
		#v_soma_sim.append(curr_v_soma)
		#v_siz_sim.append(curr_v_siz)

		if not curr_v_soma[1]:
			print("run is empty")
		else:
			peak_index = curr_v_soma[1].index(max(curr_v_soma[1]))
			amp_vl2a.append(max(curr_v_soma[1]) + 55)
			peak_t_vl2a.append(curr_t[1][peak_index])
			if i % 20 == 1:
				print("VL2a synapse " + str(i) + ", max amp: " + str(max(curr_v_soma[1]) + 55))
				print("verify VA6 is exp value, max amp: " +  str(max(curr_v_soma[0]) + 55))

		h.pop_section()

		# de-activate the synapse
		ncs2.object(i).weight[0] = 0

	# reactivate VL2a synapses
	for i in range(len(list(syns2))):
		ncs2.object(i).weight[0] = syn_strength # uS, peak conductance change

	plt.rcParams["figure.figsize"] = (4, 4) # striving for square-ness: 19 for the matplotlib popup
											  # 25 for the svg to be a square; 25, 28 
											  # update: scaling it to figure size

	# plot differences in max amplitude, kinetics of different synapses
	fig, axs = plt.subplots(nrows = 2, ncols = 2)
	fig.subplots_adjust(top=0.9)

	if record_neurite:
		df_amp_va6 = pd.DataFrame({'dist_syn_to_soma_um': gd_va6, 
									'mEPSP_at_neurite_uV': [amp*1000 for amp in amp_va6]})
		df_amp_vl2a = pd.DataFrame({'dist_syn_to_soma_um': gd_vl2a, 
									'mEPSP_at_neurite_uV': [amp*1000 for amp in amp_vl2a]})
		df_amp_va6.to_csv('mEPSP_at_syn_VA6.csv')
		df_amp_vl2a.to_csv('mEPSP_at_syn_VL2a.csv')

	print("length of gd_va6 " + str(len(gd_va6)) + ", length of amp_va6 " + str(len(amp_va6)))
	print("length of gd_vl2a " + str(len(gd_vl2a)) + ", length of amp_vl2a " + str(len(amp_vl2a)))
	axs[0,0].scatter(gd_va6, [amp*1000 for amp in amp_va6], label = "VA6 synapses", color = 'r')
	axs[0,0].scatter(gd_vl2a, [amp*1000 for amp in amp_vl2a], label = "VL2a synapses", color = 'b')
	axs[0,0].set_xlabel("synapse to soma distance (\u03BCm)") # , fontsize = 28
	axs[0,0].set_ylabel("miniEPSP size (\u03BCV)") # , fontsize = 28
	#axs[0,0].legend(loc = "upper right")
	axs[0,0].spines['right'].set_visible(False)
	axs[0,0].spines['top'].set_visible(False)
	if which_local5 == 1 and not record_neurite:
		axs[0,0].set_ylim(17, 43)
		axs[0,0].set_xlim(95, 270)
	elif which_local5 == 2: 
		axs[0,0].set_ylim(0.020, 0.050)

	axs[0,1].scatter(gd_va6, peak_t_va6, color = 'r')
	axs[0,1].scatter(gd_vl2a, peak_t_vl2a, color = 'b')
	axs[0,1].set_xlabel("synapse to soma distance (\u03BCm)")
	axs[0,1].set_ylabel("time to miniEPSP peak (ms)")
	axs[0,1].spines['right'].set_visible(False)
	axs[0,1].spines['top'].set_visible(False)

	axs[1,0].scatter(diams_va6, amp_va6, color = 'r')
	axs[1,0].scatter(diams_vl2a, amp_vl2a, color = 'b')
	axs[1,0].set_xlabel("neurite diameter (\u03BCm)")
	axs[1,0].set_ylabel("miniEPSP size (mV)")
	axs[1,0].spines['right'].set_visible(False)
	axs[1,0].spines['top'].set_visible(False)

	axs[1,1].scatter(diams_va6, peak_t_va6, color = 'r')
	axs[1,1].scatter(diams_vl2a, peak_t_vl2a, color = 'b')
	axs[1,1].set_xlabel("neurite diameter (\u03BCm)")
	axs[1,1].set_ylabel("time to miniEPSP peak (ms)")
	axs[1,1].spines['right'].set_visible(False)
	axs[1,1].spines['top'].set_visible(False)

	props = ("Units: g_pas (S/cm^2), g_syn (nS), c_m (uF/cm^2), R_a (Ohm-cm)")
	if record_neurite:
		plt.suptitle("Amplitude of miniEPSPs at synapse location, " + props + " [local5 (" + ID + ")]", 
				 fontsize = 12, y = 0.96)
	else:
		plt.suptitle("Amplitude of miniEPSPs at soma, " + props + " [local5 (" + ID + ")]", 
				 fontsize = 12, y = 0.96)

	draw()

	fig.savefig(figname)

	if showfig:
		plt.show()

def fig_measureAlongNeuron(figname = 'fig_measureAlongNeuron.svg'):
	'''
		create paper figure with traces for measuring VA6 and VL2a EPSPs at 
		soma, in axon, and in dendrite
		@ soma will be superimposed on the experimental trace
	'''

	t_traces = []
	v_soma = []
	v_dendr = []
	v_axon = []

	for i in range(0,2):
		if (i==0):
			# run VA6 first
			nc.number = 1
			nc.start = 0
			nc2.number = 0
			nc2.start = 0
		elif (i==1):
			# run VL2a second
			nc.number = 0
			nc.start = 0
			nc2.number = 1
			nc2.start = 0
		
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)
		v_s = h.Vector().record(p_soma(0.5)._ref_v) 			# soma membrane potential
		v_d = h.Vector().record(p_dendr(0.5)._ref_v)		# dendrite 
		v_a = h.Vector().record(p_axon(0.5)._ref_v)			# axon 
		t = h.Vector().record(h._ref_t)                     	# Time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(60*ms)

		t_traces.append(list(t))
		v_soma.append(list(v_s))
		v_dendr.append(list(v_d))
		v_axon.append(list(v_a))

	plt.rcParams["figure.figsize"] = (17,15)

	fig, axs = plt.subplots(nrows = 2, ncols = 3, sharey = True, sharex = True)
	# rows: VA6, VL2a; cols: soma, dendrite, axon

	# plot sim soma traces
	axs[0,0].plot(t_traces[0], v_soma[0], color = 'r')
	axs[1,0].plot(t_traces[1], v_soma[1], color = 'r')

	# plot experimental traces from MATLAB export
	# trace from Jamie's edits, 2020-07-15
	t_vl2a_tofit = trace_vl2a[0]+3.0
	axs[0,0].plot(t_va6_tofit[0:250], v_va6_tofit[0:250], label = None, color = 'r', linewidth=4, alpha=0.2)
	axs[1,0].plot(t_vl2a_tofit, v_vl2a_tofit, label = None, color = 'r', linewidth=3, alpha=0.2)

	# plot sim dendrite traces
	axs[0,1].plot(t_traces[0], v_dendr[0], color = 'limegreen')
	axs[1,1].plot(t_traces[1], v_dendr[1], color = 'limegreen')

	# plot sim axon traces
	axs[0,2].plot(t_traces[0], v_axon[0], color = 'indigo')
	axs[1,2].plot(t_traces[1], v_axon[1], color = 'indigo')

	for i in range(0,2):
		for j in range(0,3):
			axs[i,j].spines['right'].set_visible(False)
			axs[i,j].spines['top'].set_visible(False)
			#if (i,j) != (1,0):
			#	axs[i,j].spines['left'].set_visible(False)
			#	axs[i,j].spines['bottom'].set_visible(False)

	for ax in axs.flatten():
		ax.xaxis.set_tick_params(labelbottom=True)
		ax.yaxis.set_tick_params(labelleft=True)

	axs[0,0].set_ylabel('potential (mV)')
	axs[1,0].set_ylabel('potential (mV)')
	axs[0,0].set_ylabel('time (ms)')
	axs[1,0].set_xlabel('time (ms)')
	axs[0,0].set_title('soma')
	axs[0,1].set_title('dendrite')
	axs[0,2].set_title('axon')

	draw()

	fig.savefig(figname)

###
### create paper figure demonstrating that equalizing synapse count
### doesn't equalize VA6 and VL2a EPSP sizes
###
def fig_equalizeSynCount(figname = 'fig_equalizeSynCount.svg', numIters = 1000):

	# try X different permutations of removing VA6 synapses

	# save baseline traces
	t_traces, v_soma, v_siz = run_VA6_VL2a()
	t_va6_base = t_traces[0]
	t_vl2a_base = t_traces[1]
	v_va6_base = v_soma[0]
	v_vl2a_base = v_soma[1]

	# equalize synapse count
	t_va6_pruned = []
	v_va6_pruned = []
	for i in range(numIters):
		prune_VA6(prune_worst = False) # prune random selection of VA6 synapses

		t_traces, v_soma, v_siz = run_VA6_VL2a()
		t_va6_pruned.append(t_traces[0])
		v_va6_pruned.append(v_soma[0])

		reset_VA6()

	### plot figure
	plt.rcParams["figure.figsize"] = (15,15)

	fig, ax1 = plt.subplots(nrows = 1, ncols = 1, sharey = False)

	# plot baseline VA6 and VL2a traces
	ax1.plot(t_va6_base, v_va6_base, color='r', linewidth = 3)
	ax1.plot(t_vl2a_base, v_vl2a_base, color='b', linewidth = 3)

	# plot VA6 traces post synapse count equalization
	ax1.plot(t_va6_pruned[0], v_va6_pruned[0], label = "VA6 after pruning (x{})".format(str(numIters)), 
			 color='r', ls = '--', dashes=(5, 10))
	for i in range(1, len(v_va6_pruned)):
		ax1.plot(t_va6_pruned[i], v_va6_pruned[i], color='r', ls = '--',dashes=(5, 10))
	
	ax1.legend(loc = "lower right", fontsize = '40', frameon = False)

	# read experimental traces from MATLAB export
	# trace from Jamie's edits, 2020-07-15
	t_vl2a_tofit = trace_vl2a[0]+3.0
	ax1.plot(t_va6_tofit[0:250], v_va6_tofit[0:250], label = None, color = 'r', linewidth=3, alpha=0.2)
	ax1.plot(t_vl2a_tofit, v_vl2a_tofit, label = None, color = 'b', linewidth=3, alpha=0.2)

	ax1.set_xlim(-5, 40)
	ax1.set_xticks(ticks=np.arange(0, 40, step=10))
	ax1.set_ylim(-55, -48)
	ax1.set_yticks(ticks=np.arange(-55, -48, step=2))
	ax1.set_ylabel("membrane potential (mV)", fontsize = '45')
	ax1.set_xlabel("time (ms), VA6 and VL2a activate at 0 ms", fontsize = '45')
	ax1.tick_params(axis='both', which='major', labelsize=35)
	ax1.tick_params(axis='both', which='minor', labelsize=35)
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)

	props = ("Params.: g_pas (S/cm^2) = " + str(round(g_pas, 6)) + 
			 ", g_syn (nS) = " + str(round(syn_strength*1000, 3)) + 
			 ", c_m (uF/cm^2) = " + str(round(c_m, 2)) +
			 ", R_a (Ohm-cm) = " + str(round(R_a)))

	plt.suptitle(props + " [local5 (" + ID + ")]", 
				 fontsize = 24, y = 0.96)

	draw()

	fig.savefig(figname)

	plt.show()

###
### create figure showing that permuting synapse locs while preserving
### geodesic distance still doesn't equalize VA6 and VL2a EPSP sizes
###
def fig_permuteSynapseLocs(figname = 'fig_permuteSynLocs.svg', numIters = 2):

	# permute_synapse_locs output: 
	# 	t_sim, v_soma_sim: list of length run_count, 
	#					   each element list of length 2: index 0 va6 time/voltage trace, 
	#													  index 1 vl2a trace
	t_va6, v_va6_soma = permute_synapse_locs(syns, run_count = numIters, gd_threshold = 0.1,
						 plot_SIZ = False, plot_here = False)
	t_vl2a, v_vl2a_soma = permute_synapse_locs(syns2, run_count = numIters, gd_threshold = 0.1,
						 plot_SIZ = False, plot_here = False)

	### plot figure
	plt.rcParams["figure.figsize"] = (15,15)

	fig, ax1 = plt.subplots(nrows = 1, ncols = 1, sharey = False)

	# plot baseline VA6 and VL2a traces
	ax1.plot(t_va6[0][0], v_va6_soma[0][0], color='r', linewidth = 3)
	ax1.plot(t_vl2a[0][1], v_vl2a_soma[0][1], color='b', linewidth = 3)

	# plot VA6 traces post permutations
	ax1.plot(t_va6[1][0], v_va6_soma[1][0], label = "VA6 after permuting (x{})".format(str(numIters)), 
			 color='r', ls = '--', dashes=(5, 10))
	for i in range(2, len(v_va6_soma)):
		ax1.plot(t_va6[i][0], v_va6_soma[i][0], color='r', ls = '--',dashes=(5, 10))

	# plot VL2a traces post permutations
	ax1.plot(t_vl2a[1][1], v_vl2a_soma[1][1], label = "VL2a after permuting (x{})".format(str(numIters)), 
			 color='b', ls = '--', dashes=(5, 10))
	for i in range(2, len(v_vl2a_soma)):
		ax1.plot(t_vl2a[i][1], v_vl2a_soma[i][1], color='b', ls = '--',dashes=(5, 10))
	
	ax1.legend(loc = "lower right", fontsize = '40', frameon = False)

	# read experimental traces from MATLAB export
	# trace from Jamie's edits, 2020-07-15
	t_vl2a_tofit = trace_vl2a[0]+3.0
	ax1.plot(t_va6_tofit[0:250], v_va6_tofit[0:250], label = None, color = 'r', linewidth=3, alpha=0.2)
	ax1.plot(t_vl2a_tofit, v_vl2a_tofit, label = None, color = 'b', linewidth=3, alpha=0.2)

	ax1.set_xlim(-5, 40)
	ax1.set_xticks(ticks=np.arange(0, 40, step=10))
	ax1.set_ylim(-55, -48)
	ax1.set_yticks(ticks=np.arange(-55, -48, step=2))
	ax1.set_ylabel("membrane potential (mV)", fontsize = '45')
	ax1.set_xlabel("time (ms), VA6 and VL2a activate at 0 ms", fontsize = '45')
	ax1.tick_params(axis='both', which='major', labelsize=35)
	ax1.tick_params(axis='both', which='minor', labelsize=35)
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)

	props = ("Params.: g_pas (S/cm^2) = " + str(round(g_pas, 6)) + 
			 ", g_syn (nS) = " + str(round(syn_strength*1000, 3)) + 
			 ", c_m (uF/cm^2) = " + str(round(c_m, 2)) +
			 ", R_a (Ohm-cm) = " + str(round(R_a)))

	plt.suptitle(props + " [local5 (" + ID + ")]", 
				 fontsize = 24, y = 0.96)

	draw()

	fig.savefig(figname)

	plt.show()

###
### test geodesic distance effects on EPSP size
### alter the length of sections to equalize VA6 and VL2a geodesic distance means
###
def test_gd_effects(remove_231_downstream = False, plot_SIZ = False):

	t_sim = []			
	v_soma_sim = []
	v_siz_sim = []

	for i in range(0,4):
		if (i==0):
			# baseline run
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
			t_sim.append(curr_t)
			v_soma_sim.append(curr_v_soma)
			v_siz_sim.append(curr_v_siz)
		elif (i==1):
			# prune VA6 run
			prune_VA6(prune_worst = True)
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
			t_sim.append(curr_t)
			v_soma_sim.append(curr_v_soma)
			v_siz_sim.append(curr_v_siz)
		elif (i==2):
			# shorten axon run
			if remove_231_downstream:
				remove_equiv_area_downstream()
			else:
				remove_section()
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
			t_sim.append(curr_t)
			v_soma_sim.append(curr_v_soma)
			v_siz_sim.append(curr_v_siz)
		elif (i==3):
			lengthen_section()
			curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
			t_sim.append(curr_t)
			v_soma_sim.append(curr_v_soma)
			v_siz_sim.append(curr_v_siz)


	plt.rcParams["figure.figsize"] = (10,12)
	# plot manipulation results	
	fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = False)
	fig.subplots_adjust(top=0.9)

	#plt.hold(True)

	for i in range(0, len(t_sim)):
		if (i==3):
			LS = '--' #dashed
			LW = 3.0   #line as wide as the experimental trace
			label = "4. dendrite lengthened, final equalized mean geodesic distances"
		elif (i==2):
			LS = '-.' #dash dotted
			LW = 1.4
			label = "3. axon shortened, but VL2a still farther from soma"
		elif (i==1):
			LS = (0, (3, 1, 1, 1, 1, 1)) #densely dashdotdotted
			LW = 1.3
			label = "2. equalized synapse counts"
		elif (i==0):
			LS = ':' # dotted
			LW = 1.0 # thicken the loosely dotted line
			label = "1. baseline run"
		else:
			LS = ':' #dotted
			LW = 1.4

		#index = lowest_errors[i]
		index = i

		# plot VA6
		ax1.plot(t_sim[index][0], v_soma_sim[index][0], label = "soma: " + label, color='r', ls = LS, lw = LW)
		ax1.axhline(y = max(v_soma_sim[index][0]), ls = LS, color = 'r')

		# plot VL2a
		ax2.plot(t_sim[index][1], v_soma_sim[index][1], label = "soma: " + label, color='b', ls = LS, lw = LW)
		ax2.axhline(y = max(v_soma_sim[index][1]), ls = LS, color = 'b')

		if plot_SIZ:
			ax1.plot(t_sim[index][0], v_siz_sim[index][0], color='#ffa500', ls = LS, lw = LW)
			ax2.plot(t_sim[index][1], v_siz_sim[index][1], color='#ffa500', ls = LS, lw = LW)

		# at the final equalized geodesic distances, plot max line in opposite plot
		if (i==3):
			ax1.axhline(y = max(v_soma_sim[index][0]), ls = LS, lw = LW, color = 'r')
			ax2.axhline(y = max(v_soma_sim[index][0]), ls = LS, lw = LW, color = 'r')
			ax1.axhline(y = max(v_soma_sim[index][1]), ls = LS, lw = LW, color = 'b')
			ax2.axhline(y = max(v_soma_sim[index][1]), ls = LS, lw = LW, color = 'b')

			ax1.plot(t_sim[index][1], v_soma_sim[index][1], label = "overlaid final VL2a simulation trace",
					 color='b', ls = LS, lw = 1.0)

	### 
	### read experimental traces from MATLAB export
	### trace from Jamie's edits, 2020-07-15
	###
	ax1.plot(t_va6_tofit, v_va6_tofit, label = "VA6 experimental trace", color = 'r', linewidth=2)
	ax2.plot(t_vl2a_tofit, v_vl2a_tofit, label = "VL2a experimental trace", color = 'b', linewidth=2)

	ax1.legend(loc = "upper right", fontsize = '16')
	ax1.set_xlim(-5, 60)
	ax1.set_xticks(ticks=np.arange(-5, 60, step=5))
	ax1.set_ylim(-55, -48)
	ax1.set_yticks(ticks=np.arange(-55, -48, step=1))
	ax1.set_ylabel("membrane potential (mV)", fontsize = '20')
	ax1.set_xlabel("time (ms), VA6 activates at 0 ms")
	ax1.set_title("VA6 fit")

	ax2.legend(loc = "upper right", fontsize = '16')
	ax2.set_xlim(-5, 60)
	ax2.set_xticks(ticks=np.arange(-5, 60, step=5))
	ax2.set_ylim(-55, -48)
	ax2.set_yticks(ticks=np.arange(-55, -48, step=1))
	ax2.set_xlabel("time (ms), VL2a activates at 0 ms")
	ax2.set_title("VL2a fit")

	props = ("Units: g_pas (S/cm^2), g_syn (nS), c_m (uF/cm^2), R_a (Ohm-cm)")
	plt.suptitle("Equalizing geodesic distances also equalizes EPSP amplitude, " + props + " [local5 (" + ID + ")]", 
				 fontsize = 24, y = 0.96)

	plt.show()

	# return voltage traces when the two are equalized
	#return v_soma_sim[3][0], v_soma_sim[3][1]

###
### understand why VA6 amplitude falls when axon truncated
### record from:
### axon initial section axon[225](0.5) (L=1.047)
### dendrite initial section axon[796](0.136) (L=3.848)
### long part of the axon initial section is axon[231] -- 2020-10-12
### end of primary neurite axon[224](0.99) (L=54.1)

### measure_at_junct specifies whether we record axial current at the primary junction
### point, where the primary neurite splits into dendrite and soma
def measure_at_junct():

	if which_local5 != 1:
		print("function only works for local5 v1")

	t_traces = []

	### ALL currents measured flowing away from the soma, proximal to distal
	i_soma = []		# current flowing away from soma
	i_axon = []		# current flowing towards axon
	i_dendr = []	# current flowing towards dendrite
	
	for i in range(0,2):
		if (i==0):
			# run VA6 first
			nc.number = 1
			nc.start = 0
			nc2.number = 0
			nc2.start = 0
		elif (i==1):
			# run VL2a second
			nc.number = 0
			nc.start = 0
			nc2.number = 1
			nc2.start = 0
		

		### to purely run code once, copy and paste below
		nc.number = 1
		nc.start = 0

		h.load_file('stdrun.hoc')
		x = h.cvode.active(True) 

		v_junct = h.Vector().record(cell1.axon[224](1)._ref_v)
		# same as recording from cell1.axon[225](0) and cell1.axon[796](0)
		
		v_d = h.Vector().record(cell1.axon[796](0.001)._ref_v)
		v_a = h.Vector().record(cell1.axon[225](0.001)._ref_v)
		v_s = h.Vector().record(cell1.axon[224](0.99)._ref_v)

		#v_s = h.Vector().record(record_from_seg._ref_v) 		# soma membrane potential
		#v_z = h.Vector().record(p_siz(0.5)._ref_v)		# putative SIZ membrane potential
		t = h.Vector().record(h._ref_t)                     # Time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(60*ms)

		i_d = (v_d - v_junct) / (cell1.axon[796](0.001).ri()) # black, expect to be negative
		i_a = (v_a - v_junct) / (cell1.axon[225](0.001).ri()) # orange, expect to be positive
		i_s = (v_junct - v_s) / (cell1.axon[224](1).ri()) # red, expect to be negative

		plt.rcParams["figure.figsize"] = (17,12)
		plt.plot(list(t), list(i_d), label="from dendrite", color='black', lw = 3)
		plt.plot(list(t), list(i_a), label="to axon", color='orange')
		plt.plot(list(t), -1*np.array(list(i_s)), label="to soma", color='red')
		plt.plot(list(t), np.add(list(i_s), np.abs(list(i_a))), label = "inverse sum current leaving junction", 
				 ls = "--", lw = 5)
		plt.legend(loc = 'upper right')
		plt.xlabel("time (ms)")
		plt.ylabel("current (nA), positive current enters the primary junction")
		plt.ylim(-0.06, 0.08)
		plt.xlim(-2, 40)

		plt.show()

		print("total current from dendrite: " + str(np.trapz(list(i_d), x = list(t))))
		# before removing section, 0.2052775906143184
		# after removing section, 0.22124113527544212

		t_traces.append(list(t))
		i_soma.append(list(i_s))
		i_dendr.append(list(i_d))
		i_axon.append(list(i_a))

###
### compare diameters of different neurite compartments
### plot histograms of neurite diameters of following:
### primary neurite, dendritic arbor, axon initial section, axon arbor
### also creates dataframe of each segment's diameter, class, and G.D.
###
def plot_diams():

	plt.rcParams["figure.figsize"] = (17,23)
	fig, axs = plt.subplots(nrows = 4, ncols = 1, sharex = True)

	neurite_info = []

	for i in range(0,4):
		if which_local5 == 1:
			if (i==0):
				title = "primary neurite"
				sec_range = np.arange(135, 225)
				col = 'red'
			elif (i==1):
				title = "dendritic arbor"
				sec_range = np.arange(796, 1958)
				col = 'black'
			elif (i==2):
				title = "axon initial section"
				sec_range = np.arange(231, 232)
				col = 'orange'
			elif (i==3):
				title = "axonal arbor"
				sec_range = np.arange(232, 796)
				col = 'grey'
		elif which_local5 == 2:
			if (i==0):
				title = "primary neurite"
				sec_range = np.arange(1, 36)
				col = 'red'
			elif (i==1):
				title = "dendritic arbor"
				sec_range = np.arange(819, 1696)
				col = 'black'
			elif (i==2):
				title = "axon initial section"
				sec_range = np.arange(36, 37)
				col = 'orange'
			elif (i==3):
				title = "axonal arbor"
				sec_range = np.arange(37, 819)
				col = 'grey'

		diams = []
		for j in sec_range:
			for seg in cell1.axon[j]:
				toAppend = {}
				diams.append(seg.diam)
				geodesic_dist = h.distance(p_soma(0.5), seg)
				toAppend.update(neurite_class = title, diam = seg.diam, 
								geodesic_dist = geodesic_dist)
				neurite_info.append(toAppend)

		bins= np.linspace(0, 1.2, 48)
		axs[i].hist(diams, bins = bins, label = title, color = col)
		axs[i].legend(loc = "upper right")

	axs[0].set_ylabel("number of segments")
	axs[3].set_xlabel("neurite diameter (um)")

	neurite_info = pd.DataFrame(neurite_info)
	#neurite_info.to_csv("neurite_info.csv")

	plt.show()

###
### permute synapses to different branches of same geodesic distance
### should not change GD distribution, only location relative to branching
### structure and to neurite diameters, perhaps Strahler order, clustering metrics
###
def permute_synapse_locs(synapses, run_count = 10, gd_threshold = 0.1,
						 plot_SIZ = False, plot_here = False):

	gd = find_geodesic(synapses)
	t_sim = []			
	v_soma_sim = []
	v_siz_sim = []

	# simulate baseline EPSPs:
	curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
	t_sim.append(curr_t)
	v_soma_sim.append(curr_v_soma)
	v_siz_sim.append(curr_v_siz)

	# re-run simulation "a bunch" of times
	for i in range(0, run_count): 

		original_loc = []
		original_diams = []
		new_diams = []

		# iterate through all synapses, change their location
		for j in range(0, len(synapses)):
			# find gd of current synapse, then find other tree positions with similar gd
			curr_gd = gd[j]
			# save current location
			original_loc.append(synapses.object(j).get_segment())
			original_ind = re.split('\[|\]', str(original_loc[j]))[3]
			# save current synapse diameter
			original_diams.append(synapses.object(j).get_segment().diam)

			# compute tree locations with the close geodesic distance as current synapse location
			# shuffle_locs is an array of tuples: [(index in tree of relevant 3d point, GD from soma), ...]
			shuffle_locs = [(ind, close) for ind, close in enumerate(tree["gd"]) 
							if ((close < curr_gd+gd_threshold) & (close > (curr_gd-gd_threshold)))]

			# randomly pick one of the tree locs with the same GD to permute to
			# keep shuffling until we preserve the neurite arbor
			new_loc_ind = random.randint(0, len(shuffle_locs)-1)
			h.pop_section()
			if cell1.axon[int(original_ind)] in cell1.dendrites:
				# find sec # associated w/ this index in the tree
				sec = int(tree.loc[shuffle_locs[new_loc_ind][0], 'sec']) 
				while (cell1.axon[sec] not in cell1.dendrites):
					new_loc_ind = random.randint(0, len(shuffle_locs)-1)
					sec = int(tree.loc[shuffle_locs[new_loc_ind][0], 'sec'])
					h.pop_section() 
			elif cell1.axon[int(original_ind)] in cell1.axons:
				sec = int(tree.loc[shuffle_locs[new_loc_ind][0], 'sec'])
				while (cell1.axon[sec] not in cell1.axons):
					new_loc_ind = random.randint(0, len(shuffle_locs)-1)
					sec = int(tree.loc[shuffle_locs[new_loc_ind][0], 'sec'])
					h.pop_section() 

			# pull section and segment information of new location:
			sec = int(tree.loc[shuffle_locs[new_loc_ind][0], 'sec'])
			p3d = tree.loc[shuffle_locs[new_loc_ind][0], 'seg'] # NOTE: seg actually is the index of the closest 3d point
			x = cell1.axon[sec].arc3d(p3d) / cell1.axon[sec].L

			# change to new location!
			synapses.object(j).loc(cell1.axon[sec](x))
			new_diams.append(synapses.object(j).get_segment().diam)
			h.pop_section()

		###
		### consider how new synapse placement differs from baseline:
		###
		### diameter comparison:
		if i==1 and plot_here:
			plt.rcParams["figure.figsize"] = (10,8)
			bins = np.linspace(0, 1.2, 48)
			plt.hist(original_diams, bins, alpha = 0.5, label = 'original diameters', color = 'orange')
			plt.axvline(mean(original_diams), color='orange', linestyle='dashed', linewidth=1)
			plt.hist(new_diams, bins, alpha = 0.5, label = 'permutated diameters', color = 'green')
			plt.axvline(mean(new_diams), color='green', linestyle='dashed', linewidth=1)
			plt.legend(loc = 'upper right')
			plt.xlabel("neurite diameters (um)")
			plt.ylabel("# synapses")
			plt.xlim(0, 1.2)
			#plt.ylim(0, 40)
			plt.show()

		# simulate the EPSP generated by the permutated synapses
		curr_t, curr_v_soma, curr_v_siz = run_VA6_VL2a()
		t_sim.append(curr_t)
		v_soma_sim.append(curr_v_soma)
		v_siz_sim.append(curr_v_siz)

		# place synapses back to their old locations
		for j in range(0, len(synapses)):
			synapses.object(j).loc(original_loc[j])

	if plot_here:
		### plot resultant traces
		plt.rcParams["figure.figsize"] = (10,12)
		# plot manipulation results	
		fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = False)
		fig.subplots_adjust(top=0.9)

		#plt.hold(True)

		for i in range(0, len(t_sim)):
			if (i==0):
				LS = '--' #dashed
				LW = 3.0   #line as wide as the experimental trace
				label = "baseline EPSP"
			elif (i==1):
				LS = ':' #dotted
				LW = 1.4
				label = "permuted synapse EPSP"
			else:
				LS = ':' #dotted
				LW = 1.4
				label = None 

			#index = lowest_errors[i]
			index = i

			# plot VA6
			ax1.plot(t_sim[index][0], v_soma_sim[index][0], label = label, color='r', ls = LS, lw = LW)
			ax1.axhline(y = max(v_soma_sim[index][0]), ls = LS, color = 'r')

			# plot VL2a
			ax2.plot(t_sim[index][1], v_soma_sim[index][1], label = label, color='b', ls = LS, lw = LW)
			ax2.axhline(y = max(v_soma_sim[index][1]), ls = LS, color = 'b')

			if plot_SIZ:
				ax1.plot(t_sim[index][0], v_siz_sim[index][0], color='#ffa500', ls = LS, lw = LW)
				ax2.plot(t_sim[index][1], v_siz_sim[index][1], color='#ffa500', ls = LS, lw = LW)

		### 
		### read experimental traces from MATLAB export
		### trace from Jamie's edits, 2020-07-15
		###
		ax1.plot(t_va6_tofit, v_va6_tofit, label = "VA6 experimental trace", color = 'r', linewidth=2)
		ax2.plot(t_vl2a_tofit, v_vl2a_tofit, label = "VL2a experimental trace", color = 'b', linewidth=2)

		ax1.legend(loc = "upper right", fontsize = '16')
		ax1.set_xlim(-5, 60)
		ax1.set_xticks(ticks=np.arange(-5, 60, step=5))
		ax1.set_ylim(-55, -48)
		ax1.set_yticks(ticks=np.arange(-55, -48, step=1))
		ax1.set_ylabel("membrane potential (mV)", fontsize = '20')
		ax1.set_xlabel("time (ms), VA6 activates at 0 ms")
		ax1.set_title("VA6 fit")

		ax2.legend(loc = "upper right", fontsize = '16')
		ax2.set_xlim(-5, 60)
		ax2.set_xticks(ticks=np.arange(-5, 60, step=5))
		ax2.set_ylim(-55, -48)
		ax2.set_yticks(ticks=np.arange(-55, -48, step=1))
		ax2.set_xlabel("time (ms), VL2a activates at 0 ms")
		ax2.set_title("VL2a fit")

		props = ("Params.: g_pas (S/cm^2) = " + str(round(g_pas, 6)) + 
				 ", g_syn (nS) = " + str(round(syn_strength*1000, 3)) + 
				 ", c_m (uF/cm^2) = " + str(round(c_m, 2)) +
				 ", R_a (Ohm-cm) = " + str(round(R_a)))

		if synapses == syns:
			which_syn = "VA6"
		elif synapses == syns2:
			which_syn = "VL2a"
		plt.suptitle("Permuting " + which_syn + " synapse locations " + str(run_count) + 
					 " times, " + props + " [local5 (" + ID + ")]", 
					 fontsize = 24, y = 0.96)

		plt.show()

	return t_sim, v_soma_sim

if which_local5 == 1 or which_local5 == 2:
	###
	### VA6 and VL2a activations with temporal offsets, ranging from -100 to 100 ms
	### measure relative joint EPSP size at SIZ vs temporal offset;
	### also time to 80% of max EPSP size vs temporal offset 
	### Jamie also asked for in each compartment?
	###
	### 
	def fig_testTimingDiffs(record_from_seg = p_soma(0.5)):

		t_traces = []
		v_soma = []
		v_dendr = []
		v_axon = []

		# set VL2a time, stagger VA6 activation
		nc2.number = 1
		nc2.start = 40

		for i in np.arange(0,81):
			# run VA6 first
			nc.number = 1
			nc.start = i
			
			h.load_file('stdrun.hoc')
			x = h.cvode.active(True)
			v_s = h.Vector().record(record_from_seg._ref_v) 		# soma membrane potential
			v_d = h.Vector().record(p_dendr(0.5)._ref_v)		# dendrite membrane potential
			v_a = h.Vector().record(p_axon(0.5)._ref_v)		# axon membrane potential
			t = h.Vector().record(h._ref_t)                     # Time stamp vector
			h.finitialize(-55 * mV)
			h.continuerun(110*ms)

			t_traces.append(list(t))
			v_soma.append(list(v_s))
			v_dendr.append(list(v_d))
			v_axon.append(list(v_a))

		### plot example traces
		'''
		plt.rcParams["figure.figsize"] = (20,12)
		for i in range(0, 2):
			if i==0:
				trace_num = 30 # VA6 10 seconds before VL2a
				LS = 'solid'
				lag = " VA6 10 ms ahead of VL2a"
			elif i==1:
				trace_num = 50 # VA6 10 seconds behind VL2a
				LS = '--'
				lag = " VA6 10 ms behind VL2a"
			
			time = pd.Series(t_traces[trace_num])-30
			plt.plot(time, v_dendr[trace_num], label = 'dendritic trace' + lag, color = 'black', ls = LS)
			plt.plot(time, v_axon[trace_num], label = 'axonal trace' + lag, color = 'orange', ls = LS)
			plt.plot(time, v_soma[trace_num], label = 'somatic trace' + lag, color = 'red', ls = LS)

		plt.legend(loc = "upper right", fontsize = '16')
		plt.ylabel("membrane potential (mV)", fontsize = '20')
		plt.xlabel("time (ms), VL2a activates at 0 ms")
		plt.title("Raw traces for temporally offset VA6 and VL2a activation")
		plt.show()
		'''

		# plot EPSP at axon, for VA6 preceding VL2a and lagging
		plt.rcParams["figure.figsize"] = (20,10)
		fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = True)
		ax1.plot(pd.Series(t_traces[30])-30, v_axon[30], c='indigo')	# VA6 precedes VL2a
		ax2.plot(pd.Series(t_traces[50])-30, v_axon[50], c='indigo')	# VL2a precedes VA6
		ax1.spines["top"].set_visible(False)
		ax1.spines["right"].set_visible(False)
		ax2.spines["top"].set_visible(False)	
		ax2.spines["right"].set_visible(False)

		draw()
		fig.savefig('fig_axonEPSPactiveSeq.svg')

		plt.show()

		### plot max depolarizations
		plt.rcParams["figure.figsize"] = (15,15)
		fig, ax = plt.subplots(nrows = 1, ncols = 1)
		ax.plot(np.arange(-40,41), [max(v) for v in v_dendr], label = 'peak dendritic depolarization', color = 'limegreen')
		ax.plot(np.arange(-40,41), [max(v) for v in v_axon], label = 'peak axonal depolarization', color = 'indigo')
		ax.plot(np.arange(-40,41), [max(v) for v in v_soma], label = 'peak somatic depolarization', color = 'red')

		ax.legend(loc = "upper left", fontsize = '16')
		#plt.xlim(-30, 30)
		ax.set_ylabel("potential (mV)", fontsize = '20')
		ax.set_xlabel("VA6 minus VL2a activation time (ms)")
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		#plt.title("Peak depolarization for temporally offset VA6 and VL2a activation")

		draw()
		fig.savefig('fig_lag_vs_depolarization.svg')

		plt.show()