
### NOTE: file should be run from directory:
### C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model\\population_model\\

import sys
sys.path.append("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model")
from run_local5 import *
from datetime import datetime
import seaborn as sns
from matplotlib import cm
from scipy import stats

# import pop_sc_model to generate EPSP peaks given SC param sets
from pop_sc_model import *

# set up API connection to neuprint hemibrain server
from neuprint import Client
from neuprint import fetch_simple_connections, fetch_synapse_connections, fetch_neurons
from neuprint import SynapseCriteria as SC, NeuronCriteria as NC
c = Client('neuprint.janelia.org', dataset = 'hemibrain:v1.1',token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImxpdXRvbnk2NkBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdoNWZBS29QYzQxdzR0S0V1cmFXeEplbm41ZHBoajFrU2tBVS1mVD9zej01MD9zej01MCIsImV4cCI6MTc2NTAwNTUwOH0.itiAhvsvMYHVECWFVMuolEJo64pwSQgt9OLN2npyrys')

h.load_file('import3d.hoc')

class Cell():
	def __init__(self, fname, gid):
		self._gid = gid
		self.fname = fname
		self.load_morphology(fname)
		self.sec_name = ""
		self.find_sec_name()
		#self.discretize_sections()
		#self.add_biophysics()
		self.tree = pd.DataFrame()
		self.body_id = 0

	def __str__(self):
		return 'Cell[{}]'.format(self._gid)

	def load_morphology(self, nom):
		#print(nom)
		cell = h.Import3d_SWC_read()
		cell.input(nom)
		i3d = h.Import3d_GUI(cell, 0)
		i3d.instantiate(self)

	def find_sec_name(self):
		secnames = []
		for sec in h.allsec():
			name = re.split('\.|\[', sec.name())[2]
			if name not in secnames:
				secnames.append(name)
		if 'axon' in secnames:
			self.sec_name = 'axon'
		elif 'soma' in secnames:
			self.sec_name = 'soma'

	def discretize_sections(self):
		''' 
			adds at least as many spatial compartments as d_lambda rule
			maximizing segment density also allows better synapse localization 
		'''
		for sec in h.allsec():
			sec.nseg = sec.n3d()

	def add_biophysics(self, ra, cm, gpas, epas):
		# insert passive density mechanism
		mt = h.MechanismType(0)
		mt.select("pas")
		for section in h.allsec():
			# insert distributed mechanism into section
			mt.make(sec=section)	

		change_R_a(ra)
		change_c_m(cm)
		change_g_pas(gpas)
		change_e_pas(epas)

	def trace_tree(self):
		'''
			create table of all specified 3d points (0 to section.n3d()-1), x, y, z coordinates, 
		    (note, 3d point != segment, but arc3d(point i)/section length does give "x position" (0 to 1) of point)
		    and their associated section number (re.split('\[|\]', cell1.axon[192].name())[3] gives 192)
		'''
		tree = [] # tree is initially a list, dict to DataFrame is fastest to create the pandas DataFrame
		for sec in self.axon:
			num_segs = sec.n3d()
			sec_index = re.split('\[|\]', sec.name())[3]
			'''
			for seg in sec:
				loc = seg.x
				toAppend = {}
				toAppend.update(sec=sec_index, i3d=i, 
								x=loc, y=sec.y3d(i), z=sec.z3d(i), 
								arc=sec.arc3d(i), gd = geodesic_dist)
			'''
			for i in range(num_segs):
				toAppend = {} 	# each row to add is a dictionary
				loc = sec.arc3d(i) / sec.L
				geodesic_dist = eval("h.distance(self.{}[0](0.5), sec(loc))".format(self.sec_name))
				toAppend.update(sec=sec_index, i3d=i, 
								x=sec.x3d(i), y=sec.y3d(i), z=sec.z3d(i), 
								arc=sec.arc3d(i), gd = geodesic_dist)
				tree.append(toAppend)
		tree = pd.DataFrame(tree)
		return tree

	def add_synapses(self, file_path, syn_strength):
		'''
			add Exp2Syn synapses to model, based on xyz synapse locations
			requires the "tree" DataFrame attribute to be populated
		'''
		#print(file_path)
		### import synaptic locations
		conn = pd.read_csv(file_path)
		#conn = conn.drop(columns = ['type', 'partner'])
		num_synapses = conn.shape[0]
		#print("imported " + str(num_synapses) + " synapses")
		if num_synapses == 0:
			return 0, 0, 0, 0

		### KNN to map each synapse x, y, z (scaled x0.008) to the closest segment
		tree_coords = self.tree.loc[:, 'x':'z']
		syn_coords = conn.loc[:, 'x':'z'] / 125
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tree_coords)
		distances, indices = nbrs.kneighbors(syn_coords) 
		# indices: index in tree of closest section and point location to a synapse

		### add synapses onto morphology
		syns = h.List()
		j = 0 # index in syns
		for index in indices:
			sec = int(self.tree.loc[index, 'sec'])
			i3d = self.tree.loc[index, 'i3d']	# the 3d point index on the section
			#print("adding synapse " + str(j) + " to section " + str(sec))
			loc = eval("self.{}[sec].arc3d(i3d) / self.{}[sec].L".format(self.sec_name, self.sec_name))
			# 0 to 1, length along section
			#print("about to append")
			syns.append(h.Exp2Syn(self.axon[sec](loc)))

			### synapse parameters from Tobin et al paper: 
			syns.object(j).tau1 = 0.2 #ms
			syns.object(j).tau2 = 1.1 #ms
			syns.object(j).e = -10 #mV, synaptic reversal potential = -10 mV for acetylcholine? 
			# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125135/
			#syns.object(j).g = 0.0001 #uS default ### seems to have no effect on the result

			h.pop_section() # clear the section stack to avoid overflow (triggered by using ".L" above?)
			j = j + 1

		### use NetStim to activate NetCon
		nc = h.NetStim()
		nc.number = 1
		nc.start = 0
		nc.noise = 0

		ncs = h.List()
		for i in range(len(list(syns))):
			ncs.append(h.NetCon(nc, syns.object(i)))
			ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

		return syns, nc, ncs, num_synapses

	def add_synapses_xyz(self, xyz_locs, syn_strength):
		'''
			add new synapses based on loaded xyz locations
		'''
		num_synapses = xyz_locs.shape[0]
		#print("imported " + str(num_synapses) + " synapses")
		if num_synapses == 0:
			return 0, 0, 0, 0

		### KNN to map each synapse x, y, z (scaled x0.008) to the closest segment
		tree_coords = self.tree.loc[:, 'x':'z']
		syn_coords = xyz_locs.loc[:, 'x_post':'z_post'] / 125
		nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tree_coords)
		distances, indices = nbrs.kneighbors(syn_coords) 
		# indices: index in tree of closest section and point location to a synapse

		### add synapses onto morphology
		syns = h.List()
		j = 0 # index in syns
		for index in indices:
			sec = int(self.tree.loc[index, 'sec'])
			i3d = self.tree.loc[index, 'i3d']	# the 3d point index on the section
			#print("adding synapse " + str(j) + " to section " + str(sec))
			loc = eval("self.{}[sec].arc3d(i3d) / self.{}[sec].L".format(self.sec_name, self.sec_name))
			# 0 to 1, length along section
			#print("about to append")
			syns.append(h.Exp2Syn(self.axon[sec](loc)))

			### synapse parameters from Tobin et al paper: 
			syns.object(j).tau1 = 0.2 #ms
			syns.object(j).tau2 = 1.1 #ms
			syns.object(j).e = -10 #mV, synaptic reversal potential = -10 mV for acetylcholine? 
			# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125135/
			#syns.object(j).g = 0.0001 #uS default ### seems to have no effect on the result

			h.pop_section() # clear the section stack to avoid overflow (triggered by using ".L" above?)
			j = j + 1

		### use NetStim to activate NetCon, initially inactive
		nc = h.NetStim()
		nc.number = 0
		nc.start = 0
		nc.noise = 0

		ncs = h.List()
		for i in range(len(list(syns))):
			ncs.append(h.NetCon(nc, syns.object(i)))
			ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

		return syns, nc, ncs, num_synapses

	def add_synapses_subtree(self, sec_for_subtree, syn_count, syn_strength):
		'''
			add <syn_count> synapses to random sections in the subtree of 
			self.axon[<sec_for_subtree>]
		'''

		# get section numbers in the subtree
		subtree_secs = self.axon[sec_for_subtree].subtree()
		subtree_sec_nums_brack = [str(sec).partition('axon')[2] for sec in subtree_secs]
		subtree_sec_nums = [re.findall("\[(.*?)\]", sec)[0] for sec in subtree_sec_nums_brack] # debracket 

		### add synapses onto morphology
		syns = h.List()
		j = 0
		for index in range(syn_count):

			sec = int(random.choice(subtree_sec_nums))
			loc = random.uniform(0, 1)

			syns.append(h.Exp2Syn(self.axon[sec](loc)))

			### synapse parameters from Tobin et al paper: 
			syns.object(j).tau1 = 0.2 #ms
			syns.object(j).tau2 = 1.1 #ms
			syns.object(j).e = -10 #mV, synaptic reversal potential = -10 mV for acetylcholine? 
			# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3125135/
			#syns.object(j).g = 0.0001 #uS default ### seems to have no effect on the result

			h.pop_section() # clear the section stack to avoid overflow 
			j += 1

		### use NetStim to activate NetCon, initially inactive
		nc = h.NetStim()
		nc.number = 0
		nc.start = 0
		nc.noise = 0

		ncs = h.List()
		for i in range(len(list(syns))):
			ncs.append(h.NetCon(nc, syns.object(i)))
			ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

		num_synapses = syn_count

		return syns, nc, ncs, num_synapses

	def total_length(self):
		total_length = 0
		for sec in self.axon: 
			total_length += sec.L
		return total_length

	def surf_area(self):
		total_area = 0
		for sec in h.allsec():
			for seg in sec:
				total_area += seg.area()
		return total_area

def time_to_percent_peak(t, v, perc):
	'''
		Given a time trace, voltage trace, and a percentage X%, give the closest time in the
		time trace to X% of the voltage's peak amplitude. 
	'''
	base, peak = min(v), max(v)
	peak_loc = np.where(np.array(v) == peak)[0][0] # index of peak location
	perc_of_peak = perc * (peak - base) + base # value of percent of peak amplitude
	if peak_loc == 0: # safety catch for if v trace is all 0, maybe replace with np.nan, then use nansum 
		return 0
	# interpolate t value where v reaches inputted percent of peak amplitude
	time_of_perc = np.interp(perc_of_peak, np.array(v)[0:peak_loc], np.array(t)[0:peak_loc])
	# subsets portion of trace up to the peak
	#ind_of_perc = np.abs(np.array(v)[0:peak_loc] - perc_of_peak).argmin() 
	#time_of_perc = np.array(t)[ind_of_perc]
	return time_of_perc

def modify_EPSP_in_all_conns():
	'''
		20-12-08: single use script to modify 'epsp_exp' column in 20-09-26_all_conns
			save out as a new csv 20-12-08_all_conns
			same as the 9-26 version, but with EPSP values refreshed to Jamie's 10/01 corrections

		update: change this to the revised 12-08 EPSP amplitudes
			save out as overwriting the csv 20-12-08_all_conns
	'''

	prev_conn = pd.read_csv("20-09-26_all_conns.csv")
	#epsp_dir = pd.read_csv("20-10-01_Jamie's refreshed epsp_peaks.csv", index_col = 0)
	epsp_dir = pd.read_csv("20-12-08_Jamie's epsp_peaks, max of avgs.csv", index_col = 0)
	for i in range(prev_conn.shape[0]):
		prev_conn.at[i, 'epsp_exp'] = epsp_dir.loc[prev_conn.iloc[i]['pn'], prev_conn.iloc[i]['lhn']]
	prev_conn.to_csv("20-12-08_all_conns.csv")

# previously was local5 params: epas = -55, cm = 1.2, synstrength = 3.5e-5, ra = 125, gpas = 4.4e-5
# now error peak no skip params
def find_peak_vals_MC(epas = -55, cm = 0.6, synstrength = 5.5e-5, ra = 350, gpas = 5.8e-5, params = 'abs_norm_noskip',
					save_excel = False, show_plot = False, show_scatter = False, show_kinetics_scatter = False,
					conn_file = "20-12-08_all_conns.csv"):
	'''
		Given biophysical parameter inputs, output the resulting simulated EPSPs across the 
		entire population. 

		TODO: change synstrength parameter name to gsyn
	'''

	###
	### START SIMULATION CODE: 
	### INPUTS:
	all_conns = pd.read_csv(conn_file)
	all_conns = all_conns.assign(num_syn=np.zeros(len(all_conns))) 	# num synapses
	all_conns = all_conns.assign(lhn_SA=np.zeros(len(all_conns)))	# surface area

	param_set = params
	e_pas = epas # mV, reversal potential
	c_m = cm #uF/cm^2
	syn_strength = synstrength # uS, peak synaptic conductance
	R_a = ra # ohm-cm
	g_pas = gpas # S/cm^2

	### parameters:
	### biophysical parameters:
	#if param_set == 0:
		### parameters not preloaded
	#	continue 
	if param_set == 1:
		### 8/23: from local5v1
		R_a = 125 # ohm-cm
		g_pas = 4.4e-5 # S/cm^2
		param_print = True # whether to print param values when changing them
	elif param_set == 2:
		### 8/31: small param search, optimize to peak:
		R_a = 300
		g_pas = 1.2e-5
	elif param_set == 3:
		### 8/31: small param search, optimize to total error:
		### don't use, too similar to local5 params
		R_a = 100
		g_pas = 4.e-5
	elif param_set == 4:
		### 9/1: optimize to peak values
		R_a = 350
		g_pas = 1.8e-5
		c_m = 1
	elif param_set == 5:
		### completed 9/10: optimize to peak values, after local5 back to v1.1
		syn_strength = 4.5e-5
		R_a = 350
		g_pas = 1.8e-5
		c_m = 1.2
	elif param_set == 6:
		### 9/12: fit to avgs of connection peak
		c_m = 0.6
		syn_strength = 3.5e-5
		R_a = 350
		g_pas = 3.4e-5

	### run through all PN-LHN instances, simulate unitary EPSP and track peak amplitude 
	sim_traces = []
	for i in range(len(all_conns)):
		curr_trace = {}
		curr_trace.update(lhn = all_conns.lhn[i], pn=all_conns.pn[i])

		swc_path = "swc\\{}-{}.swc".format(all_conns.lhn[i], all_conns.lhn_id[i])
		syn_path = "syn_locs\\{}-{}_{}-{}.csv".format(all_conns.lhn[i], all_conns.lhn_id[i], all_conns.pn[i], all_conns.pn_id[i])

		cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
		cell1.discretize_sections()
		cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
		cell1.tree = cell1.trace_tree()
		synapses, netstim, netcons, num = cell1.add_synapses(syn_path, syn_strength)

		all_conns.loc[i, 'lhn_SA'] = cell1.surf_area()

		if num > 0:
			netstim.number = 1
			netstim.start = 0

			# activate synapses
			h.load_file('stdrun.hoc')
			x = h.cvode.active(True)
			v_s = h.Vector().record(cell1.axon[0](0.5)._ref_v) 		# soma membrane potential
			#v_z = h.Vector().record(p_siz(siz_loc)._ref_v)		# putative SIZ membrane potential
			t = h.Vector().record(h._ref_t)                     # Time stamp vector
			h.finitialize(-55 * mV)
			h.continuerun(40*ms)

			all_conns.loc[i, 'epsp_sim'] = float(max(v_s)+55)
			all_conns.loc[i, 'num_syn'] = num

			# KINETICS:
			# time from 10 to 90% peak:
			t_10to90 = time_to_percent_peak(t, v_s, 0.90) - time_to_percent_peak(t, v_s, 0.10)
			# time from 0.1 to 80% peak:
			t_0to80 = time_to_percent_peak(t, v_s, 0.80) - time_to_percent_peak(t, v_s, 0.0001)

			# TODO: track average transfer impedance to SIZ and average geodesic distance to SIZ
			# perhaps also the stdevs of above

			curr_trace.update(t_sim = t, v_sim = v_s, v_sim_peak = max(v_s)+55, 
							  v_exp_peak = all_conns.epsp_exp[i], 
							  rise_time_10to90 = t_10to90, rise_time_0to80 = t_0to80)
		else:
			all_conns.loc[i, 'epsp_sim'] = 0 # EPSP size = 0 if no synapses
			all_conns.loc[i, 'num_syn'] = 0	

			curr_trace.update(t_sim = [0, 20], v_sim = [-55, -55], v_sim_peak = 0, 
							  v_exp_peak = all_conns.epsp_exp[i],
							  rise_time_10to90 = 0, rise_time_0to80 = 0)

		sim_traces.append(curr_trace)

	### average together traces: find instances of each connection, then average each set of instances
	# values per connection are found from the average trace
	# I verified that the EPSP sim peaks are virtually the same with this method and the old one
	# kinetic predictions should be improved
	sim_avgs_v2 = []
	lhn_list = ['CML2', 'L1', 'L11', 'L12', 'L13', 'L15', 'ML3', 'ML8', 'ML9', 'V2', 'V3', 'local2', 'local5', 'local6']
	pn_list = ['DA4l', 'DC1', 'DL4', 'DL5', 'DM1', 'DM3', 'DM4', 'DP1m', 'VA1v', 'VA2', 'VA4', 'VA6', 
				'VA7l', 'VC1', 'VC2', 'VL2a', 'VL2p']
	# display simulation averages per connection type (not individual traces)
	# then save as svg with param class
	for lhn in lhn_list:
		for pn in pn_list:
			trace_locs = [i for i in range(len(sim_traces)) if sim_traces[i]["lhn"]==lhn and sim_traces[i]["pn"]==pn]

			# average the traces at trace_locs if the LHN-PN pair exists
			if len(trace_locs) > 0:
				t_interp = np.arange(0, 21, 0.05)

				avg_sim = np.zeros(len(t_interp))
				for ind in trace_locs:
					interp_trace = np.interp(t_interp, sim_traces[ind]["t_sim"], [x+55 for x in sim_traces[ind]["v_sim"]])

					avg_sim = [sum(pair) for pair in zip(avg_sim, interp_trace)]

				avg_sim = [val/len(trace_locs) for val in avg_sim]

				# calculate kinetics for avg sim trace
				t_10to90_sim = time_to_percent_peak(t_interp, avg_sim, 0.90) - time_to_percent_peak(t_interp, avg_sim, 0.10)
				t_0to80_sim = time_to_percent_peak(t_interp, avg_sim, 0.80) - time_to_percent_peak(t_interp, avg_sim, 0.0001)

				# calculate kinetics for exp trace
				trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(lhn, pn), header = None, dtype = np.float64)
				t_exp = trace_exp[0]+1.25 # slightly adjust to align with rise time of EPSP
				v_exp = trace_exp[1]
				t_10to90_exp = time_to_percent_peak(t_exp, v_exp, 0.90) - time_to_percent_peak(t_exp, v_exp, 0.10)

				# add LHN, PN, and info about the simulated trace to data table:
				toAppend = {}
				toAppend.update(e_pas = epas, c_m = cm, syn_strength = synstrength,	R_a = ra, g_pas = gpas,
								lhn = lhn, pn = pn, epsp_sim = max(avg_sim),
								epsp_exp = sim_traces[trace_locs[0]]['v_exp_peak'],
								t_sim_10to90 = t_10to90_sim, t_sim_0to80 = t_0to80_sim,
								t_exp_10to90 = t_10to90_exp)
				sim_avgs_v2.append(toAppend)
	sim_avgs = pd.DataFrame(sim_avgs_v2)

	'''
	### old method:
	# values per connection (PN-LHN) are averages from each instance of the connection trace
	conn_indices = {} # keys are tuples of LHN, PN names; values are indices of the connections
	for i in range(len(sim_traces)):
		curr_conn = (sim_traces[i]["lhn"], sim_traces[i]["pn"])
		if curr_conn not in conn_indices:
			conn_indices[curr_conn] = [i]
		else:
			conn_indices[curr_conn].append(i)
	sim_avgs = []
	for conn in conn_indices:
		toAppend = {}
		curr_lhn = conn[0]
		curr_pn = conn[1]
		conn_peaks = [sim_traces[i]["v_sim_peak"] for i in conn_indices[conn]]
		conn_t_10to90 = [sim_traces[i]["rise_time_10to90"] for i in conn_indices[conn]]
		conn_t_0to80 = [sim_traces[i]["rise_time_0to80"] for i in conn_indices[conn]]

		# also pull out experimental 10 to 90 rise time
		trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(curr_lhn, curr_pn), header = None, dtype = np.float64)
		t_exp = trace_exp[0]+1.25 # slightly adjust to align with rise time of EPSP
		v_exp = trace_exp[1]
		t_10to90_exp = time_to_percent_peak(t_exp, v_exp, 0.90) - time_to_percent_peak(t_exp, v_exp, 0.10)

		toAppend.update(lhn = curr_lhn, pn = curr_pn, epsp_sim = np.mean(conn_peaks),
						epsp_exp = sim_traces[conn_indices[conn][0]]["v_exp_peak"], 
						t_sim_10to90 = np.mean(conn_t_10to90), t_sim_0to80 = np.mean(conn_t_0to80),
						t_exp_10to90 = t_10to90_exp)
		sim_avgs.append(toAppend)
	sim_avgs = pd.DataFrame(sim_avgs)
	'''

	# compute normalized RSS error
	sum_peak_err = 0
	for i in range(len(sim_avgs)):
		normalized_resid = (sim_avgs.loc[i, 'epsp_sim'] - sim_avgs.loc[i, 'epsp_exp']) / sim_avgs.loc[i, 'epsp_exp']
		sim_avgs.loc[i, 'resid'] = normalized_resid * sim_avgs.loc[i, 'epsp_exp']
		sim_avgs.loc[i, 'norm_resid'] = normalized_resid
		if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
			sum_peak_err += np.abs(normalized_resid)
		#else:
			#print("skipping {}, {}".format(sim_avgs.loc[i, 'lhn'], sim_avgs.loc[i, 'pn']))
		#sum_peak_err += normalized_resid**2

	if show_plot:
		plot_traces(sim_traces, cm, synstrength, ra, gpas)
	if save_excel:
		all_conns.to_excel('{}_mcsim_params{}_each_inst.xlsx'.format(date.today().strftime("%y-%m-%d"), str(params)))
		sim_avgs.to_excel('{}_mcsim_params{}_avgs.xlsx'.format(date.today().strftime("%y-%m-%d"), str(params)))
	if show_scatter:
		plt.rcParams["figure.figsize"] = (15,15)

		fig, ax = plt.subplots(nrows = 1, ncols = 2)

		#plt.scatter(all_conns.loc[:, 'epsp_exp'], all_conns.loc[:, 'epsp_sim'], color = 'black')
		ax[0].scatter(sim_avgs.loc[:, 'epsp_exp'], sim_avgs.loc[:, 'epsp_sim'], 
					s = 2, color = 'black')
		ax[0].set_xlabel("experimental EPSP peak (mV)")
		ax[0].set_ylabel("simulated EPSP peak (mV)")
		ax[0].plot([0, 7], [0, 7], color = 'grey', ls = '--')
		props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(synstrength*1000, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2, R_a = " + str(ra) + 
			" Ohm-cm")
		#plt.suptitle(props + " [current params]", 
		#		 fontsize = 24, y = 0.96)

		# test out new averaging method for kinetics
		ax[1].scatter(sim_avgs_v2.loc[:, 'epsp_exp'], sim_avgs_v2.loc[:, 'epsp_sim'], 
					s = 2, color = 'black')
		ax[1].set_xlabel("experimental EPSP peak (mV)")
		ax[1].set_ylabel("simulated EPSP peak (mV)")
		ax[1].plot([0, 7], [0, 7], color = 'grey', ls = '--')

		'''
		draw()

		fig.savefig('{}_mcsim_params{}_scatter.svg'.format(date.today().strftime("%y-%m-%d"), str(params)))
		'''
		plt.show()
	### fig_mcmodel_kinetics_predictions
	### scatter of rise time predictions vs actual
	if show_kinetics_scatter:
		plt.rcParams['figure.figsize'] = (2, 2)

		fig, ax = plt.subplots(nrows=1, ncols=2)

		ax[0].scatter(sim_avgs["t_exp_10to90"], sim_avgs["t_sim_10to90"], color = 'blue', s = 2) # size = 2
		ax[0].plot([0, 11], [0, 11], color = 'grey', alpha = 0.3) # unity line
		ax[0].spines["top"].set_visible(False)
		ax[0].spines["right"].set_visible(False)
		ax[0].set_xlabel('experimental rise time (ms)', fontsize = 9)
		ax[0].set_ylabel('simulated rise time (ms)', fontsize = 9)

		# test out new averaging method for kinetics
		ax[1].scatter(sim_avgs_v2["t_exp_10to90"], sim_avgs_v2["t_sim_10to90"], color = 'blue', s = 2)
		ax[1].plot([0, 11], [0, 11], color = 'grey', alpha = 0.3) # unity line
		ax[1].spines["top"].set_visible(False)
		ax[1].spines["right"].set_visible(False)
		ax[1].set_xlabel('experimental rise time (ms)', fontsize = 9)
		ax[1].set_ylabel('simulated rise time (ms)', fontsize = 9)
		'''
		draw()
		fig.savefig("fig_mcmodel_kinetics_predictions.svg")
		'''
		plt.show()

	return sim_traces, sim_avgs, sum_peak_err

def plot_traces(sim_traces, cm, synstrength, ra, gpas):
	'''
		given inputs (to be described), overlay simulated and experimental traces
		in cells of a large grid
	'''

	plt.rcParams["figure.figsize"] = (9,7)

	# 14 LHN by 17 PN plot
	fig, axs = plt.subplots(nrows = 14, ncols = 17, sharex = True, sharey = True)
	lhn_list = ['CML2', 'L1', 'L11', 'L12', 'L13', 'L15', 'ML3', 'ML8', 'ML9', 'V2', 'V3', 'local2', 'local5', 'local6']
	pn_list = ['DA4l', 'DC1', 'DL4', 'DL5', 'DM1', 'DM3', 'DM4', 'DP1m', 'VA1v', 'VA2', 'VA4', 'VA6', 'VA7l', 'VC1', 'VC2', 'VL2a', 'VL2p']
	[ax.set_xlim(0,21) for subrow in axs for ax in subrow]
	[ax.set_ylim(0,7) for subrow in axs for ax in subrow]
	plt.subplots_adjust(wspace=0, hspace=0)
	[axs[0, i].set_title(pn_list[i]) for i in range(len(pn_list))]
	[axs[i, 0].set_ylabel(lhn_list[i]) for i in range(len(lhn_list))]
	[ax.set_frame_on(False) for subrow in axs for ax in subrow]


	avg_sim_traces = pd.DataFrame({'t': np.arange(0, 21, 0.1)})
	# display simulation averages per connection type (not individual traces)
	# then save as svg with param class
	for lhn in lhn_list:
		for pn in pn_list:
			trace_locs = [i for i in range(len(sim_traces)) if sim_traces[i]["lhn"]==lhn and sim_traces[i]["pn"]==pn]

			# average the traces at trace_locs
			if len(trace_locs) > 0:
				t_interp = np.arange(0, 21, 0.1)

				avg_trace = np.zeros(len(t_interp))
				for ind in trace_locs:
					interp_trace = np.interp(t_interp, sim_traces[ind]["t_sim"], [x+55 for x in sim_traces[ind]["v_sim"]])

					avg_trace = [sum(pair) for pair in zip(avg_trace, interp_trace)]

				avg_trace = [val/len(trace_locs) for val in avg_trace]

				### plot simulated traces in proper grid location
				row = lhn_list.index(lhn)
				col = pn_list.index(pn)
				axs[row, col].plot(t_interp, avg_trace, color = 'green', lw = 0.8) # plot 

				# save avg s
				avg_sim_traces['{}_{}_sim'.format(lhn, pn)] = avg_trace
	avg_sim_traces.to_csv('figdata_avg_sim_traces_mc.csv')


	for i in range(len(sim_traces)):

		### plot simulated and experimental traces
		row = lhn_list.index(sim_traces[i]["lhn"])
		col = pn_list.index(sim_traces[i]["pn"])
		# read & plot experimental trace
		trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(sim_traces[i]["lhn"], sim_traces[i]["pn"]), header = None, dtype = np.float64)
		t_exp = trace_exp[0]+1.25 # slightly adjust to align with rise time of EPSP
		v_exp = trace_exp[1]
		axs[row, col].plot(t_exp, v_exp, color = 'red', lw = 0.8)

		axs[row, col].plot(sim_traces[i]["t_sim"], [x+55 for x in sim_traces[i]["v_sim"]], color = 'grey', alpha = 0.2, lw = 0.4)

	props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(synstrength*1000, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2, R_a = " + str(ra) + 
			" Ohm-cm")
	plt.suptitle(props + " [current params]", 
				 fontsize = 15, y = 0.96)

	draw()
	fig.savefig('fig_pop_mc_traces.svg')

	plt.show()

###
### t1, v1 the simulated trace, t2, v2 the experimental trace to fit to
### weight error from VL2a x3 to normalize amplitudes
###
def find_error(t1, v1, t2, v2):

	# normalize both traces by peak of experimental trace
	v1_norm = (v1+55) / max(v2+55) 
	v2_norm = (v2+55) / max(v2+55)

	peak_err = np.abs(max(v2_norm) - max(v1_norm))

	trace_range = int(np.floor(max(t2)))
	t_discr = range(1, trace_range) # discretize time blocks

	v1_interp = np.interp(t_discr, t1, v1_norm) # sim
	v2_interp = np.interp(t_discr, t2, v2_norm) # exp

	# mean of absolute value of difference
	trace_err = np.mean(np.abs(v2_interp - v1_interp))

	return peak_err, trace_err

def param_search_MC(conn_file = "20-12-08_all_conns.csv"):
	'''
		conn_file: the file containing list of LHN and PN body IDs and EPSP sizes
					was 20-08-27_all_conns.csv prior to final body ID refresh on 20-09-26
					after EPSP refresh is now 20-12-08_all_conns.csv
	'''

	all_conns = pd.read_csv(conn_file)

	start_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')

	e_pas = -55 # mV

	# note: for other run tiling other part of g_pas:
	# need to use the refreshed version of all_conns, then re-run the below eventually too

	# 20-12-08 after refreshing EPSP peaks, local6 morphologies -- hopefully first of final
	# still likely need to tile other part of g_pas
	# 12 * 4 * 13 * 16 = 9984
	syn_strength_s = np.arange(2.5e-5, 8.1e-5, 0.5e-5)
	c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
	g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
	R_a_s = np.arange(75, 451, 25) # ohm-cm 

	# different types of errors evaluated (over all connections) per paramater set
	err_per_paramset = []
	# EPSP amplitude and kinetics per connection, for each parameter set
	sim_per_conn_per_paramset = pd.DataFrame()

	# iterate through all biophysical parameter combinations
	for syn_strength_i in syn_strength_s:
		for c_m_i in c_m_s:
			for g_pas_i in g_pas_s:
				for R_a_i in R_a_s:

					### following errors skip local5/vl2a
					abs_norm_skip = 0 # sum of absolute values of normalized residuals
					abs_nonorm_skip = 0 # sum of absolute values of non-normalized residuals
					squ_norm_skip = 0 # sum of squares of normalized residuals
					
					### doesn't skip local5/vl2a
					abs_norm_noskip = 0 # sum of absolute values of normalized residuals
					abs_nonorm_noskip = 0 # sum of absolute values of non-normalized residuals
					squ_norm_noskip = 0 # sum of squares of normalized residuals
					
					sim_traces, sim_avgs, rss_err = find_peak_vals_MC(cm = c_m_i, ra = R_a_i, synstrength = syn_strength_i,
														  gpas = g_pas_i, save_excel = False, show_scatter = False)

					for i in range(len(sim_avgs)):
						normalized_resid = (sim_avgs.loc[i, 'epsp_sim'] - sim_avgs.loc[i, 'epsp_exp']) / sim_avgs.loc[i, 'epsp_exp']
						
						abs_norm_noskip += np.abs(normalized_resid) # normalized residual
						abs_nonorm_noskip += np.abs(normalized_resid * sim_avgs.loc[i, 'epsp_exp']) # non-normalized residual
						squ_norm_noskip += normalized_resid**2 # squared normalized residual

						if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
							abs_norm_skip += np.abs(normalized_resid) # normalized residual
						if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
							abs_nonorm_skip += np.abs(normalized_resid * sim_avgs.loc[i, 'epsp_exp']) # non-normalized residual
						if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
							squ_norm_skip += normalized_resid**2 # squared normalized residual

					# save parameter values, (output trace indices), fit errors
					params_toAppend = {}
					params_toAppend.update(g_syn = syn_strength_i, g_pas = g_pas_i, R_a = R_a_i, 
									c_m = c_m_i,
									err_abs_norm_skip = abs_norm_skip, err_abs_norm_noskip = abs_norm_noskip,
									err_abs_nonorm_skip = abs_nonorm_skip,
									err_squ_norm_skip = squ_norm_skip, 
									err_squ_norm_noskip = squ_norm_noskip,
									err_abs_nonorm_noskip = abs_nonorm_noskip)
					
					# save overall error summed over all connection for this parameter set
					err_per_paramset.append(params_toAppend)
					# save EPSP prediction per connection
					if sim_per_conn_per_paramset.empty:
						sim_per_conn_per_paramset = sim_avgs
					else:
						sim_per_conn_per_paramset = sim_per_conn_per_paramset.append(sim_avgs)

					# update a CSV every 2000 parameters
					if len(err_per_paramset) % 2000 == 1:
						pd.DataFrame(err_per_paramset).to_csv('{}_err_per_{}paramsets_temp.csv'.format(datetime.today().strftime("%y-%m-%d"), str(len(err_per_paramset))))
						sim_per_conn_per_paramset.to_csv('{}_sim_per_conn_{}paramsets_temp.csv'.format(datetime.today().strftime("%y-%m-%d"), str(len(err_per_paramset))))

				#print("finished running " + str(str(round(g_pas_i, 6))) + " S/cm^2")

	err_per_paramset = pd.DataFrame(err_per_paramset)

	err_per_paramset.to_csv('{}_err_per_{}paramsets.csv'.format(datetime.today().strftime("%y-%m-%d"), str(len(err_per_paramset))))
	sim_per_conn_per_paramset.to_csv('{}_sim_per_conn_{}paramsets.csv'.format(datetime.today().strftime("%y-%m-%d"), str(len(err_per_paramset))))

	end_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')
	print("start time: {}, end time: {}".format(start_time, end_time))

	return err_per_paramset, sim_per_conn_per_paramset

def update_sim_per_conn(sim_per_conn_path = 'fit_outs\\20-12-27_MC_sim_per_conn_9216+9984.csv',
						modified_targets = ['local6', 'V2'],
						conn_file = "20-12-08_all_conns.csv"):
	''' given a few modified SWCs for postsynaptic cells, re-run a partial set of body IDs for ALL params and 
		update the sim_per_conn spreadsheet, then generate new best params
	'''

	all_conns = pd.read_csv(conn_file)
	prev_sim_per_conn = pd.read_csv(sim_per_conn_path)

	start_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')

	e_pas_i = -55 # mV

	syn_strength_s = [2.5e-5]
	c_m_s = [0.6]
	g_pas_s = [1.2e-5] # S/cm^2, round to 6 places
	R_a_s = [75] # ohm-cm 

	# 20-12-08 after refreshing EPSP peaks, local6 morphologies -- hopefully first of final
	# still likely need to tile other part of g_pas
	# 12 * 4 * 13 * 16 = 9984
	#syn_strength_s = np.arange(2.5e-5, 8.1e-5, 0.5e-5)
	#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
	#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
	#R_a_s = np.arange(75, 451, 25) # ohm-cm 

	# different types of errors evaluated (over all connections) per paramater set
	err_per_paramset = []
	# EPSP amplitude and kinetics per connection, for each parameter set
	sim_per_conn_per_paramset = pd.DataFrame()

	param_iter = 0
	# iterate through all biophysical parameter combinations
	for syn_strength_i in syn_strength_s:
		for c_m_i in c_m_s:
			for g_pas_i in g_pas_s:
				for R_a_i in R_a_s:

					# only simulate peaks for target cell body IDs in modified_cells
					for cell_type in modified_targets:
						cell_ids = [cell_id for cell_name, cell_id in zip(all_conns.lhn, all_conns.lhn_id) 
											if cell_type in cell_name]

						# find PNs giving input to the LHN
						input_type_ids = all_conns.loc[all_conns.lhn == cell_type][['pn', 'pn_id']].drop_duplicates()

						# iterate through inputs onto the LHN (may only be 1)
						for i, type_id_row in input_type_ids.iterrows(): 
							input_type, input_id = type_id_row['pn'], type_id_row['pn_id']
							print(cell_type, input_type)
							print(all_conns.loc[(all_conns.lhn == cell_type) & \
										(all_conns.pn == input_type), 'epsp_exp'])
							epsp_exp = all_conns.loc[(all_conns.lhn == cell_type) & \
										(all_conns.pn == input_type), 'epsp_exp'].iloc[0] # all EPSPs for a conn equal

							uEPSP_outputs = []
							for cell_id in cell_ids: 
								# simulate the EPSP, kinetics for each target body ID and the corresponding input
								uEPSP_output = sim_uEPSP(target = cell_type, target_id = cell_id, 
														 upstream = input_type, upstream_id = input_id,
														 R_a = R_a_i, c_m = c_m_i, g_pas = g_pas_i, 
														 e_pas = e_pas_i, syn_strength = syn_strength_i)
								uEPSP_outputs.append(uEPSP_output)
							uEPSP_outputs = pd.DataFrame(uEPSP_outputs)

							epsp_sim, t_sim_10to90, t_sim_0to80 = mean(uEPSP_outputs.EPSP_sim), \
											mean(uEPSP_outputs.t_sim_10to90), mean(uEPSP_outputs.t_sim_0to80)
							resid, norm_resid = epsp_sim - epsp_exp, (epsp_sim - epsp_exp) / epsp_exp

							print('uEPSP amp: ', [epsp_sim, t_sim_10to90, t_sim_0to80, resid, norm_resid])
							# find appropriate row to update in MC_sim_per_conn
							prev_sim_per_conn.loc[(prev_sim_per_conn.R_a==R_a_i) & (prev_sim_per_conn.g_pas==g_pas_i) \
								& (prev_sim_per_conn.c_m==c_m_i) & (prev_sim_per_conn.syn_strength==syn_strength_i) \
								& (prev_sim_per_conn.lhn==cell_type) & (prev_sim_per_conn.pn==input_type), \
								['epsp_sim', 't_sim_10to90', 't_sim_0to80', 'resid', 'norm_resid']] \
								= [epsp_sim, t_sim_10to90, t_sim_0to80, resid, norm_resid]

					param_iter += 1

					if param_iter % 2000 == 1:
						prev_sim_per_conn.to_csv(f'{datetime.today().strftime("%y-%m-%d")}_sim_per_conn_{param_iter}paramsets_temp.csv')

	prev_sim_per_conn.to_csv(f'{datetime.today().strftime("%y-%m-%d")}_sim_per_conn_newlocal6_V2_{param_iter}.csv')	

	end_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')
	print("start time: {}, end time: {}".format(start_time, end_time))
	
	return err_per_paramset, sim_per_conn_per_paramset

def sim_uEPSP(target, target_id, upstream, upstream_id,
				R_a, c_m, g_pas, e_pas, syn_strength):
	'''	return uEPSP attributes given a target (postsynaptic) and upstream (presynaptic) cell
	'''
	swc_path = "swc\\{}-{}.swc".format(target, str(target_id))
	syn_path = "syn_locs\\{}-{}_{}-{}.csv".format(target, str(target_id), upstream, str(upstream_id))

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()
	synapses, netstim, netcons, num_syn = cell1.add_synapses(syn_path, syn_strength)

	surf_area = cell1.surf_area()

	if num_syn > 0:
		netstim.number = 1
		netstim.start = 0

		# activate synapses
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)
		v_trace_soma = h.Vector().record(cell1.axon[0](0.5)._ref_v) 		# soma membrane potential
		#v_z = h.Vector().record(p_siz(siz_loc)._ref_v)		# putative SIZ membrane potential
		t_trace = h.Vector().record(h._ref_t)                     # Time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(40*ms)
		
		EPSP_sim = float(max(v_trace_soma)+55)

		# KINETICS:
		# time from 10 to 90% peak:
		t_10to90 = time_to_percent_peak(t_trace, v_trace_soma, 0.90) - time_to_percent_peak(t_trace, v_trace_soma, 0.10)
		# time from 0.1 to 80% peak:
		t_0to80 = time_to_percent_peak(t_trace, v_trace_soma, 0.80) - time_to_percent_peak(t_trace, v_trace_soma, 0.0001)

		# TODO: track average transfer impedance to SIZ and average geodesic distance to SIZ
		# perhaps also the stdevs of above
	else:
		EPSP_sim, t_trace, v_trace_soma = 0, [0, 20], [-55, -55]
		t_10to90, t_0to80 = None, None

	uEPSP_output = {'EPSP_sim': EPSP_sim, 'num_syn': num_syn, 't_sim_10to90': t_10to90, 
					't_sim_0to80': t_0to80, 't_trace': t_trace, 'v_trace_soma': v_trace_soma}
	return uEPSP_output

def analyze_fits():
	'''
		given a csv of errors per parameter set, for 
		SCv1, SCv2, and MC fits (each fit should use same error types).
		display for each error type the best param sets across fit types:
			1) scatter of simulated vs. exp. EPSP peaks across population
			2) [TODO] a bar chart of errors per connection
	'''

	# file paths of error per param CSVs, for the different model types
	err_per_param_csvs = {
		'SCv1': 'fit_outs//20-12-11_SCv1_err_per_6174paramsets.csv',
		'SCv2': 'fit_outs//20-12-11_SCv2_err_per_6174paramsets.csv',
		'MC': 'fit_outs//20-12-15_MC_err_per_9984+9216paramsets_raw.csv'
		# MC: added 20-12-21 9216 params, 9984 being re-run with slight modified EPSPs
	}

	# read in error tables
	err_per_params = {}
	for model_type, csv in err_per_param_csvs.items():
		err_per_params[model_type] = pd.read_csv(csv, index_col = False)

	# find the list of error types
	err_types = [col for col in err_per_params['MC'].columns if 'err' in col]

	# create a sim vs exp EPSP scatter for each error type
	for err_type in err_types:
		fig, ax = plt.subplots(nrows = 1, ncols = 1)

		# for each model type, plot scatter of EPSP values
		for model_type, err_per_param in err_per_params.items():
			print(f'plotting simulated EPSPs for {model_type}')
			best_param_set_ind = err_per_param[err_type].idxmin()
			best_param_set = {}
			if 'SC' in model_type:
				best_param_set['g_syn'] = round(err_per_param.iloc[best_param_set_ind]['g_syn'], 3)
				best_param_set['g_pas'] = round(err_per_param.iloc[best_param_set_ind]['g_pas'], 6)
				best_param_set['c_m'] = round(err_per_param.iloc[best_param_set_ind]['c_m'], 1)

				sim_traces, sim_avgs, sum_peak_err = find_peak_vals_SC(version = int(model_type[-1]), 
					gpas = best_param_set['g_pas'], cm = best_param_set['c_m'], gsyn = best_param_set['g_syn'])

				ax.scatter(sim_avgs.loc[:, 'epsp_exp'], sim_avgs.loc[:, 'epsp_sim'], s = 3.5,
							label = f"{model_type}: g_syn={best_param_set['g_syn']} nS, \n"\
									f"g_pas={best_param_set['g_pas']} S/cm^2, \nc_m={best_param_set['c_m']} \u03BCF/cm^2")
			elif 'MC' in model_type:
				best_param_set['g_syn'] = round(err_per_param.iloc[best_param_set_ind]['g_syn'], 6)
				best_param_set['g_pas'] = round(err_per_param.iloc[best_param_set_ind]['g_pas'], 6)
				best_param_set['c_m'] = round(err_per_param.iloc[best_param_set_ind]['c_m'], 1)
				best_param_set['R_a'] = round(err_per_param.iloc[best_param_set_ind]['R_a'], 1)

				sim_traces, sim_avgs, sum_peak_err = find_peak_vals_MC(ra = best_param_set['R_a'],
					gpas = best_param_set['g_pas'], cm = best_param_set['c_m'], synstrength = best_param_set['g_syn'],
					params = err_type)

				ax.scatter(sim_avgs.loc[:, 'epsp_exp'], sim_avgs.loc[:, 'epsp_sim'], s = 3.5,
							label = f"{model_type}: g_syn={round(best_param_set['g_syn']*1000,3)} nS, \n"\
									f"g_pas={best_param_set['g_pas']} S/cm^2, \nc_m={best_param_set['c_m']} \u03BCF/cm^2, "\
									f"\nR_a={best_param_set['R_a']} \u03A9m-cm")

		# add axis labels, horizontal line
		ax.legend(loc = 'center left', prop={'size': 8}, bbox_to_anchor = (1.01, 0.5), borderaxespad = 0)
		plt.subplots_adjust(right=0.75) # give room to legend on right
		ax.plot([0, 7], [0, 7], color = 'grey', ls = '--', alpha = 0.5)
		ax.set_xlabel("experimental EPSP peak (mV)")
		ax.set_ylabel("simulated EPSP peak (mV)")
		ax.set_title(f'error type: {err_type}')
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		plt.savefig(f'fit_comp_{err_type}.png', format = 'png', bbox_inches='tight', dpi = 300)
		#plt.show()

def shelve_all_resids():
	toshelve = ['all_resids']
	shelf_name = '20-09-27_mcfit_all_resids.out'
	shelf = shelve.open(shelf_name, 'n')
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

def transf_imped_of_inputs(down, down_id, up, up_id, siz_sec, siz_seg, transf_freq = 20, toPlot = False):
	'''
		after initiating a downstream cell and synapses onto the cell, 
		calculate the mean transfer impedance between the synaptic locations
		and a given section number of the cell 

		from docs for Impedance.transfer()
			The value returned can be thought of as either |v(loc)/i(x)| or |v(x)/i(loc)| 
			Probably the more useful way of thinking about it is to assume a current stimulus of 1nA 
			injected at x and the voltage in mV recorded at loc
	'''
	swc_path = "swc\\{}-{}.swc".format(down, down_id)
	syn_path = "syn_locs\\{}-{}_{}-{}.csv".format(down, down_id, up, up_id)

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()
	synapses, netstim, netcons, num = cell1.add_synapses(syn_path, syn_strength)

	# set up Impedance measurement class
	imp = h.Impedance()
	imp.loc(siz_seg, sec = cell1.axon[siz_sec])
	imp.compute(transf_freq)	# starts computing transfer impedance @ freq 

	syn_info = []
	# iterate through synapses
	for syn in synapses:
		# find Z_c from synapse to siz_loc AND distance between the points, append to list
		curr_loc = syn.get_segment()
		curr_transf_imp = imp.transfer(curr_loc)
		curr_distance = h.distance(cell1.axon[siz_sec](siz_seg), curr_loc)

		# convert to dictionary input with interpretable outputs
		toAppend = {}
		toAppend.update(dist_to_siz = curr_distance, Zc_to_siz = curr_transf_imp)
		syn_info.append(toAppend)

	# plot synapse to SIZ distance vs transfer impedance
	if toPlot:
		plt.scatter([val[dist_to_siz] for val in syn_info], [val[Zc_to_siz] for val in syn_info])
		plt.xlabel('synapse to SIZ (um)')
		plt.ylabel('transfer impedance (MOhm)')
		plt.show()

	return syn_info
# example code for above method:
d, d_id, u, u_id, s_sec, s_seg = 'local5', '5813105722', 'VA6', '1881751117', 996, 0
#transf_imped_of_inputs(down = d, down_id = d_id, up = u, up_id = u_id, 
#						siz_sec = s_sec, siz_seg = s_seg, transf_freq = 20)

def instantiate_lhns():
	'''
		code I copy pasted into the command line to manually
		look at the transfer impedance / other electrotonic measures
		in the GUI for two example LHNs
	'''

	# population fit biophysical parameters
	R_a = 350 # Ohm-cm
	c_m = 0.6 # uF/cm^2
	g_pas = 5.8e-5 # S/cm^2
	e_pas = -55 # mV
	syn_strength = 5.5e-5 # uS

	# change to path for hemibrain DM1 
	swc_path = "swc\\ML9-542634516.swc"
	# swc_path = "swc\\KCs (pasha)\\KCa'b'-ap1-487834111.swc"
	# swc_path = "swc\\KCs (pasha)\\KCab-s-331662717.swc"
	# swc_path = "swc\\KCs (pasha)\\KCg-m-354775482.swc"
	# swc_path = "swc\\L12-391609333.swc"
	# swc_path = "swc\\local5-5813105722.swc"

	# local5 params
	R_a = 375 # ohm-cm ### NOTE: tripling R_a from 125
	c_m = 1.2 # uF/cm^2
	g_pas = 4.4e-5 # S/cm^2 
	e_pas = -55 # mV
	syn_strength = 3.5e-5 # uS, peak synaptic conductance

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	syn_path = "syn_locs\\ML9-542634516_DM1-542634818.csv"

	synapses, netstim, netcons, num = cell1.add_synapses(syn_path, syn_strength)

def find_input_attrs(target_name = 'ML9', target_body_id = 542634516, weight_threshold = 10, 
								siz_sec = 569, siz_seg = 0.01, transf_freq = 20, 
								axon_sec = 609, axon_seg = 0.58,
								toPlot = False,
								param_set = 'pop_fit'):
	'''
		given the name and body ID of an existing LHN (which has a skeleton in the swc folder)
		instantiate Cell and use hemibrain to add all synapses above a threshold weight
		return DataFrame with EPSP and impedance attributes about each unique connection type
	'''
	try:
		swc_path = "swc\\{}-{}.swc".format(target_name, str(target_body_id))
	except:
		print('no SWC found')
	### add neuprint call if the SWC doesn't exist inside

	# get swc straight from neuprint:
	#skel = fetch_skeleton(body = target_body_id, format = 'swc') # import skeleton

	if param_set == 'pop_fit':
		# biophysical parameters from our fits
		R_a = 350
		c_m = 0.6
		g_pas = 5.8e-5
		e_pas = -55 # one parameter left same in both, we used -55 in our fits
		syn_strength = 5.5e-5 # uS
	elif param_set == 'retr_local5_fit':
		R_a = 375 # ohm-cm # x3 from 125
		c_m = 1.2 # uF/cm^2
		g_pas = 4.4e-5 # S/cm^2 
		e_pas = -55 # mV
		syn_strength = 3.5e-5 # uS, peak synaptic conductance

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	conns = fetch_simple_connections(upstream_criteria = None, downstream_criteria = target_body_id, min_weight = weight_threshold)
	
	# get number of post-synapses on the target neuron
	try:
		target, r = fetch_neurons(target_body_id)
		target_syn_count = target.post[0]
	except:
		print('likely no internet connection')

	### instantiate synapses for each connection with weight > threshold
	all_conns = []
	for pre_name in set(conns.type_pre):
		# find all body IDs for this presynaptic neuron type
		pre_bodyIds = [conns.bodyId_pre[ind] for ind in range(len(conns.type_pre)) if conns.type_pre[ind] == pre_name]

		# get all synapse xyz locations for the body IDs in this neuron type (may be just 1 body ID)
		syn_locs = pd.DataFrame(columns = ['x_post', 'y_post', 'z_post'])
		for pre_id in pre_bodyIds:
			curr_syn_locs = fetch_synapse_connections(source_criteria = pre_id, target_criteria = target_body_id)
			syn_locs = syn_locs.append(curr_syn_locs[['x_post', 'y_post', 'z_post']])

		# add synapses onto skeleton
		curr_syns, netstim, netcons, num = cell1.add_synapses_xyz(xyz_locs = syn_locs, syn_strength = syn_strength)

		print('adding {} synapses from {} to {}'.format(str(num), pre_name, target_name))

		# measure uEPSP for connection at pSIZ and distal axon
		# activate the stim
		netstim.number = 1
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)
		v_siz = h.Vector().record(cell1.axon[siz_sec](siz_seg)._ref_v)
		v_axon = h.Vector().record(cell1.axon[axon_sec](axon_seg)._ref_v)
		if target_name == 'ML9' and target_body_id == 542634516:
			# for some reason this ML9's axon[0](1) is at the primary branch point
			v_soma = h.Vector().record(cell1.soma[0](0.5)._ref_v)	
		else:
			v_soma = h.Vector().record(cell1.axon[0](0.5)._ref_v)
		t = h.Vector().record(h._ref_t)                     				# Time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(40*ms)
		if toPlot:
			plt.plot(list(t), list(v_siz), label = 'siz')
			plt.plot(list(t), list(v_axon), label = 'axon')
			plt.plot(list(t), list(v_soma), label = 'soma')
			plt.legend(loc = 'upper right')
			plt.show()
		netstim.number = 0

		# measure rise time of EPSP at pSIZ
		t_10to90_siz = time_to_percent_peak(t, v_siz, 0.90) - time_to_percent_peak(t, v_siz, 0.10)

		toAppend = {}
		toAppend.update(post_name = target_name, post_id = target_body_id,
							pre_name = pre_name, pre_id = str(pre_bodyIds)[1:-1],
							syns = curr_syns, syn_count = len(curr_syns),
							syn_budget = len(curr_syns) / target_syn_count,
							num_instances = len(pre_bodyIds), stim = [netstim], 
							uEPSP_siz = max(list(v_siz))+55, uEPSP_axon = max(list(v_axon))+55, 
							uEPSP_soma = max(list(v_soma))+55,
							t_10to90_siz = t_10to90_siz)
		all_conns.append(toAppend)

	# set up Impedance measurement class
	imp = h.Impedance()
	imp.loc(siz_seg, sec = cell1.axon[siz_sec])
	imp.compute(transf_freq)	# starts computing transfer impedance @ freq 

	### iterate through all connections and measure impedances
	for conn in all_conns:
		curr_syns = conn['syns']

		# iterate through each synapse in the connection
		syn_info = []
		for syn in curr_syns:
			# find Z_c = transfer impedance from synapse to siz_loc 
			curr_transf_imp = imp.transfer(syn.get_segment())
			# find Z_i = input impedance at synapse
			curr_input_imp = imp.input(syn.get_segment())
			# find distance from synapse to siz_loc
			curr_distance = h.distance(cell1.axon[siz_sec](siz_seg), syn.get_segment())
			# find voltage transfer ratio from synapse to siz_loc
			curr_transf_ratio = imp.ratio(syn.get_segment())

			# record individual synapse info
			toAppend = {}
			toAppend.update(dist_to_siz = curr_distance, Zc_to_siz = curr_transf_imp, 
							Zi = curr_input_imp, V_ratio = curr_transf_ratio)
			syn_info.append(toAppend)

		# update 'conn'
		conn.update(mean_dist_to_siz = mean([val['dist_to_siz'] for val in syn_info]),
						mean_Zc_to_siz = mean([val['Zc_to_siz'] for val in syn_info]),
						mean_Zi = mean([val['Zi'] for val in syn_info]),
						mean_V_ratio = mean([val['V_ratio'] for val in syn_info]))

		if toPlot:
			plt.scatter([val['dist_to_siz'] for val in syn_info], [val['Zc_to_siz'] for val in syn_info], 
						label = "{} w/ {} synapses".format(conn['pre_name'], str(conn['syn_count'])),
						alpha = 0.2)

	if toPlot:
		# plot synapse to SIZ distance vs transfer impedance
		plt.legend(loc = 'upper right')
		plt.xlabel('distance, synapse to SIZ (um)')
		plt.ylabel('transfer impedance (MOhm)')
		plt.title('inputs onto {} {}'.format(target_name, str(target_body_id)))
		#plt.show()

		plt.rcParams["figure.figsize"] = (10,10)
		all_conns = pd.DataFrame(all_conns)
		subset = pd.DataFrame(all_conns[['syn_count', 'uEPSP_siz', 'uEPSP_axon', 't_10to90_siz', 'mean_dist_to_siz', 'mean_Zc_to_siz', 'mean_Zi', 'mean_V_ratio']])
		sns.set_theme(style="ticks")
		sns.set(font_scale = 0.5)
		g = sns.pairplot(subset, height = 1, aspect = 0.8, corner = True)
		g.savefig('{}_{}_scatter_matrix.svg'.format(target_name, str(target_body_id)))

	return all_conns

def conn_attrs(target_neuron_file = 'LHN_list_siz_axon_locs.csv', weight_threshold = 3, transf_freq = 20,
					param_set = 'pop_fit'):
	'''
		for each neuron in a list (i.e. a list of LHNs), find information about its
		input connections, such as EPSP size, impedance measures, synapse counts, etc.

		possible inputs: target_neuron_file = 'KC_list_siz_axon_locs.csv' # list of ~23 hand picked KCs
						target_neuron_file = 'LHN_list_siz_axon_locs.csv' # list of our experimental-matched LHNs
	'''
	nrns = pd.read_csv(target_neuron_file)

	# iterate through each target neuron, concatenate relevant file info
	nrns_input_attrs = pd.DataFrame()
	for i in range(nrns.shape[0]):
		curr_input_attrs = find_input_attrs(target_name = nrns.iloc[i].lhn, target_body_id = nrns.iloc[i].lhn_id,
												weight_threshold = weight_threshold, transf_freq = transf_freq,
												siz_sec=nrns.iloc[i].siz_sec, siz_seg = nrns.iloc[i].siz_seg,
												axon_sec=nrns.iloc[i].axon_sec, axon_seg = nrns.iloc[i].axon_seg,
												param_set = param_set)

		nrns_input_attrs = nrns_input_attrs.append(curr_input_attrs)

	return nrns_input_attrs

### TODO 11/30: try above using different parameter sets and see if it decreases KC uEPSPs @ soma
### 
### update KC swc files and KC_list.csv file with Pasha's updates
### perhaps try soma[0](0.5) rather than axon[0](0.5) for soma recording? 
### focus on shuffling analysis since we might not end up using KC analysis
### redo shuffling analysis for a few more ePN inputs onto LHN
### redo shuffling expanding to more classes of inputs as target locs

### should also test how well the retr local5 params do at predicting the population

def analyze_attrs(n):
	'''
		some analysis code for generating graphs about correlations among
		attributes for each connection
	'''
	import pandas as pd

	### analyze LHN data
	n = pd.read_csv('conn_attrs_all_LHNs.csv')
	n_pn = n.loc[(n['pre_name'].str.contains('adPN')) | (n['pre_name'].str.contains('lPN'))] # excitatory PN inputs
	n_pn_out = n_pn.loc[~(n_pn['post_name'].str.contains('local'))]	# excitatory PN inputs onto LHONs
	n_pn_local = n_pn.loc[(n_pn['post_name'].str.contains('local'))] # excitatory PN inputs onto LHLNs

	plt.scatter(n_pn_out.syn_count, n_pn_out.mean_Zc_to_siz, color = 'red', label = 'inputs to LHONs')
	plt.scatter(n_pn_local.syn_count, n_pn_local.mean_Zc_to_siz, color = 'blue', label = 'inputs to LHLNs')
	plt.legend(loc='upper right')

	plt.xlabel('number of synapses in connection')
	plt.ylabel('avg. transfer impedance from synapses to SIZ (MOhm)')
	plt.show()

	# plot connected points for geodesic distance vs Z_c (or perhaps uEPSP/# synapses/budget?)
	for lhn_id in set(n_pn_out.post_id):
	    lhn = n_pn_out.query('post_id == @lhn_id')['post_name'].iloc[0]
	    plt.plot(n_pn_out.query('post_id==@lhn_id')['mean_dist_to_siz'], n_pn_out.query('post_id==@lhn_id')['mean_Zc_to_siz'], label = '{}-{}'.format(lhn, str(lhn_id)))

	# pairplot scatter matrix
	sub_n = pd.DataFrame(n[['syn_count', 'uEPSP_siz', 'uEPSP_axon', 't_10to90_siz', 'mean_dist_to_siz', 'mean_Zc_to_siz', 'mean_Zi', 'mean_V_ratio']])
	sns.set_theme(style="ticks")
	sns.set(font_scale = 0.6)
	g_n = sns.pairplot(sub_n, height= 1, aspect = 0.8, corner = True, plot_kws = {"s":3})
	g_n.savefig('all_LHNs_scatter_mat.svg')

	### analyze KC data
	n = pd.read_csv('conn_attrs_some_KCs.csv')
	n_pn = n.loc[(n['pre_name'].str.contains('adPN')) | (n['pre_name'].str.contains('lPN'))] # excitatory PN inputs
	# plot transf impedance vs syn count of each KC, can also do EPSP at SIZ / anything else
	for kc_id in set(n_pn.post_id):
		kc = n_pn.query('post_id == @kc_id')['post_name'].iloc[0]
		plt.plot(n_pn.query('post_id==@kc_id')['syn_count'], n_pn.query('post_id==@kc_id')['mean_Zc_to_siz'], 
					label = "{}-{}".format(kc, str(kc_id)))
	plt.legend(loc = 'upper right')

def shuffle_syn_locs_by_class(target_name = 'ML9', target_body_id = 542634516, weight_threshold = 3, 
								input_names = ['DP1m_adPN', 'DM4_adPN'], 
								siz_sec = 569, siz_seg = 0.0153, transf_freq = 20, 
								axon_sec = 609, axon_seg = 0.58,
								toPlot = False,
								conn_class = ['adPN', 'lPN'],
								run_count = 2):
	'''
		given a downstream (target_name) neuron, an upstream (conn_class) neuron class, and 
		one representative of that class (input_name), repeatedly shuffle the synaptic locations of
		that one representative, using the synapse locations of other neurons of that
		class as potential shuffle locations. 
		weight_threshold = 0: thus, candidate synapse locations can come from a connection type in conn_class
			with as little as this # of synapses

		TODO: in this class potentially add info on clustering within a connection and relative to other
				connections within the class
		
		generate: histogram of possible uEPSP amplitudes for each shuffle, with the baseline
			uEPSP size marked with a vertical line
		return: list of simulated uEPSP values at shuffled synapse locations
		potentially: will need param: input_body_id = 635062078 (ML9)
	'''
	print('shuffling inputs onto {} {}'.format(target_name, str(target_body_id)))

	# instantiate target (post-synaptic) cell
	try:
		swc_path = "swc\\{}-{}.swc".format(target_name, str(target_body_id))
	except:
		print('no SWC found')
	# biophysical parameters from our fits
	R_a = 350
	c_m = 0.6
	g_pas = 5.8e-5
	e_pas = -55 # one parameter left same in both, we used -55 in our fits
	syn_strength = 5.5e-5 # uS

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	# get number of post-synapses on the target neuron
	target = None
	print('finding # of post-synapses')
	while target is None:
		try:
			target, r = fetch_neurons(target_body_id)
		except:
			print('server access failure, repeating')
			pass
	target_syn_count = target.post[0]

	# identify all potential synaptic sites for connections in class (xyz coordinates)
	# could consider a Regex search here:
	# fetch_simple_connections(upstream_criteria = NC(type='.*adPN.*', regex = True), downstream_criteria = target_body_id, min_weight = 0)
	conns = None
	print('finding names of inputs')
	while conns is None:
		try:
			conns = fetch_simple_connections(upstream_criteria = None, downstream_criteria = target_body_id, min_weight = weight_threshold)
		except:
			print('server access failure, repeating')
			pass
	# find names of input neurons which are of the 'to place' input class
	if 'all_dendritic' in conn_class:
		# add all neuron types which have uEPSP at the SIZ bigger than uEPSP at the axon
		# equivalent, roughly, to all neuron types with uEPSPs 
		conn_attrs = pd.read_csv('conn_attrs_all_LHNs_with_smallInputs.csv')
		conn_attrs = conn_attrs.loc[(conn_attrs['post_name']==target_name) & (conn_attrs['post_id']==target_body_id)]
		nrns_in_class = []
		for row_ind in range(conn_attrs.shape[0]):
			if conn_attrs.iloc[row_ind]['uEPSP_siz'] > conn_attrs.iloc[row_ind]['uEPSP_axon']:
				nrns_in_class.append(conn_attrs.iloc[row_ind]['pre_name'])
		print("neurons in the all_dendritic class: {}".format(str(nrns_in_class)))
	else:
		# add all neuron types which have 
		nrns_in_class = [pre_name for pre_name in list(filter(None, conns.type_pre)) if any(class_marker in pre_name for class_marker in conn_class)]
		print("neurons in the \'to place\' class: {}".format(str(nrns_in_class)))

	# iterate through neuron types in 'to place' class and add their synapse locations
	potent_syn_locs = pd.DataFrame(columns = ['type_pre', 'bodyId_pre', 'x_post', 'y_post', 'z_post'])
	# treat each unique connection type as one:
	for nrn_name in set(nrns_in_class):
		# find all body IDs for this presynaptic neuron type
		pre_bodyIds = [conns.bodyId_pre[ind] for ind in range(len(conns.type_pre)) if conns.type_pre[ind] == nrn_name]

		# get all synapse xyz locations for the body IDs in this neuron type (may be just 1 body ID)
		nrn_syn_count = 0
		curr_syn_locs = None
		# neuprint sometimes throws 502 server errors, try to catch them:
		print('retrieving postsynapse locations for {}'.format(nrn_name))
		num_failures = 0
		while curr_syn_locs is None:
			try:
				curr_syn_locs = fetch_synapse_connections(source_criteria = pre_bodyIds, target_criteria = target_body_id)
			except:
				print('server access failure, repeating')
				num_failures += 1
				pass
			if num_failures > 100:
				break
		if num_failures > 100:
			continue
		curr_syn_locs = curr_syn_locs[['bodyId_pre', 'x_post', 'y_post', 'z_post']]
		curr_syn_locs = curr_syn_locs.assign(type_pre = nrn_name)
		potent_syn_locs = potent_syn_locs.append(curr_syn_locs)
		nrn_syn_count += curr_syn_locs.shape[0]

		print('added pot. syn locs from type {}, {} insts., {} total syns'.format(nrn_name, str(len(pre_bodyIds)), 
																					str(nrn_syn_count)))

	# KNN to map each potential synapse location x, y, z (scaled x0.008) to the closest segment
	tree_coords = cell1.tree.loc[:, 'x':'z']
	syn_coords = potent_syn_locs.loc[:, 'x_post':'z_post'] / 125
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tree_coords)
	distances, indices = nbrs.kneighbors(syn_coords)
	# indices: index in tree of closest section and point location to a synapse

	# produce dictionary of shuffle data for each input type:
	# keys: string of input connection type; 
	# value: [base_attrs dictionary, run_attrs dataframe (row per shuffle)]
	shuffle_results = {}
	for input_name in input_names:

		# instantiate sites of baseline specified input connection onto skeleton
		input_syn_locs = potent_syn_locs.loc[potent_syn_locs.type_pre == input_name]
		input_bodyIds = [conns.bodyId_pre[ind] for ind in range(len(conns.type_pre)) if conns.type_pre[ind] == input_name]
		curr_syns, netstim, netcons, num = cell1.add_synapses_xyz(xyz_locs = input_syn_locs, syn_strength = syn_strength)
		print('adding {} synapses from {} to {}'.format(str(num), input_name, target_name))

		# measure uEPSP for connection at pSIZ and distal axon and soma
		netstim.number = 1 # activate stim
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)
		v_siz = h.Vector().record(cell1.axon[siz_sec](siz_seg)._ref_v)
		v_axon = h.Vector().record(cell1.axon[axon_sec](axon_seg)._ref_v)
		if target_name == 'ML9' and target_body_id == 542634516:
			# for some reason this ML9's axon[0](1) is at the primary branch point
			v_soma = h.Vector().record(cell1.soma[0](0.5)._ref_v)	
		else:
			v_soma = h.Vector().record(cell1.axon[0](0.5)._ref_v)
		t = h.Vector().record(h._ref_t)                     				# Time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(40*ms)
		if toPlot:
			plt.plot(list(t), list(v_siz), label = 'siz')
			plt.plot(list(t), list(v_axon), label = 'axon')
			plt.plot(list(t), list(v_soma), label = 'soma')
			plt.legend(loc = 'upper right')
			plt.show()
		netstim.number = 0 # de-activate stim
		# measure rise time of EPSP at pSIZ
		t_10to90_siz = time_to_percent_peak(t, v_siz, 0.90) - time_to_percent_peak(t, v_siz, 0.10)

		# save attributes of baseline input connection
		base_input_attrs = {}
		base_input_attrs.update(post_name = target_name, post_id = target_body_id,
							pre_name = input_name, pre_id = str(input_bodyIds)[1:-1],
							syns = curr_syns, syn_count = len(curr_syns),
							syn_budget = len(curr_syns) / target_syn_count,
							num_instances = len(input_bodyIds), stim = [netstim], 
							uEPSP_siz = max(list(v_siz))+55, uEPSP_axon = max(list(v_axon))+55, 
							uEPSP_soma = max(list(v_soma))+55,
							t_10to90_siz = t_10to90_siz)

		# repeatedly permute synapses to other potential locations, record new connection attributes
		print('commence {} shuffles for input {}'.format(str(run_count), input_name))
		run_attrs = []
		for i in range(run_count):
			# locations (rows in potent_syn_locs) to permute each synapse to
			permute_locs = random.sample(range(potent_syn_locs.shape[0]), len(curr_syns))

			# for each synapse, move it to the new location
			original_loc = []
			for j in range(len(curr_syns)):
				# save current location
				original_loc.append(curr_syns.object(j).get_segment())

				# find section and segment info of new shuffle location
				new_tree_ind = indices[permute_locs[j]]
				sec = int(cell1.tree.loc[new_tree_ind, 'sec'])
				i3d = cell1.tree.loc[new_tree_ind, 'i3d']	# the 3d point index on the section
				loc = cell1.axon[sec].arc3d(i3d) / cell1.axon[sec].L

				# move synapse to new location
				curr_syns.object(j).loc(cell1.axon[sec](loc))
				h.pop_section()

			# for the new synapse distribution
			# record geodesic distribution / input impedance distribution / transfer imp distr

			# simulate EPSP
			# measure uEPSP for connection at pSIZ and distal axon and soma
			netstim.number = 1 # activate stim
			h.load_file('stdrun.hoc')
			x = h.cvode.active(True)
			v_siz = h.Vector().record(cell1.axon[siz_sec](siz_seg)._ref_v)
			v_axon = h.Vector().record(cell1.axon[axon_sec](axon_seg)._ref_v)
			if target_name == 'ML9' and target_body_id == 542634516:
				# for some reason this ML9's axon[0](1) is at the primary branch point
				v_soma = h.Vector().record(cell1.soma[0](0.5)._ref_v)	
			else:
				v_soma = h.Vector().record(cell1.axon[0](0.5)._ref_v)
			t = h.Vector().record(h._ref_t)                     				# Time stamp vector
			h.finitialize(-55 * mV)
			h.continuerun(40*ms)
			netstim.number = 0 # de-activate stim
			# measure rise time of EPSP at pSIZ
			t_10to90_siz = time_to_percent_peak(t, v_siz, 0.90) - time_to_percent_peak(t, v_siz, 0.10)

			# save attributes of permuted input connection
			toAppend = {}
			toAppend.update(permute_ind = i, 
							uEPSP_siz = max(list(v_siz))+55, uEPSP_axon = max(list(v_axon))+55, 
							uEPSP_soma = max(list(v_soma))+55)
			run_attrs.append(toAppend)

			if i % 100 == 1:
				print('permutation {}: uEPSP @ SIZ = {}'.format(str(i), str(toAppend['uEPSP_siz'])))

			# reset synaptic locations back to their old locations
			for j in range(len(curr_syns)):
				curr_syns.object(j).loc(original_loc[j])

		run_attrs = pd.DataFrame(run_attrs)

		# plot histogram of uEPSP (at SIZ, soma) sizes
		#plt.hist()

		# plot overlay of various geodesic, Z_input, Z_c histograms

		shuffle_results[input_name] = [base_input_attrs, run_attrs]
		
	return shuffle_results

def test_shuffle_count():
	'''
		test number of runs needed to converge on stable uEPSP distribution when doing
		multiple runs of shuffling
	'''

	run_lengths = [2, 4, 6, 8]

	shuffle_run = {}

	for rc in run_lengths:
		b, r = shuffle_syn_locs_by_class(run_count = rc)
		shuffle_run[rc] = r

	fig, axs = plt.subplots(nrows = 2, ncols = 2)
	axs = axs.reshape(-1)	# allows us to linearly iterate through subplots
	i = 0
	for key, val in shuffle_run.items():
		axs[i].hist(val.uEPSP_siz, bins = 20)
		axs[i].set_title('shuffle synapses {} times'.format(str(key)))
		axs[i].axvline(b['uEPSP_siz'], color = 'red', linestyle = 'dashed')
		i += 1
	axs[0].set_ylabel('frequency')
	axs[-1].set_xlabel('uEPSP @ SIZ (mV)')
	plt.show()

	return shuffle_run

def fig_shuffle_example():
	'''L1 483716037 example shuffle histogram
		just has one PN input 
		NOTE: to plot a different PN-LHN input, chance siz and axon locations'''
	shuffles = shuffle_syn_locs_by_class(input_names = ['DP1m_adPN'], run_count = 500,
											target_name = 'L1', target_body_id = 483716037, 
											siz_sec=10, siz_seg = 0.996269,
											axon_sec=183, axon_seg = 0.5,
											conn_class = ['adPN', 'lPN'],
											weight_threshold = 3)

	fig, axs = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True, figsize = (1.5,1.5))
	for key, val in shuffles.items():
		axs.hist(val[1].uEPSP_siz, bins = 30)
		#axs.set_title('{}, {}s{}i.'.format(str(key), str(val[0]['syn_count']), str(val[0]['num_instances'])))
		axs.axvline(val[0]['uEPSP_siz'], color = 'red', linestyle = 'dashed', label = 'baseline')
	axs.set(ylabel = 'frequency', xlabel = 'uEPSP at SIZ (mV)')
	axs.spines['top'].set_visible(False), axs.spines['right'].set_visible(False)
	axs.legend(frameon = False, prop = {'size':7},
				bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           		ncol=2, mode="expand", borderaxespad=0.)
	#fig.subplots_adjust(wspace=0, hspace=0)
	plt.savefig('figs\\example_shuffHist_L1.svg', format = 'svg')
	#axs.set_title('{}, {}s{}i.'.format(str(key), str(val[0]['syn_count']), str(val[0]['num_instances'])))
	#plt.suptitle('target: {} {}, s=synapses, i=instances'.format(target_name, str(target_body_id)))
	return axs, shuffles

# shuffling hypotheses:
# - is it mainly small inputs that have uEPSPs << mean shuffled uEPSPs? (bc less synapses=more clustering)
# - for these small inputs, is there something systematic about their Strahler order (i.e. very low)
#		moreover, are the inputs spread across many distal branches (i.e. not coincidentally low ordered)
def shuffle_inputs_on_targets(target_neuron_file = 'LHN_list_siz_axon_locs.csv', run_count = 500, 
							to_shuffle_weight_threshold = 10, to_place_weight_threshold = 3,
							to_shuffle_class = ['adPN', 'lPN'], to_place_class = ['all_dendritic']):
	'''
		NOTE: if running for different to_shuffle or to_place classes, change title of output CSV

		for each target neuron in a list (i.e. a list of LHNs):
			find instances within specified classes of inputs = "to shuffle" (with >threshold synapses), 
			shuffle their synapse locations using another specified 
			set of potential input class synapse locations = "to place" (>threshold synapses) 
			using shuffle_syn_locs method
		params: to_shuffle_class = ['adPN', 'lPN'] for excitatory PNs
								 = ['adPN', 'lPN', 'vPN'] if including inhibitory PNs
				to_place_class	 = ['all_dendritic'] then use results of conn_attrs to filter targets by 
										whether EPSP_SIZ > EPSP_axon, i.e. they are dendritic targeting
										NOTE: would likely include non-PN inputs
								 = ['adPN', 'lPN'] to use ePN locations as targets to potentially shuffle to

		if an input onto the target also happens to have multiple other instances of itself (i.e. sister PNs)
			synapsing onto the target, its instantiation (i.e. when synapses are added onto the target) will include
			the weaker, sub-weight threshold connections
			even if multiple instances of a connection type have a total synapse count >threshold, if each 
			individual instance is <threshold than it won't be identified as a 'to_shuffle' neuron
	'''
	start_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')

	nrns = pd.read_csv(target_neuron_file)

	all_shuffle_data = {}
	# for each target neuron in our list, shuffle its to_shuffle_class inputs
	for i in range(nrns.shape[0]):
		target_name = nrns.iloc[i].lhn
		target_body_id = nrns.iloc[i].lhn_id
		if to_place_class == 'all_dendritic' and 'local' in target_name:
			continue
		# find input neurons:
		conns = None
		print('preparing to find input neurons for {} {}'.format(target_name, str(target_body_id)))
		while conns is None:
			try:
				conns = fetch_simple_connections(upstream_criteria = None, downstream_criteria = target_body_id, 
											min_weight = to_shuffle_weight_threshold)
			except:
				print('server access failure, repeating')
				pass
		# filter input neurons by whether they are in to_shuffle_class: 
		ePN_inputs = [pre for pre in list(filter(None, conns.type_pre)) if any(class_type in pre for class_type in to_shuffle_class)]
		print(str(ePN_inputs))
		ePN_inputs = list(set(ePN_inputs)) # treat each input "type" as simultaneously active (i.e. sister PNs)
		print('# sig. inputs: {}, # rows: {}'.format(str(len(ePN_inputs)), 
						str(int(np.ceil(len(ePN_inputs)/3)))))
		# for each input neuron, function shuffle its synapses using to_place_class as candidate locations 
		# returns dictionary w/ keys as input neuron names, values as [base_attrs dictionary, run_attrs dataframe (row per shuffle)]
		if len(ePN_inputs) > 0:
			shuffles = shuffle_syn_locs_by_class(input_names = ePN_inputs, run_count = run_count,
												target_name = target_name, target_body_id = target_body_id, 
												siz_sec=nrns.iloc[i].siz_sec, siz_seg = nrns.iloc[i].siz_seg,
												axon_sec=nrns.iloc[i].axon_sec, axon_seg = nrns.iloc[i].axon_seg,
												conn_class = to_place_class,
												weight_threshold = to_place_weight_threshold)
		
			plt.rcParams["figure.figsize"] = (6,3)
			fig, axs = plt.subplots(nrows = int(np.ceil(len(ePN_inputs)/3)), ncols = 3,
										constrained_layout = True)
			axs = axs.reshape(-1)	# allows us to linearly iterate through subplots
			i = 0
			for key, val in shuffles.items():
				axs[i].hist(val[1].uEPSP_siz, bins = 30)
				axs[i].set_title('{}, {}s{}i.'.format(str(key), str(val[0]['syn_count']), str(val[0]['num_instances'])))
				axs[i].axvline(val[0]['uEPSP_siz'], color = 'red', linestyle = 'dashed')
				i += 1
			axs[0].set_ylabel('frequency')
			axs[-1].set_xlabel('uEPSP @ SIZ (mV)')
			#plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
			plt.suptitle('target: {} {}, s=synapses, i=instances'.format(target_name, str(target_body_id)))
			#plt.show()
			#plt.tight_layout()
			plt.savefig('shuffles\\{}-{}.png'.format(target_name, str(target_body_id)), 
							format = 'png', bbox_inches='tight', dpi = 300)
		else:
			print('{} {}: no inputs w/ > threshold syns to shuffle'.format(target_name, str(target_body_id)))

		all_shuffle_data[(target_name, target_body_id)] = shuffles

	# save out some tabular data
	base_v_shuff_EPSP = []
	all_shuffles = [[lhn_info, shuffle_info] for lhn_info, shuffle_info in all_shuffle_data.items()]
	for lhn in all_shuffles:
		for pn, shuff in lhn[1].items():
			lhn_name, lhn_id = lhn[0][0], lhn[0][1]
			pn_name = pn
			base_EPSP = shuff[0]['uEPSP_siz']
			shuff_EPSP_med = np.median(shuff[1].uEPSP_siz)
			shuff_EPSP_mea = np.mean(shuff[1].uEPSP_siz)
			shuff_EPSP_std = np.std(shuff[1].uEPSP_siz)
			syn_count = shuff[0]['syn_count']
			num_instances = shuff[0]['num_instances']
			toA = {}
			toA.update(lhn_name = lhn_name, lhn_id = lhn_id, pn_name = pn_name, base_EPSP = base_EPSP, 
						shuff_EPSP_med = shuff_EPSP_med, shuff_EPSP_mea = shuff_EPSP_mea, 
						shuff_EPSP_std = shuff_EPSP_std, 
						syn_count = syn_count, num_instances = num_instances)
			base_v_shuff_EPSP.append(toA)
	base_v_shuff_EPSP = pd.DataFrame(base_v_shuff_EPSP)

	# base_v_shuff.to_csv('20-12-04_shuff_ePN2ePN_LHN_750.csv')

	end_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')
	print("start time: {}, end time: {}".format(start_time, end_time))

	return all_shuffle_data, base_v_shuff_EPSP

def analyze_shuffs():
	'''
		some code to analyze shuffling EPSPs
	'''

	b = pd.read_csv('20-12-06_shuff_ePN2ePNiPN_LHN_1000.csv')
	b_out = b.loc[~b['lhn_name'].str.contains('local')]
	c = pd.read_csv('20-12-04_shuff_ePN2ePN_LHN_750.csv')
	c_out = c.loc[~c['lhn_name'].str.contains('local')]

	# scatter of median shuffled vs baseline

	# read the local5 section Jamie wrote, and all the other things he wrote and sent, incl. the transfer resistance 
	# thing on slack
	# start working on that figure

	# visualize z_scored amount of 
	#plt.scatter(b_out.syn_count, (b_out.base_EPSP - b_out.shuff_EPSP_med) / b_out.shuff_EPSP_std)

	# plot z-score (and best fit lines) of baseline vs shuffled EPSPs
	# for ePN to just ePN vs ePN to ePN+iPN on LHONs
	z= np.polyfit(b_out.syn_count, (b_out.base_EPSP - b_out.shuff_EPSP_med) / b_out.shuff_EPSP_std, 1)
	p = np.poly1d(z)
	plt.plot(b_out.syn_count, p(b_out.syn_count), c = 'tab:blue', ls = 'dashed', label = 'target ePN+iPN fit')
	plt.scatter(b_out.syn_count, (b_out.base_EPSP - b_out.shuff_EPSP_med) / b_out.shuff_EPSP_std, c = 'tab:blue', label = 'target ePN+iPN', s = 2.5)

	z= np.polyfit(c_out.syn_count, (c_out.base_EPSP - c_out.shuff_EPSP_med) / c_out.shuff_EPSP_std, 1)
	p = np.poly1d(z)
	plt.plot(c_out.syn_count, p(c_out.syn_count), c = 'tab:orange', ls = 'dashed', label = 'target ePN fit')
	plt.scatter(c_out.syn_count, (c_out.base_EPSP - c_out.shuff_EPSP_med) / c_out.shuff_EPSP_std, c = 'tab:orange', label = 'target ePN', s = 2.5)

	plt.axhline(2, color = 'red', linestyle = 'dashed')
	plt.axhline(-2, color = 'red', linestyle = 'dashed')

	plt.legend(loc = 'upper right')
	plt.xlabel('synapse count')
	plt.ylabel('z-score of baseline EPSP vs shuffled EPSPs')

def add_hh(downstream_of):
	# use to visualize the subtree of a particular section, i.e.
	# when trying to identify the dendritic proximal section
	# then go into Distributed Mech -> Manager -> HomogeneousMech -> hh
	global m
	m = h.MechanismType(0)
	m.select('hh')
	for sec in cell1.axon[downstream_of].subtree():
		m.make(sec=sec)
def remove_hh():
	for sec in h.allsec():
		m.remove(sec=sec)

def assign_LHLN_branch_points():
	'''read in Jamie's labelled Point Nodes and map onto NEURON swc nodes'''

	JJ_labels = pd.read_csv('axonDendriteNodes LHLN 20210112.csv')

	lhn_branch_points = {}	
	for i, row in JJ_labels.iterrows():
		if not pd.isna(row['axon start']):
			name, body_id = row['target_name'], row['target_body_id']
			a, d, a_1, d_1 = row['axon start'], row['dendrite start'], row['axon first branch'], row['dendrite first branch']

			lhn_branch_points[(name, body_id)] = assign_SWC_PointNo_to_NEURON_tree(target_name = name, 
													target_body_id = body_id, nodes_to_map = [d,d_1,a,a_1])

	# dictionary: key -- (lhln name, lhln id), value -- list of lists, each contained 
	# list structure -- [node id, section, segment]; dendrite, dendrite branch out, axon, axon branch out
	return lhn_branch_points

def assign_SWC_PointNo_to_NEURON_tree(target_name = 'local6', target_body_id = 417186656,
									nodes_to_map = [241,1737,341,1745]):

	try:
		swc_path = "swc\\{}-{}.swc".format(target_name, str(target_body_id))
	except:
		print('no SWC found')

	# get swc text file
	headers = ['PointNo', 'Label', 'X', 'Y', 'Z', 'Radius', 'Parent']
	raw_swc = pd.read_csv(swc_path, sep=' ', skiprows=4, names=headers)

	# instantiate NEURON cell object
	cell1, curr_syns, netstim, netcons, num = visualize_inputs(target_name=target_name, target_body_id=target_body_id)

	# map rows from text file to section+segments in NEURON object
	tree_coords = cell1.tree.loc[:, 'x':'z']
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tree_coords)
	pointno_to_sec = [] # values of [PointNo, NEURON section, NEURON segment]
	for node in nodes_to_map:
		node_coords = raw_swc.loc[raw_swc.PointNo==node][['X','Y','Z']]
		distances, indices = nbrs.kneighbors(node_coords)	# index is closest row in cell1.tree to PointNo
		node_section = int(cell1.tree.iloc[indices[0][0]]['sec'])
		node_i3d = int(cell1.tree.iloc[indices[0][0]]['i3d'])
		node_segment = cell1.axon[node_section].arc3d(node_i3d) / cell1.axon[node_section].L
		print(f'PointNo {str(node)} maps to section {str(node_section)} w/ dist {str(distances[0][0])}')
		
		pointno_to_sec.append([node, node_section, node_segment])

	return pointno_to_sec

def visualize_inputs(target_name = 'V2', target_body_id = 852302504, input_name = None,
						syn_locs = None):
	'''
		given a downstream (target_name) neuron + ID, an upstream neuron, instantiate synapses
		(potentially pulling from neuprint) for the sake of visualization

		i.e. go to ModelView in the GUI to see the morphology, synapse locations, etc.
		EX: target_name = 'CML2', target_body_id = 572988717, input_name = 'VA6_adPN'
			target_name = 'CML2', target_body_id = 698180486, input_name = 'VA6_adPN'
			target_name = 'CML2', target_body_id = 696795331, input_name = 'VA6_adPN'
			target_name = 'ML9', target_body_id = 542634516, input_name = 'DP1m_adPN'
			target_name = 'ML9', target_body_id = 573329304, input_name = 'DM1_lPN'
			target_name = 'ML9', target_body_id = 573337611, input_name = 'DM1_lPN'
			target_name = 'L12', target_body_id = 452664348, input_name = 'VA1d_adPN' # 16+7 synapses
			target_name = 'L12', target_body_id = 421957711, input_name = 'DP1m_adPN'
			target_name = 'L12', target_body_id = 603681826, input_name = 'DP1m_adPN'
			target_name = 'L11', target_body_id = 360578457, input_name = 'DM1_lPN' # 18 
			target_name = 'L11', target_body_id = 297921527, input_name = 'DM1_lPN'
			target_name = 'L11', target_body_id = 572988605, input_name = 'DM1_lPN'
			target_name = 'L13', target_body_id = 544007573, input_name = 'VA2_adPN'
			target_name = 'L13', target_body_id = 793702856, input_name = 'VA2_adPN'
			target_name = 'L15', target_body_id = 422307542, input_name = 'DC1_adPN'
			target_name = 'L15', target_body_id = 5813009429, input_name = 'DC1_adPN'
			target_name = 'V2', target_body_id = 1037510115, input_name = 'VL2a_adPN'
			target_name = 'V2', target_body_id = 5813016204, input_name = 'VL2a_adPN'
			target_name = 'V2', target_body_id = 852302504, input_name = 'VL2a_adPN'
			target_name = 'V3', target_body_id = 883338122, input_name = 'VL2a_adPN'
			target_name = 'V3', target_body_id = 917450071, input_name = 'VL2a_adPN'
			target_name = 'ML3', target_body_id = 483017681, input_name = 'VL2p_adPN'
			target_name = 'ML3', target_body_id = 543321179, input_name = 'VL2p_adPN'
			target_name = 'ML3', target_body_id = 573683438, input_name = 'VL2p_adPN'
			target_name = 'CML2', target_body_id = 572988717, input_name = None # NO VA6 synapses
			target_name = 'ML8', target_body_id = 548872750, input_name = 'DM1_lPN' # synapse between 2 tufts
			target_name = 'ML8', target_body_id = 5813089504, input_name = 'DM1_lPN'
			target_name = 'ML8', target_body_id = 571666400, input_name = 'DM1_lPN'
			target_name = 'L1', target_body_id = 575806223, input_name = 'DM1_lPN'
			target_name = 'L1', target_body_id = 483716037, input_name = 'DM1_lPN'
	'''
	global cell1, curr_syns, netstim, netcons, num

	print(f'importing {target_name} {target_body_id}')

	# instantiate target (post-synaptic) cell
	try:
		swc_path = "swc\\{}-{}.swc".format(target_name, str(target_body_id))
	except:
		print('no SWC found')
	# biophysical parameters from our fits
	R_a = 350
	c_m = 0.6
	g_pas = 5.8e-5
	e_pas = -55 # one parameter left same in both, we used -55 in our fits
	syn_strength = 5.5e-5 # uS

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	# get number of post-synapses on the target neuron
	try:
		target, r = fetch_neurons(target_body_id)
		target_syn_count = target.post[0]
	except:
		print('likely no internet connection')

	# add all input synaptic site locations
	if input_name:
		if not syn_locs.empty:
			pass # synapse locs are pre-loaded
		else:
			conns = fetch_simple_connections(upstream_criteria = input_name, downstream_criteria = target_body_id)
			pre_bodyIds = conns.bodyId_pre
			syn_locs = fetch_synapse_connections(source_criteria = pre_bodyIds, target_criteria = target_body_id)
		curr_syns, netstim, netcons, num = cell1.add_synapses_xyz(xyz_locs = syn_locs, syn_strength = syn_strength)
		print('adding {} synapses from {} to {}; budget = {}'.format(str(num), input_name, target_name, str(num/target_syn_count)))
		if target_name == 'local5' and target_body_id == 5813105722:
			# dendrite initial section is axon[195] for un-retraced local5
			num_in_dendr = 0
			for syn in curr_syns:
				if str(syn.get_segment()).partition('(')[0] in [str(val) for val in cell1.axon[195].subtree()]:
					num_in_dendr += 1
			print('proportion synapses in dendrite: {}'.format(str(num_in_dendr/num)))
	else:
		print('not adding synapses')
		curr_syns, netstim, netcons, num = None, None, None, None

	# access a random section
	h.load_file('stdrun.hoc')
	x = h.cvode.active(True)
	v_siz = h.Vector().record(cell1.axon[0](0.5)._ref_v)

	return cell1, curr_syns, netstim, netcons, num

def visualize_custom_var(prox_sec, custom_var):
	''' plot custom variable within dummy variable 
		then can superimpose custom variable onto ShapePlot
		prox_sec = collect all impedance attrs downstream of this section
		custom_var = which variable to ShapePlot
			transfer / ratio / input
	'''
	global inps, ratios, transf, gd 	# for plotting

	m = h.MechanismType(0)
	m.select('var')
	for sec in h.allsec():
		m.make(sec=sec)
	# measure impedance
	imp = h.Impedance()
	imp.loc(0.5, sec= cell1.axon[prox_sec])
	imp.compute(0)

	# assign custom variable (i.e. input impedance) to the dummy variable
	for sec in h.allsec():
		for seg in sec:
			seg.zin_var = eval(f'imp.{custom_var}(seg)') # mystery is why zin_var is the attribute (not `var`)
	# then open Shape Plot and visualize! can also set it up w pyplot

	inps, ratios, transf, gd = [], [], [], []
	# assign input impedance to the dummy variable
	for sec in cell1.axon[prox_sec].subtree():
		for seg in sec:
			inps.append(imp.input(seg))
			ratios.append(imp.ratio(seg))
			gd.append(h.distance(cell1.axon[prox_sec](0.5), seg))
			transf.append(imp.transfer(seg))

	return inps, ratios, transf, gd

def fig_dendrite_linearity():
	'''plotting sub-figures for case study of impedance properties'''
	# fig_5 plot: (dendritic linearity) L1 horizontal plot
	fig, axs = plt.subplots(1,3, figsize=(6, 1.5), sharey=True)
	plt.rcParams.update({'font.size': 9})
	rock = sns.color_palette('rocket_r', as_cmap=True)
	axs[0].scatter(inps, gd, c=inps, s=1, cmap=rock, vmin=0, vmax = 4000)
	axs[0].set(xlim=[0,4500], xlabel = 'input resistance (MOhm)', ylabel = 'distance from pSIZ (\u03BCm)')
	#axs[0].set_xlabel(fontsize = 9), axs[0].set_ylabel(fontsize = 9)
	axs[1].scatter(ratios, gd, c=ratios, s=1, cmap=rock, vmin=0, vmax=1)
	axs[1].set(xlim=[0,1], xlabel = 'voltage transfer ratio')
	axs[2].scatter(transf, gd, c=transf, s=1, cmap=rock, vmin=0, vmax=1700)
	axs[2].set(xlim=[0,1600], xlabel = 'transfer resistance (MOhm)')
	for i in range(3):
	    axs[i].spines["top"].set_visible(False), axs[i].spines["right"].set_visible(False)
	for ax in [axs[0], axs[1], axs[2]]:
	    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
	        item.set_fontsize(9)
	plt.savefig('L1_dendr_imp_measures.svg', format = 'svg')

def fig_local5_linearity():
	### passive norm plots for local5
	custom_var = 'input'
	#visualize_inputs(target_name = 'local5', target_body_id = 5813105722, input_name = None)
	# visualize_custom_var returns inps, ratios, transf, gd
	i_d, r_d, t_d, g_d = visualize_custom_var(prox_sec = 195, custom_var = custom_var) # dendrite 
	i_a, r_a, t_a, g_a = visualize_custom_var(prox_sec = 996, custom_var = custom_var) # axons
	#fig_imp_props_vs_gd(i_d, r_d, t_d, g_d,
	#					i_a, r_a, t_a, g_a,
	#					target_name = 'local5', i_range=[0,3500], t_range=[0,800])
	#plt.show() # to generate and save the figure

	g_d, i_d, t_d, r_d, branchout_dist_d = subtree_imp_props(target_name, target_body_id, 195, 0.327,
												940, 0)
	g_a, i_a, t_a, r_a, branchout_dist_a = subtree_imp_props(target_name, target_body_id, 996, 0,
												1003, 0)

	### TODO: fit lines to local5

	fig, axs = plt.subplots(1,1)
	axs.scatter([-1*g for g in g_d], t_d, c=t_d, s=scatter_size, cmap=rock, vmin=t_range[0], vmax = t_range[1])
	axs.scatter(g_a, t_a, c=t_a, s=scatter_size, cmap=rock, vmin=t_range[0], vmax = t_range[1])
	axs.set(ylim=t_range, ylabel = 'transfer resistance (M\u03A9)', xlabel = 'distance from pSIZ (\u03BCm)')


def fig_imp_props_vs_gd(i_d, r_d, t_d, g_d,
						i_a, r_a, t_a, g_a,
						target_name = 'L1', i_range=[0,4000], t_range=[0,1700]):
	'''
		fig 5 (dendritic linearity) L1 vertical plot:
		
	'''
	#  run after running visualize_custom_var to collect lists
	fig, axs = plt.subplots(3,1, figsize=(2, 6), sharex=True)
	plt.rcParams.update({'font.size': 9})
	rock = sns.color_palette('rocket_r', as_cmap=True)
	scatter_size = 0.3
	axs[0].scatter([-1*g for g in g_d], i_d, c=i_d, s=scatter_size, cmap=rock, vmin=i_range[0], vmax = i_range[1])
	axs[0].scatter(g_a, i_a, c=i_a, s=scatter_size, cmap=rock, vmin=i_range[0], vmax = i_range[1])
	axs[0].set(ylim=i_range, ylabel = 'input resistance (M\u03A9)')
	#axs[0].set_xlabel(fontsize = 9), axs[0].set_ylabel(fontsize = 9)
	axs[1].scatter([-1*g for g in g_d], r_d, c=r_d, s=scatter_size, cmap=rock, vmin=0, vmax=1)
	axs[1].scatter(g_a, r_a, c=r_a, s=scatter_size, cmap=rock, vmin=0, vmax=1)
	axs[1].set(ylim=[0,1], ylabel = 'voltage transfer ratio')
	axs[2].scatter([-1*g for g in g_d], t_d, c=t_d, s=scatter_size, cmap=rock, vmin=t_range[0], vmax = t_range[1])
	axs[2].scatter(g_a, t_a, c=t_a, s=scatter_size, cmap=rock, vmin=t_range[0], vmax = t_range[1])
	axs[2].set(ylim=t_range, ylabel = 'transfer resistance (M\u03A9)', xlabel = 'distance from pSIZ (\u03BCm)')
	for i in range(3):
		axs[i].spines["top"].set_visible(False), axs[i].spines["right"].set_visible(False)
	# this font thing seems to do nothing
	for ax in [axs[0], axs[1], axs[2]]:
		for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
			item.set_fontsize(9)
		for item in (ax.get_xticklabels() + ax.get_yticklabels()):
			item.set_fontsize(8)
	plt.savefig(f'figs\\{target_name}_imp_props_vs_gd.svg', format = 'svg')


def fig_zi_vs_k():
	# plot z_i vs k for L1:
	fig, axs = plt.subplots(1,1, figsize=(2, 2))
	# then can plot any variables against each other, i.e.:
	axs.scatter(ratios, inps, c=transf, cmap=rock, s=1, vmin=0, vmax=1700)  # color is voltage transfer resistance
	axs.set(xlabel='voltage transfer ratio', ylabel='input resistance (MOhm)',
			xlim=[0,1], ylim=[0,7500])
	# inverse fit:
	inv_fit = lambda x: 5585 * 0.2/x
	# calculate R^2
	r_ss= sum([(z_i - inv_fit(ratios[k_ind]))**2 for k_ind, z_i in enumerate(inps)]) # residual sum of squares
	mean_inps = mean(inps)
	t_ss = sum([(z_i - mean_inps)**2 for z_i in inps])
	Rsqu = 1 - r_ss/t_ss
	axs.plot(np.arange(0.15, 1.05, 0.05), [inv_fit(x) for x in np.arange(0.15, 1.05, 0.05)], color = 'orange', 
				ls = 'dashed', label = f'$R^2={{{round(Rsqu,3)}}}$')
	axs.legend(loc='upper right', frameon = False, prop={'size':7})
	axs.spines["top"].set_visible(False), axs.spines["right"].set_visible(False)
	plt.tight_layout()
	plt.savefig('figs\\L1_Zi_vs_k.eps', format='eps')

def plot_imp_on_dendrites(custom_var = 'transfer', var_min = 0, var_max = 1700):
	'''plot z_c/z_i/k onto the 2D morphology of a neuron
		NOTE: change colormap by altering nrn.def file in C:/nrn/lib/
		NOTE: the code is NOT executable as a function, copy paste into command line
	'''

	custom_var = 'transfer'
	var_min = 0
	var_max = 1700

	custom_var = 'ratio'
	var_min = 0
	var_max = 1

	custom_var = 'input'
	var_min = 0
	var_max = 4000

	visualize_inputs(target_name = 'L1', target_body_id = 483716037, input_name = None)
	i_d, r_d, t_d, g_d = visualize_custom_var(prox_sec = 205, custom_var = custom_var) # dendrite 
	i_a, r_a, t_a, g_a = visualize_custom_var(prox_sec = 11, custom_var = custom_var) # axons
	ps = h.PlotShape(True)
	# L1: to orient into hemibrain 'coordinates' (dorsal up, facing brain)
	# ps.rotate(0,0,0,275*(np.pi/180),-15*(np.pi/180),0)
	ps.scale(var_min, var_max)
	ps.variable('zin_var')

	ax = ps.plot(plt, cmap = cm.hsv) # try rainbow colormap
	plt.show()

	ps = h.PlotShape(True)
	ps.scale(var_min, var_max)
	ps.scale(var_min, var_max)

	''' code to print RGB values for magma colormap (close to seaborn's rocket), to plug
		into a shape.cm file for nrn.def to read
	rock = sns.color_palette('rocket_r', as_cmap=True)
	for i in np.arange(0, 1.01, 0.05):
		rgb = [str(round(val*256)) for val in rock(i)]
		print('\t'.join(rgb[:3]))

	To generate a vertical standalone colorbar:
	plt.figure(figsize=(0.1,1))
	plt.imshow([[0,1]], cmap=sns.color_palette('rocket_r', as_cmap=True))
	plt.gca().set_visible(False)
	cax = plt.axes([0.1,0.1,0.05,0.6])
	plt.colorbar(orientation='vertical', cax=cax)

	plt.figure(figsize=(1,0.1)) # horizontal colorbar
	plt.imshow([[0,1]], cmap=sns.color_palette('rocket_r', as_cmap=True))
	plt.gca().set_visible(False)
	cax = plt.axes([0.1,0.5,0.6,0.05])
	plt.colorbar(orientation='horizontal', cax=cax)
	'''
def plot_dendritic_attrs(target_neuron_file = 'LHN_list_siz_dendr_locs.csv', toPlot = 'zc_v_gd',
							downsample = False, downsample_by = 12, 
							align_branch_out = True, save_out = False, fit_line = True):
	'''
		across LHNs, plot input resistance vs transfer ratio for dendritic segments
		we only iterate over dendritic subtrees here, as specified by 'prox_dendr_sec' in the csv
		
		toPlot = 'zi_v_k' or 'zc_v_gd'
		target_neuron_file = 'LHN_list_siz_axon_locs.csv' doesn't include dendritic branch out pts
															and has fewer neurons (less clutter?)
		align_branch_out: aligns branch out point to x-axis (geodesic dist)
		fit_line: fit and plot a straight line to the arbor points
	'''
	nrns = pd.read_csv(target_neuron_file)

	iters = 0

	lhns_colormap = {}
	import matplotlib
	rainbow = matplotlib.cm.get_cmap('hsv')
	for i, lhn in enumerate(nrns.lhn.unique()):
		lhns_colormap[lhn] = rainbow(i * 1/len(nrns.lhn.unique()))

	lhns_plotted = []
	arbor_fits = []
	fig, ax = plt.subplots(1, 1, figsize=(2, 2))
	for i, row in nrns.iterrows():
		if pd.notna(row['prox_dendr_sec']):

			target_name, target_body_id = row['lhn'], row['lhn_id']

			# capture initial dendritic section, and branch out point of arbor
			prox_dendr_secs = [int(val) for val in row['prox_dendr_sec'].split(', ')]
			prox_dendr_segs = [float(val) for val in row['prox_dendr_seg'].split(', ')]
			dendr_sec, dendr_seg = prox_dendr_secs[0], prox_dendr_segs[0]
			branchout_sec, branchout_seg = None, None
			if pd.notna(row['dendr_branch_out_sec']):
				branchout_sec, branchout_seg = int(row['dendr_branch_out_sec']), row['dendr_branch_out_seg']

			# currently only look into one of the dendritic sections (if multiple arbors)
			gd, zi, zc, k, branchout_dist = subtree_imp_props(target_name, target_body_id, dendr_sec, dendr_seg,
												branchout_sec, branchout_seg)
			if not branchout_dist:
				branchout_dist = 0
			# add code to realign zc_v_gd plots to the dendritic branch out points
			if align_branch_out and toPlot=='zc_v_gd':
				gd = [val-branchout_dist for val in gd]

			# fit a straight line to the dendritic arbor for zc_v_gd
			if toPlot=='zc_v_gd' and fit_line:
				gd_arbor = [g for g in gd if g > 0]
				zc_arbor = [z for ind, z in enumerate(zc) if gd[ind] > 0]
				arbor_fit = stats.linregress(gd_arbor, zc_arbor)
				toAppend = {'lhn': target_name, 'lhn_id': target_body_id, 
								'fit_type': 'arbor',
								'slope': arbor_fit.slope, 'intercept': arbor_fit.intercept,
								'r^2': arbor_fit.rvalue**2}
				arbor_fits.append(toAppend)
				# dendritic initial section
				gd_dis = [g for g in gd if g < 0]
				zc_dis = [z for ind, z in enumerate(zc) if gd[ind] < 0]
				if len(gd_dis)>0:
					DIS_fit = stats.linregress(gd_dis, zc_dis)
					toAppend = {'lhn': target_name, 'lhn_id': target_body_id, 
									'fit_type': 'DIS',
									'slope': DIS_fit.slope, 'intercept': DIS_fit.intercept,
									'r^2': DIS_fit.rvalue**2}
					arbor_fits.append(toAppend)
				#coeffs = np.polyfit(gd_arbor, zc_arbor, 1)
				#arbor_fit = np.poly1d(coeffs)
			
			if downsample:
				ds_len = int(np.floor(len(gd)/downsample_by)) # length of downsampled lists
				ds_idxs = random.sample(range(len(gd)), ds_len)
				# TODO: create a loop for this:
				gd = [val for i, val in enumerate(gd) if i in ds_idxs]
				zi = [val for i, val in enumerate(zi) if i in ds_idxs]
				zc = [val for i, val in enumerate(zc) if i in ds_idxs]
				k = [val for i, val in enumerate(k) if i in ds_idxs]
			if target_name not in lhns_plotted or target_body_id == 483716037:
				if toPlot=='zi_v_k':
					pts = ax.scatter(k, zi, s=1, alpha= 0.2, color=lhns_colormap[target_name],
								 label = target_name)
				elif toPlot=='zc_v_gd':
					#pts = ax.scatter(gd, zc, s=1, alpha= 0.2, color=lhns_colormap[target_name],
				#				 label = target_name)
					plt.plot(range(180), arbor_fit.intercept + range(180)*arbor_fit.slope, 
									color=lhns_colormap[target_name], linestyle = 'dashed', alpha=0.5,
									lw = 1)
					if len(gd_dis)>0:
						plt.plot(range(-80,0), DIS_fit.intercept + range(-80,0)*DIS_fit.slope,
									color=lhns_colormap[target_name], linestyle = 'dashed', alpha=0.5,
									lw = 1)

				lhns_plotted.append(target_name)
			else:
				if toPlot=='zi_v_k':
					ax.scatter(k, zi, s=1, alpha = 0.2, color = lhns_colormap[target_name])
				elif toPlot=='zc_v_gd':
					#ax.scatter(gd, zc, s=1, alpha = 0.2, color = lhns_colormap[target_name])
					plt.plot(range(180), arbor_fit.intercept + range(180)*arbor_fit.slope, 
								color=lhns_colormap[target_name], linestyle = 'dashed', alpha=0.5,
								lw = 1)
					if len(gd_dis)>0:
						plt.plot(range(-80,0), DIS_fit.intercept + range(-80,0)*DIS_fit.slope,
									color=lhns_colormap[target_name], linestyle = 'dashed', alpha=0.5,
									lw = 1)

			iters += 1
			#if iters > 4: break
	arbor_fits = pd.DataFrame(arbor_fits)

	if toPlot=='zi_v_k':
		ax.set(xlabel='voltage transfer ratio', ylabel='input resistance (MOhm)', xlim=[0,1], ylim=[0,7500])
	elif toPlot=='zc_v_gd':
		if align_branch_out: xlab = 'distance from branch out point (\u03BCm)'
		else: xlab = 'distance from pSIZ (\u03BCm)'
		ax.set(xlabel=xlab, ylabel='transfer resistance (MOhm)', ylim=[0,1800])
		if align_branch_out: ax.axvline(0, color = 'grey', ls = 'dashed', alpha = 0.2)
	lgnd = ax.legend(loc = 'center left', prop={'size':7}, ncol = 1, frameon=False, bbox_to_anchor = (1.01, 0.5),
						borderaxespad = 0)
	for handle in lgnd.legendHandles:
		handle.set_sizes([6.0]) # increase size of points in legend
	ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False)
	plt.subplots_adjust(right=0.9) # give room to legend on right
	
	if save_out:
		if toPlot=='zi_v_k':
			#plt.savefig('figs\\allLHNs_Zi_vs_k.jpg', format = 'jpg', bbox_inches='tight', dpi = 500)
			plt.savefig('figs\\allLHNs_Zi_vs_k.svg', format='svg')
		elif toPlot=='zc_v_gd':
			plt.savefig('figs\\allLHNs_Zc_vs_gd.svg', format='svg')
			#plt.savefig('figs\\allLHNs_Zc_vs_gd.jpg', format = 'jpg', bbox_inches='tight', dpi = 500)

	return arbor_fits

def compare_dendr_axon_linearity(target_neuron_file = 'LHN_list_siz_dendr_locs.csv'):
	'''fit lines to the axonal and dendritic arbors, compare slopes of 
		the two sets of best fit lines'''

	nrns = pd.read_csv(target_neuron_file)

	arbor_fits = []
	for i, row in nrns.iterrows():
		if pd.notna(row['prox_dendr_sec']):

			target_name, target_body_id = row['lhn'], row['lhn_id']
			isLocal = False
			if 'local' in target_name:
				isLocal = True

			d_branch_sec, d_branch_seg = int(row['dendr_branch_out_sec']), float(row['dendr_branch_out_seg'])
			a_branch_sec, a_branch_seg = int(row['ax_branch_out_sec']), float(row['ax_branch_out_seg'])

			# evaluate dendritic arbor (after first branch point)
			gd, zi, zc, k, branchout_dist = subtree_imp_props(target_name, target_body_id, 
												d_branch_sec, d_branch_seg)
			arbor_fit = stats.linregress(gd, zc)
			toAppend = {'lhn': target_name, 'lhn_id': target_body_id, 'is_local': isLocal, 
							'slope_dendr': arbor_fit.slope, 'intercept_dendr': arbor_fit.intercept,
							'r2_dendr': arbor_fit.rvalue**2}

			# evaluate axonal arbor (after first branch point)
			gd, zi, zc, k, branchout_dist = subtree_imp_props(target_name, target_body_id, 
												a_branch_sec, a_branch_seg)
			arbor_fit = stats.linregress(gd, zc)

			toAppend.update(slope_ax=arbor_fit.slope, intercept_ax = arbor_fit.intercept,
							r2_ax=arbor_fit.rvalue**2)

			arbor_fits.append(toAppend)
	arbor_fits = pd.DataFrame(arbor_fits)
	# quick and dirty plot comparing slopes of dendrite and axon
	# 21-01-19_dendr_vs_ax_linearity.csv
	a = arbor_fits
	plt.scatter(a.loc[~a.is_local]['slope_dendr'], a.loc[~a.is_local]['slope_ax'], label = 'output neurons')
	plt.scatter(a.loc[a.is_local]['slope_dendr'], a.loc[a.is_local]['slope_ax'], label = 'local neurons')
	plt.legend()
	plt.plot([0,-10], [0,-10], ls = 'dashed')
	plt.show()
	return arbor_fits

def analyze_arbor_fits(arbor_fits):

	lhns_colormap = {}
	import matplotlib
	rainbow = matplotlib.cm.get_cmap('hsv')
	for i, lhn in enumerate(arbor_fits.lhn.unique()):
		lhns_colormap[lhn] = rainbow(i * 1/len(arbor_fits.lhn.unique()))

	fig, ax = plt.subplots(1,1)
	for name, sub in arbor_fits.groupby('lhn'):
		ax.scatter(sub['slope'], sub['r^2'], label = name, color = lhns_colormap[name])
	ax.legend()

	#boxplot
	sns.boxplot(x='lhn', y='slope', data=arbor_fits)
	#compare DIS vs arbor fits
	plt.hist(arbor_fits.loc[arbor_fits['fit_type']=='arbor']['slope'], label = 'arbor fit', alpha = 0.3, bins = 10)
	plt.hist(arbor_fits.loc[arbor_fits['fit_type']=='DIS']['slope'], label = 'dendritic initial section fit', alpha=0.3, bins=10)
	plt.legend()
	plt.xlabel('slope')
	plt.hist(arbor_fits.loc[arbor_fits['fit_type']=='arbor']['r^2'], label = 'arbor fit', alpha = 0.3, bins = 10)
	plt.hist(arbor_fits.loc[arbor_fits['fit_type']=='DIS']['r^2'], label = 'dendritic initial section fit', alpha=0.3, bins=10)
	plt.legend()
	plt.xlabel('r^2')

def subtree_imp_props(target_name, target_body_id, dendr_sec, dendr_seg,
						branchout_sec = None, branchout_seg = None):
	'''given an LHN and a proximal dendritic section and segment, 
		measure the impedance properties of that section's subtree, 
		referenced to that section'''

	# biophysical parameters from our fits (NEEDS UPDATING)
	R_a = 350
	c_m = 0.6
	g_pas = 5.8e-5
	e_pas = -55 # one parameter left same in both, we used -55 in our fits
	syn_strength = 5.5e-5 # uS
	# instantiate target (post-synaptic) cell
	try:
		swc_path = "swc\\{}-{}.swc".format(target_name, str(target_body_id))
	except:
		print('no SWC found')
	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	print(target_name)	
	gd, zi, zc, k = [], [], [], [] # collect impedance measures for all dendritic segments
	branchout_dist = None
	imp = h.Impedance()
	imp.loc(dendr_seg, sec= cell1.axon[dendr_sec])
	imp.compute(0)
	for sec in cell1.axon[dendr_sec].subtree():
		for seg in sec:
			#try:
			gd.append(h.distance(cell1.axon[dendr_sec](dendr_seg), seg))
			zi.append(imp.input(seg))
			zc.append(imp.transfer(seg))
			k.append(imp.ratio(seg))
			# identify branch_out point (seg w/in 3 um of branchout):
			if branchout_sec and branchout_seg: # check if values are in spreadsheet
				if h.distance(cell1.axon[branchout_sec](branchout_seg), seg) < 3 and not branchout_dist:
					branchout_dist = gd[len(gd)-1]
			#except:
			#	print('error in evaluating impedance properties')

	return gd, zi, zc, k, branchout_dist

def analyze_passive_norm():
	'''
		code for brainstorming how to visually show passive normalization/
		the dendritic linearity part of the paper
	'''

	dendritic_sec = 3 # for ML9 542634516

	### after initializing ML9 using `visualize` code above!
	### one of the first dendritic sections distal to the SIZ is axon[3]
	gd, zi, zc, k = [], [], [], []
	imp = h.Impedance()
	imp.loc(0.5, sec= cell1.axon[dendritic_sec])
	imp.compute(0)
	for sec in cell1.axon[dendritic_sec].subtree():
		for seg in sec:
			gd.append(h.distance(cell1.axon[dendritic_sec](0.5), seg))
			zi.append(imp.input(seg))
			zc.append(imp.transfer(seg))
			k.append(imp.ratio(seg))
	# then can plot any variables against each other, i.e.:
	plt.scatter(k, [z/max(zi) for z in zi], s = 4, label = 'dendritic segments') # cool inverse relationship
	plt.plot(np.arange(0.2, 1.05, 0.05), [0.2/x for x in np.arange(0.2, 1.05, 0.05)], color = 'red', ls = 'dashed', label = 'y=0.2/x')
	plt.legend(loc='upper right')
	# calculate R^2
	norm_zi = [z/max(zi) for z in zi]
	r_ss= sum([(z - 0.2/k[k_ind])**2 for k_ind, z in enumerate(norm_zi)]) # residual sum of squares
	t_ss = sum([(z - mean(norm_zi))**2 for z in norm_zi])
	Rsqu = 1 - r_ss/t_ss

def probe_len_thick_AIS(target_neuron= 'local5', target_body_id = 5813105722,
						dendr_input = 'VA6_adPN', axon_input = 'VL2a_adPN',
						ais = 1002, uEPSP_measure_sec = 996, uEPSP_measure_seg = 0.5,
						toPrune = True):
	'''
		given a neuron and its axon initial section (local5 or local6) and one input
		targeting the axon and one targeting the dendrite, 
		produce a heatmap of the effect on the difference between axon and dendrite targeting
		uEPSPs, under the influence of thickening and shortening 

		for each combination of thicken and shorten, run a sub-method which produces the 
			EPSP sizes of VL2a and VA6

		target_neuron= 'local5', target_body_id = 5813105722,
						dendr_input = 'VA6_adPN', axon_input = 'VL2a_adPN',
						ais = 1002, uEPSP_measure_sec = 996, uEPSP_measure_seg = 0.5,
						toPrune = True
		target_neuron= 'local6', target_body_id = 386825553,
						dendr_input = 'DA4l_adPN', axon_input = 'VL2a_adPN',
						ais = 1002, uEPSP_measure_sec = 996, uEPSP_measure_seg = 0.5,
						toPrune = True
	'''

	thickness_range = np.arange(-0.3, 1.4, 0.1)
	length_range = np.arange(-60, 31, 5)
	#thickness_range = np.arange(-0.3, 1.0, 0.5)
	#length_range = np.arange(-60, 31, 40)

	conns = fetch_simple_connections(upstream_criteria = dendr_input, downstream_criteria = target_body_id)
	pre_bodyIds = conns.bodyId_pre
	dendr_syn_locs = fetch_synapse_connections(source_criteria = pre_bodyIds, target_criteria = target_body_id)
	conns = fetch_simple_connections(upstream_criteria = axon_input, downstream_criteria = target_body_id)
	pre_bodyIds = conns.bodyId_pre
	axon_syn_locs = fetch_synapse_connections(source_criteria = pre_bodyIds, target_criteria = target_body_id)

	if toPrune:
		dendr_syn_locs = dendr_syn_locs.iloc[0:len(axon_syn_locs)]	

	AIS_change_siz = pd.DataFrame(columns = list(length_range))
	AIS_change_soma = pd.DataFrame(columns = list(length_range))
	AIS_change_soma_t = pd.DataFrame(columns = list(length_range))
	for t in thickness_range:
		EPSPs_per_thickness_siz = []
		EPSPs_per_thickness_soma = []
		t_per_thickness_soma = []
		for l in length_range:
			dendr_EPSP_siz, dendr_EPSP_soma, dendr_t10to90_siz = change_AIS_sim_EPSP(target_neuron, target_body_id, dendr_input,
												uEPSP_measure_sec, uEPSP_measure_seg, dendr_syn_locs, ais, t, l)
			axon_EPSP_siz, axon_EPSP_soma, axon_t10to90_siz = change_AIS_sim_EPSP(target_neuron, target_body_id, axon_input,
												uEPSP_measure_sec, uEPSP_measure_seg, axon_syn_locs, ais, t, l)
			EPSPs_per_thickness_siz.append(dendr_EPSP_siz - axon_EPSP_siz)
			EPSPs_per_thickness_soma.append(dendr_EPSP_soma - axon_EPSP_soma)
			t_per_thickness_soma.append(dendr_t10to90_siz - axon_t10to90_siz)

		AIS_change_siz.loc[t] = EPSPs_per_thickness_siz
		AIS_change_soma.loc[t] = EPSPs_per_thickness_soma
		AIS_change_soma_t.loc[t] = t_per_thickness_soma

	return AIS_change_siz, AIS_change_soma, AIS_change_soma_t
	#ax = sns.heatmap(AIS_change_matrix)


def change_AIS_sim_EPSP(target_neuron, target_body_id, input_neuron, 
							uEPSP_measure_sec, uEPSP_measure_seg, syn_locs, ais, t, l):
	''' t: amount to thicken/thin; l: amount to lengthen/shorten, in micrometers
	'''

	# instantiate neuron and synapses
	cell1, curr_syns, netstim, netcons, num = visualize_inputs(target_name = target_neuron, 
													target_body_id = target_body_id, 
													input_name = input_neuron, 
													syn_locs = syn_locs)

	# alter AIS length and thickness
	cell1.axon[ais].L = cell1.axon[ais].L + l
	for seg in cell1.axon[ais]:
		seg.diam = seg.diam + t

	# simulate EPSP
	# measure uEPSP for connection at pSIZ and distal axon and soma
	netstim.number = 1 # activate stim
	h.load_file('stdrun.hoc')
	x = h.cvode.active(True)
	v_siz = h.Vector().record(cell1.axon[uEPSP_measure_sec](uEPSP_measure_seg)._ref_v)
	v_soma = h.Vector().record(cell1.axon[0](0.5)._ref_v)
	t = h.Vector().record(h._ref_t)                     				# Time stamp vector
	h.finitialize(-55 * mV)
	h.continuerun(40*ms)
	netstim.number = 0 # de-activate stim
	# measure rise time of EPSP at pSIZ
	t_10to90_siz = time_to_percent_peak(t, v_siz, 0.90) - time_to_percent_peak(t, v_siz, 0.10)

	# TODO: change output to dictionary {'siz': value, 'soma': value} to be more scalable
	return max(v_siz), max(v_soma), t_10to90_siz

def testTimingDiffs(target_name = 'local5', target_body_id = 5813105722,
					input1 = 'VA6_adPN', input2 = 'VL2a_adPN', 
					seed_set = 420,
					plot_loc = 'siz'):
	'''
		instantiate a neuron (local5), add 2 inputs (i.e. VA6, VL2a), stimulate these
			inputs at various time lags and measure relative joint EPSP size at SIZ vs 
			temporal offset;
			also time to 80% of max EPSP size vs temporal offset 

		params:
			input1: 'VA6_adPN'
			input2: 'VL2a_adPN' or 'simulated_dendritic_input'
			seed_set: change each time to get different random synapse placements 
						(I think the random seed carries into method calls)
			plot_loc: 'soma', 'prox_axon', 'dist_axon', 'prox_dendr', 'siz'
	'''
	global cell1, curr_syns1, curr_syns2
	
	random.seed(seed_set)
	if target_name == 'local5' and target_body_id == 5813105722:
		dendr_init_sec = 195

	# instantiate target (post-synaptic) cell
	try:
		swc_path = "swc\\{}-{}.swc".format(target_name, str(target_body_id))
	except:
		print('no SWC found')
	# biophysical parameters from our fits
	R_a = 350
	c_m = 0.6
	g_pas = 5.8e-5
	e_pas = -55 # one parameter left same in both, we used -55 in our fits
	syn_strength = 5.5e-5 # uS

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	# get number of post-synapses on the target neuron
	target, r = fetch_neurons(target_body_id)
	target_syn_count = target.post[0]

	# add all input1 synaptic site locations
	conns1 = fetch_simple_connections(upstream_criteria = input1, downstream_criteria = target_body_id)
	pre_bodyIds1 = conns1.bodyId_pre
	syn_locs1 = fetch_synapse_connections(source_criteria = pre_bodyIds1, target_criteria = target_body_id)
	curr_syns1, netstim1, netcons1, num1 = cell1.add_synapses_xyz(xyz_locs = syn_locs1, syn_strength = syn_strength)
	print('adding {} synapses from {} to {}; budget = {}'.format(str(num1), input1, target_name, str(num1/target_syn_count)))
	if target_name == 'local5' and target_body_id == 5813105722:
		# dendrite initial section is axon[195] for un-retraced local5
		num_in_dendr = 0
		for syn in curr_syns1:
			if str(syn.get_segment()).partition('(')[0] in [str(val) for val in cell1.axon[dendr_init_sec].subtree()]:
				num_in_dendr += 1
		print('proportion synapses in dendrite: {}'.format(str(num_in_dendr/num1)))

	# add all input2 synaptic site locations
	if input2 == 'simulated_dendritic_input':
		print('creating a simulated dendritic connection')
		### CHANGE: we'll make 75 fake dendritic synapses to correspond to VL2a's synapse count
		curr_syns2, netstim2, netcons2, num2 = cell1.add_synapses_subtree(sec_for_subtree = dendr_init_sec, 
																			syn_count = 75,
																			syn_strength = syn_strength)
		print('adding {} synapses from {} to {}; budget = {}'.format(str(num2), input2, target_name, str(num2/target_syn_count)))
		if target_name == 'local5' and target_body_id == 5813105722:
			# dendrite initial section is axon[195] for un-retraced local5
			num_in_dendr = 0
			for syn in curr_syns2:
				if str(syn.get_segment()).partition('(')[0] in [str(val) for val in cell1.axon[dendr_init_sec].subtree()]:
					num_in_dendr += 1
			print('proportion synapses in dendrite: {}'.format(str(num_in_dendr/num2)))
	else:
		conns2 = fetch_simple_connections(upstream_criteria = input2, downstream_criteria = target_body_id)
		pre_bodyIds2 = conns2.bodyId_pre
		syn_locs2 = fetch_synapse_connections(source_criteria = pre_bodyIds2, target_criteria = target_body_id)
		curr_syns2, netstim2, netcons2, num2 = cell1.add_synapses_xyz(xyz_locs = syn_locs2, syn_strength = syn_strength)
		print('adding {} synapses from {} to {}; budget = {}'.format(str(num2), input2, target_name, str(num2/target_syn_count)))
		if target_name == 'local5' and target_body_id == 5813105722:
			# dendrite initial section is axon[195] for un-retraced local5
			num_in_dendr = 0
			for syn in curr_syns2:
				if str(syn.get_segment()).partition('(')[0] in [str(val) for val in cell1.axon[dendr_init_sec].subtree()]:
					num_in_dendr += 1
			print('proportion synapses in dendrite: {}'.format(str(num_in_dendr/num2)))

	# set recording structures + locations
	# v: keys are recording locations, values are lists of voltage traces, index is input1 timing
	# t: list of time traces, index is input1 timing (matches index for v)
	t = []
	v = {'soma': [], 'prox_axon': [], 'dist_axon': [], 'prox_dendr': [], 'siz': []}
	if target_name == 'local5' and target_body_id == 5813105722:
		measure_locs = [('soma', 93), ('prox_axon', 1003), ('dist_axon', 1429), ('prox_dendr', 197),
						('siz', 996)] # section corresponding to each 

	# measure individual maximum amplitudes at each measure_loc for linearity analysis:
	# max amplitude for input1:
	netstim1.number = 1
	netstim1.start = 0
	netstim2.number = 0
	netstim2.start = 0
	h.load_file('stdrun.hoc')
	x = h.cvode.active(True)
	v_temp = {}
	for loc, sec in measure_locs:
		v_temp[loc] = h.Vector().record(cell1.axon[sec](0.5)._ref_v) # voltage trace vectors
	t_temp = h.Vector().record(h._ref_t)                     # time stamp vector
	h.finitialize(-55 * mV)
	h.continuerun(130*ms)
	max_v, locs = [], []
	for loc,sec in measure_locs:
		max_v.append(max(list(v_temp[loc])))
		locs.append(loc)
	max_v_per_loc = pd.DataFrame(data = max_v, index = locs, columns = ['input1'])
	# max amplitude for input2:
	netstim1.number = 0
	netstim1.start = 0
	netstim2.number = 1
	netstim2.start = 0
	h.load_file('stdrun.hoc')
	x = h.cvode.active(True)
	v_temp = {}
	for loc, sec in measure_locs:
		v_temp[loc] = h.Vector().record(cell1.axon[sec](0.5)._ref_v) # voltage trace vectors
	t_temp = h.Vector().record(h._ref_t)                     # time stamp vector
	h.finitialize(-55 * mV)
	h.continuerun(130*ms)
	max_v, locs = [], []
	for loc,sec in measure_locs:
		max_v.append(max(list(v_temp[loc])))
		locs.append(loc)
	max_v_per_loc['input2'] = max_v

	# input2 timing is set: 
	netstim2.number = 1
	netstim2.start = 50

	# input1 timing varies:
	print('initiating staggered input activation')
	for i in np.arange(0,101):
		netstim1.number = 1
		netstim1.start = i
		
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)
		v_temp = {}
		for loc, sec in measure_locs:
			v_temp[loc] = h.Vector().record(cell1.axon[sec](0.5)._ref_v) # voltage trace vectors
		t_temp = h.Vector().record(h._ref_t)                     # time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(130*ms)
		t.append(list(t_temp))
		for loc,sec in measure_locs:
			v[loc].append(list(v_temp[loc]))

	# plot peak EPSP at plot_loc for different timing
	plot_peak_EPSP(plot_loc = plot_loc, v = v, t = t, input1 = input1, input2 = input2,
					max_v_per_loc = max_v_per_loc, netstim2start = netstim2.start)

	return t, v, max_v_per_loc

def plot_peak_EPSP(plot_loc, v, t, input1, input2, max_v_per_loc, netstim2start):
	fig, ax = plt.subplots(nrows = 1, ncols = 1)
	ax.plot(np.arange(0,101), [max([val-e_pas for val in trace]) for trace in v[plot_loc]], 
					label = 'compound EPSP max @ {}'.format(plot_loc))
	ax.axvline(netstim2start, label = '{} activation time'.format(input2), color = 'red', ls = 'dashed')
	ax.axhline((max_v_per_loc['input1'][plot_loc]+55) + (max_v_per_loc['input2'][plot_loc]+55),
					label = 'linear sum of individual EPSPs', color = 'orange', ls = 'dashed')

	# plot example traces
	ax.plot(t[20], np.array(v[plot_loc][20])-e_pas, label = '{} precedes {} by 30 ms'.format(input1, input2))
	ax.plot(t[int(netstim2start)], np.array(v[plot_loc][int(netstim2start)])-e_pas, 
				label = 'simultaneous activation')
	ax.plot(t[80], np.array(v[plot_loc][80])-e_pas, label = '{} lags {} by 30 ms'.format(input1, input2))
	ax.legend(loc = 'upper right')
	ax.set_ylabel("depolarization (mV)")
	ax.set_xlabel("{} activation time".format(input1, input2))
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	plt.show()

def probe_mEPSPs(target_name = 'ML9', target_body_id = 542634516, input_name = 'DP1m_adPN',
					siz_sec = 569, siz_seg = 0.01,
					toPlot = True):
	'''
		given a downstream (target_name) neuron + ID, an upstream neuron, instantiate synapses
		(potentially pulling from neuprint) and simulate each mEPSP individually, allowing for
		recording of mEPSP amplitude, kinetics, etc. 			
	'''

	# Change syn_strength to new fit values!
	syn_strength = 5.5e-5 # uS

	# instantiate the cell and input synapses of interest
	cell1, curr_syns, netstim, netcons, num = visualize_inputs(target_name = target_name, 
												target_body_id = target_body_id, input_name = input_name)

	# first set all synapses to weight 0
	for i in range(len(list(curr_syns))):
		netcons.object(i).weight[0] = 0 # uS, peak conductance change

	# sequentially activate all input synapses
	per_synapse_data = []
	# re-run mEPSP simulation for each synapse
	for i in (range(len(list(curr_syns)))):

		# activate a single synapse
		netcons.object(i).weight[0] = syn_strength
		if i % 20 == 1:
			print("activating synapse " + str(i))

		# measure mEPSP for connection at pSIZ 
		# activate the stim
		netstim.number = 1
		h.load_file('stdrun.hoc')
		x = h.cvode.active(True)
		v_siz = h.Vector().record(cell1.axon[siz_sec](siz_seg)._ref_v)
		#v_axon = h.Vector().record(cell1.axon[axon_sec](axon_seg)._ref_v)
		#v_soma = h.Vector().record(cell1.axon[0](0.5)._ref_v)
		t = h.Vector().record(h._ref_t)                     				# Time stamp vector
		h.finitialize(-55 * mV)
		h.continuerun(40*ms)	# initiate run
		netstim.number = 0
		# measure rise time of EPSP at pSIZ
		t_10to90_siz = time_to_percent_peak(t, v_siz, 0.90) - time_to_percent_peak(t, v_siz, 0.10)

		toAppend = {}
		toAppend.update(synapse_number = i, mEPSP_siz = max(list(v_siz))+55,
							mEPSP_t10to90_siz = t_10to90_siz,
							syn_count = len(curr_syns),
							local_diam = curr_syns.object(i).get_segment().diam,
							dist_to_siz = h.distance(cell1.axon[siz_sec](siz_seg), curr_syns.object(i).get_segment()))
		per_synapse_data.append(toAppend)

		# de-activate the synapse
		netcons.object(i).weight[0] = 0
	per_synapse_data = pd.DataFrame(per_synapse_data)

	# reset all synapse strengths before other tests:
	for i in range(len(list(curr_syns))):
		netcons.object(i).weight[0] = syn_strength # uS, peak conductance change

	if toPlot:
		fig, axs = plt.subplots(nrows = 2, ncols = 2)

		axs[0,0].scatter(per_synapse_data.dist_to_siz, per_synapse_data.mEPSP_siz)
		axs[0,0].set_xlabel('distance to SIZ (um)'), axs[0,0].set_ylabel('mEPSP @ SIZ (mV)')

		axs[0,1].scatter(per_synapse_data.local_diam, per_synapse_data.mEPSP_siz)
		axs[0,1].set_xlabel('local diameter (um)'), axs[0,1].set_ylabel('mEPSP @ SIZ (mV)')

		axs[1,0].scatter(per_synapse_data.dist_to_siz, per_synapse_data.mEPSP_t10to90_siz)
		axs[1,0].set_xlabel('distance to SIZ (um)'), axs[1,0].set_ylabel('t 10 to 90% peak @ SIZ (ms)')

		axs[1,1].scatter(per_synapse_data.local_diam, per_synapse_data.mEPSP_t10to90_siz)
		axs[1,1].set_xlabel('local diameter (um)'), axs[1,1].set_ylabel('t 10 to 90% peak @ SIZ (ms)')

		fig.suptitle(f"{input_name} inputs onto {target_name} {target_body_id}")

	return per_synapse_data

def sim_DM1(params = 'Gouwens'):
	'''
		Simulate a uEPSP to the DM1 neuron from the hemibrain, to compare 
		attenuation between the hemibrain version and the Gouwens version
		Goal: to understand how true is Gouwens' claim that:
			"EPSPs still did not cause substantial voltage changes at the axon terminals"
		Our results suggest that fly neurons, including PNs, are electrotonically compact
		enough that EPSPs would cause substantial voltage changes

		params: 'Gouwens' for Gouwens 2009 parameters
				'our fit' for our population-level MC model fit parameters
	'''

	# change to path for hemibrain DM1 
	swc_path = "swc\\DM1-542634818_hemibrain.swc"

	# biophysical parameters from Gouwens or from our fits
	if params == 'Gouwens':
		R_a = 266.1
		c_m = 0.79
		g_pas = 4.8e-5
		e_pas = -60
		syn_strength = 0.00027 # uS, peak synaptic conductance, g_syn
	elif params == 'our fit':
		R_a = 350
		c_m = 0.6
		g_pas = 5.8e-5
		e_pas = -60 # one parameter left same in both, we used -55 in our fits
		syn_strength = 5.5e-5 # uS

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	# from using Point Group GUI to find sections in dendrite, randomly chosen to mimic fig 6D
	syn_secs = [2894, 2479, 6150, 2716, 6037, 5259, 3611, 5178, 5100, 5036, 4947, 4436, 2637,
				2447, 3838, 2873, 4780, 6468, 3297, 2435, 4073, 2438, 6119, 4476, 2768]

	syns = h.List()
	for i in range(len(syn_secs)):
		syns.append(h.Exp2Syn(cell1.axon[syn_secs[i]](0.5)))
		### synapse parameters from Tobin et al paper: 
		syns.object(i).tau1 = 0.2 #ms
		syns.object(i).tau2 = 1.1 #ms
		syns.object(i).e = -10 #mV, synaptic reversal potential = -10 mV for acetylcholine? 

	### use NetStim to activate NetCon
	nc = h.NetStim()
	nc.number = 1
	nc.start = 0
	nc.noise = 0

	ncs = h.List()
	for i in range(len(list(syns))):
		ncs.append(h.NetCon(nc, syns.object(i)))
		ncs.object(i).weight[0] = syn_strength # uS, peak conductance change

	### measure the depolarization at the synapse locations, and at a few points in the distal axon
	### compare the attenuation for both Gouwens and the hemibrain DM1

	h.load_file('stdrun.hoc')
	x = h.cvode.active(True)

	# establish recordings at all input synapse locations
	v_input = []
	for i in range(len(list(syns))):
		record_loc = syns.object(i).get_segment()
		v_input.append(h.Vector().record(record_loc._ref_v))

	# establish recordings at a few axon locations, in both MB and LH
	v_record = []
	axon_secs = [2094, 1925, 1878, 1740, 647, 896, 1339, 1217]
	for i in range(len(axon_secs)):
		v_record.append(h.Vector().record(cell1.axon[axon_secs[i]](0.5)._ref_v))

	v_soma = h.Vector().record(cell1.soma[0](0.75)._ref_v) 		# soma membrane potential
	t = h.Vector().record(h._ref_t)                     # Time stamp vector

	h.finitialize(-60 * mV)
	h.continuerun(60*ms)

	# plot

	plt.rcParams["figure.figsize"] = (15,15)
	plt.plot(list(t), list(v_soma), color = 'red', label = 'soma recording')
	for trace in v_input:
		if v_input.index(trace) == 0:
			plt.plot(list(t), list(trace), color = 'blue', label = 'synapse site recording')
		else:
			plt.plot(list(t), list(trace), color = 'blue')
	for trace in v_record:
		if v_record.index(trace) == 0:
			plt.plot(list(t), list(trace), color = 'green', label = 'axon site recording')
		else:
			plt.plot(list(t), list(trace), color = 'green')
	plt.xlabel('time (ms)')
	plt.ylim([-60, -40])
	plt.ylabel('membrane potential (mV)')
	plt.legend(loc = 'upper right')
	plt.title('Hemibrain DM1 uEPSP, parameters from {}'.format(params))
	plt.show()

	# print area
	print("DM1 area: {} um^2".format(str(cell1.surf_area())))

def find_KC_classes():
	'''
		cycle through folder of KC SWCs, pull out body IDs and search for their subclasses
		output csv of body ID and KC 
	'''
	print('todo')

'''
Past parameter fits: 

# 20-10-05 tile other portion of g_pas w/ finer granularity of R_a
# 8 * 4 * 12 * 13 = 4992
#syn_strength_s = [2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5, 4.5e-5, 5.0e-5, 5.5e-5, 6.0e-5]
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.2e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 25) # ohm-cm 
# ended 20-10-07

# 20-10-01 tile other portion of R_a
# 8 * 4 * 13 * 6 = 2496
#syn_strength_s = [2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5, 4.5e-5, 5.0e-5, 5.5e-5, 6.0e-5]
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(75, 351, 50) # ohm-cm 
# start time: 20-10-01-01:59:30, end time: 20-10-02-12:14:34

# 20-09-27 after refreshing body IDs after Pasha's final revisions, saving all_resids
# 8 * 4 * 13 * 7 = 2912
#syn_strength_s = [2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5, 4.5e-5, 5.0e-5, 5.5e-5, 6.0e-5]
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 50) # ohm-cm 

# 20-09-27 after refreshing body IDs after Pasha's final revisions, saving all_resids
# 8 * 4 * 13 * 7 = 2912
#syn_strength_s = [2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5, 4.5e-5, 5.0e-5, 5.5e-5, 6.0e-5]
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 50) # ohm-cm 
# ended 20-09-29

# 20-09-13_broader search range: 
# 4 x 4 x 12 x 7 = 1344 + 144 from previous search!
#syn_strength_s = [3.0e-5, 4.0e-5, 5.0e-5, 5.5e-5]
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.8e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 50) # ohm-cm

# 20-09-12_after changing error term to peak residual sum of squares PER connection: 
# 2 x 3 x 6 x 4 = 144
#syn_strength_s = np.arange(0.000035, 0.000046, 0.00001) # uS
#c_m_s = np.arange(0.6, 1.21, 0.3) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.8e-5, 0.8e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 100) # ohm-cm

# 20-09-10_refit after local5 reverted to v1.1
# 1440
#syn_strength_s = np.arange(0.000030, 0.000056, 0.000005) # uS
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.8e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 75) # ohm-cm

# 20-09-04_change syn conductance: 1:28am to 
#syn_strength_s = np.arange(0.0000325, 0.00005, 0.000005) # uS
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.8e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 75) # ohm-cm
# define g_pas and R_a

# 20-08-31_larger_param_search
#c_m_s = np.arange(1.0, 1.21, 0.1) # uF/cm^2
#g_pas_s = np.arange(1.0e-5, 5.4e-5, 0.2e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 351, 25) # ohm-cm

# 20-08-31_small_param_search: 11*6*105 = 6930 possibilities
#g_pas_s = np.arange(1.2e-5, 5.4e-5, 0.4e-5) # S/cm^2, round to 6 places
#R_a_s = np.arange(50, 350, 50) # ohm-cm

'''

'''
filename = "swc\\local6-356467849.swc"
#def instantiate_swc(filename):
"""load an swc file and instantiate it"""

# a helper library, included with NEURON
h.load_file('import3d.hoc')

# load the data. Use Import3d_SWC_read for swc, Import3d_Neurolucida3 for
# Neurolucida V3, Import3d_MorphML for MorphML (level 1 of NeuroML), or
# Import3d_Eutectic_read for Eutectic. (There is also an 
# Import3d_Neurolucida_read for old Neurolucida files, but I've never seen one
# in practice; try Import3d_Neurolucida3 first.)
cell = h.Import3d_SWC_read()
cell.input(filename)

# easiest to instantiate by passing the loaded morphology to the Import3d_GUI
# tool; with a second argument of 0, it won't display the GUI, but it will allow
# use of the GUI's features
i3d = h.Import3d_GUI(cell, 0)
i3d.instantiate(None)
#i3d.cbexport()
'''

# If you have a CellBuilder cb, the management tab options are in cb.manage, 
# and the main function for exporting as a HOC template is cb.manage.save_class.
# cb = h.CellBuild(0)

### LEGACY Code:
'''
### select LHN
	if which_LHN == "local6":
		if which_LHN_vers == 1:
			LHN_ID = 'local6-356467849'
		elif which_LHN_vers == 2:
			LHN_ID = 'local6-417186656'
		elif which_LHN_vers == 3:
			LHN_ID = 'local6-479917037'
		elif which_LHN_vers == 4:
			LHN_ID = 'local6-418865948'
	if which_LHN == "local5":
		if which_LHN_vers == 1:
			LHN_ID = 'local5-5813105722'
		elif which_LHN_vers == 2:
			LHN_ID = 'local5-696126258'
	if which_LHN == "local2":
		if which_LHN_vers == 1:
			LHN_ID = 'local2-5813055963'
		elif which_LHN_vers == 2:
			LHN_ID = 'local2-666450841'
	if which_LHN == "L1":
		if which_LHN_vers == 1:
			LHN_ID = 'L1-483716037'
	if which_LHN == "L12":
		if which_LHN_vers == 1:
			LHN_ID = 'L12-5813077898'
	if which_LHN == "L13":
		if which_LHN_vers == 1:
			LHN_ID = 'L13-793702856'
	if which_LHN == "ML3":
		if which_LHN_vers == 1:
			LHN_ID = 'ML3-574040939'
	if which_LHN == "ML8":
		if which_LHN_vers == 1:
			LHN_ID = 'ML8-509928512'
	if which_LHN == "ML9":
		if which_LHN_vers == 1:
			LHN_ID = 'ML9-640963556'
		if which_LHN_vers == 2:
			LHN_ID = 'ML9-573337611'
	if which_LHN == "V2":
		if which_LHN_vers == 1:
			LHN_ID = 'V2-1037510115'
	if which_LHN == "V3":
		if which_LHN_vers == 1:
			LHN_ID = 'V3-915724147'

	### select PN
	if which_PN == "DA4l":
		PN_ID = 'DA4l-544021095'
	if which_PN == "VA6":
		PN_ID = "VA6-1881751117"
	if which_PN == "VL2a":
		PN_ID = "VL2a-5813069089"
	if which_PN == "VL2p":
		PN_ID = "VL2p-1944507292"
	if which_PN == "DL5":
		PN_ID = "DL5-693483018"
	if which_PN == "DM1":
		PN_ID = "DM1-542634818"
	if which_PN == "DM4":
		PN_ID = "DM4-573333835"
	if which_PN == "DP1m":
		PN_ID = "DP1m-635062078"
	if which_PN == "VA2":
		PN_ID = "VA2-1977579449"
	if which_PN == "VC1":
		PN_ID = "VC1-606090268"
'''