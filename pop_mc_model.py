
### the file containing list of LHN and PN body IDs and EPSP sizes
### was 20-08-27_all_conns.csv prior to final body ID refresh on 20-09-26
conn_file = "20-09-26_all_conns.csv"

### NOTE: file should be run from directory:
### C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model\\population_model\\

import sys
sys.path.append("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model")
from run_local5 import *
from datetime import date

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
	# interpolate t value where v reaches inputted percent of peak amplitude
	time_of_perc = np.interp(perc_of_peak, np.array(v)[0:peak_loc], np.array(t)[0:peak_loc])
	# subsets portion of trace up to the peak
	#ind_of_perc = np.abs(np.array(v)[0:peak_loc] - perc_of_peak).argmin() 
	#time_of_perc = np.array(t)[ind_of_perc]
	return time_of_perc

# previously was local5 params: epas = -55, cm = 1.2, synstrength = 3.5e-5, ra = 125, gpas = 4.4e-5
# now error peak no skip params
def find_peak_vals(epas = -55, cm = 0.6, synstrength = 5.5e-5, ra = 350, gpas = 5.8e-5, params = 'abs_norm_noskip',
					save_excel = False, show_plot = False, show_scatter = False, show_kinetics_scatter = False):
	'''
		Given biophysical parameter inputs, output the resulting simulated EPSPs across the 
		entire population. 
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

		fig, ax = plt.subplots(nrows = 1, ncols = 1)

		#plt.scatter(all_conns.loc[:, 'epsp_exp'], all_conns.loc[:, 'epsp_sim'], color = 'black')
		ax.scatter(sim_avgs.loc[:, 'epsp_exp'], sim_avgs.loc[:, 'epsp_sim'], 
					s = 2, color = 'black')
		ax.set_xlabel("experimental EPSP peak (mV)")
		ax.set_ylabel("simulated EPSP peak (mV)")
		ax.plot([0, 7], [0, 7], color = 'grey', ls = '--')
		props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(synstrength*1000, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2, R_a = " + str(ra) + 
			" Ohm-cm")
		plt.suptitle(props + " [current params]", 
				 fontsize = 24, y = 0.96)

		draw()

		fig.savefig('{}_mcsim_params{}_scatter.svg'.format(date.today().strftime("%y-%m-%d"), str(params)))

		plt.show()

	### fig_mcmodel_kinetics_predictions
	### scatter of rise time predictions vs actual
	if show_kinetics_scatter:
		plt.rcParams['figure.figsize'] = (2, 2)

		fig, ax = plt.subplots(nrows=1, ncols=1)

		ax.scatter(sim_avgs["t_exp_10to90"], sim_avgs["t_sim_10to90"], color = 'blue')

		ax.plot([0, 11], [0, 11], color = 'grey', alpha = 0.3) # unity line

		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.set_xlabel('experimental rise time (ms)', fontsize = 9)
		ax.set_ylabel('simulated rise time (ms)', fontsize = 9)

		draw()
		fig.savefig("fig_mcmodel_kinetics_predictions.svg")
		plt.show()

	return sim_traces, sim_avgs, sum_peak_err

def plot_traces(sim_traces, cm, synstrength, ra, gpas):

	plt.rcParams["figure.figsize"] = (8,7)

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

	# rewrite to display simulation averages (not individual traces), then save as svg with param class
	for lhn in lhn_list:
		for pn in pn_list:
			trace_locs = [i for i in range(len(sim_traces)) if sim_traces[i]["lhn"]==lhn and sim_traces[i]["pn"]==pn]

			# average the traces at trace_locs


	for i in range(len(sim_traces)):

		### plot simulated and experimental traces
		row = lhn_list.index(sim_traces[i]["lhn"])
		col = pn_list.index(sim_traces[i]["pn"])
		# read & plot experimental trace
		trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(sim_traces[i]["lhn"], sim_traces[i]["pn"]), header = None, dtype = np.float64)
		t_exp = trace_exp[0]+1.25 # slightly adjust to align with rise time of EPSP
		v_exp = trace_exp[1]
		axs[row, col].plot(t_exp, v_exp, color = 'red')

		axs[row, col].plot(sim_traces[i]["t_sim"], [x+55 for x in sim_traces[i]["v_sim"]], color = 'grey', alpha = 0.2)

	props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(synstrength*1000, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2, R_a = " + str(ra) + 
			" Ohm-cm")
	plt.suptitle(props + " [current params]", 
				 fontsize = 24, y = 0.96)

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

def param_search():

	all_conns = pd.read_csv(conn_file)

	start_time = date.today().strftime('%y-%m-%d-%H:%M:%S')

	e_pas = -55 # mV

	# 20-09-27 after refreshing body IDs after Pasha's final revisions, saving all_resids
	# 8 * 4 * 13 * 7 = 2912
	syn_strength_s = [2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5, 4.5e-5, 5.0e-5, 5.5e-5, 6.0e-5]
	c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
	g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
	R_a_s = np.arange(50, 351, 50) # ohm-cm 

	sim_params = []
	all_resids = []

	# iterate through all biophysical parameter combinations
	for syn_strength_i in syn_strength_s:
		for c_m_i in c_m_s:
			for g_pas_i in g_pas_s:
				for R_a_i in R_a_s:

					### following errors skip local5/vl2a
					sum_peak_err = 0 # sum of absolute values of normalized residuals
					sum_peak_err_notnorm = 0 # sum of absolute values of non-normalized residuals
					sum_peak_err_rss = 0 # sum of squares of normalized residuals
				
					### doesn't skip local5/vl2a
					sum_peak_err_noskip = 0 # sum of absolute values of normalized residuals
					sum_peak_err_notnorm_noskip = 0 # sum of absolute values of non-normalized residuals
					
					sim_traces, sim_avgs, rss_err = find_peak_vals(cm = c_m_i, ra = R_a_i, synstrength = syn_strength_i,
														  gpas = g_pas_i, save_excel = False, show_scatter = False)

					for i in range(len(sim_avgs)):
						normalized_resid = (sim_avgs.loc[i, 'epsp_sim'] - sim_avgs.loc[i, 'epsp_exp']) / sim_avgs.loc[i, 'epsp_exp']
						if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
							sum_peak_err += np.abs(normalized_resid) # normalized residual

						sum_peak_err_noskip += np.abs(normalized_resid) # normalized residual
						sum_peak_err_notnorm_noskip += np.abs(normalized_resid * sim_avgs.loc[i, 'epsp_exp']) # non-normalized residual

						if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
							sum_peak_err_notnorm += np.abs(normalized_resid * sim_avgs.loc[i, 'epsp_exp']) # non-normalized residual
						if sim_avgs.loc[i, 'lhn'] != 'local5' or sim_avgs.loc[i, 'pn'] != 'VL2a':
							sum_peak_err_rss += normalized_resid**2 # squared normalized residual

					# save parameter values, (output trace indices), fit errors
					params_toAppend = {}
					params_toAppend.update(g_syn = syn_strength_i, g_pas = g_pas_i, R_a = R_a_i, 
									c_m = c_m_i,
									error_peak = sum_peak_err, error_peak_noskip = sum_peak_err_noskip,
									error_peak_notnorm = sum_peak_err_notnorm,
									error_peak_rss = sum_peak_err_rss, 
									error_peak_notnorm_noskip = sum_peak_err_notnorm_noskip,
									all_resids_index = len(all_resids))

					# save overall statistics AND residuals per connection for this parameter set
					sim_params.append(params_toAppend)
					all_resids.append(sim_avgs) 

				#print("finished running " + str(str(round(g_pas_i, 6))) + " S/cm^2")

	sim_params = pd.DataFrame(sim_params)

	sim_params.to_excel('{}_mcfit_{}.xlsx'.format(date.today().strftime("%y-%m-%d"), str(len(sim_params))))

	end_time = date.today().strftime('%y-%m-%d-%H:%M:%S')
	print("start time: {}, end time: {}".format(start_time, end_time))

	return sim_params, all_resids

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

def transf_imped_of_inputs(down, down_id, up, up_id, siz_sec, siz_seg, transf_freq = 20):
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

		syn_info.append([curr_distance, curr_transf_imp])


	# plot synapse to SIZ distance vs transfer impedance
	plt.scatter([syn[0] for syn in syn_info], [syn[1] for syn in syn_info])
	plt.show()

d, d_id, u, u_id, s_sec, s_seg = 'local5', '5813105722', 'VA6', '1881751117', 996, 0
transf_imped_of_inputs(down = d, down_id = d_id, up = u, up_id = u_id, 
						siz_sec = s_sec, siz_seg = s_seg, transf_freq = 20)

def instantiate_lhns():
	'''
		code I copy pasted into the command line to manually
		look at the transfer impedance / other electrotonic measures
		in the GUI for two example LHNs
	'''

	# change to path for hemibrain DM1 
	swc_path = "swc\\ML9-542634516.swc"
	# swc_path = "swc\\L12-391609333.swc"

	# biophysical parameters from our fits
	R_a = 350
	c_m = 0.6
	g_pas = 5.8e-5
	e_pas = -60 # one parameter left same in both, we used -55 in our fits
	syn_strength = 5.5e-5 # uS

	cell1 = Cell(swc_path, 0) # first argument is name of swc file, second is a gid'
	cell1.discretize_sections()
	cell1.add_biophysics(R_a, c_m, g_pas, e_pas) # ra, cm, gpas, epas
	cell1.tree = cell1.trace_tree()

	syn_path = "syn_locs\\ML9-542634516_DM1-542634818.csv"

	synapses, netstim, netcons, num = cell1.add_synapses(syn_path, syn_strength)

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

'''
Past parameter fits: 

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