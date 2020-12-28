
### the file containing list of LHN and PN body IDs and EPSP sizes
### was 20-08-27_all_conns.csv prior to final body ID refresh on 20-09-26
### EPSP value refresh on 20-12-08
conn_file = "20-12-08_all_conns.csv"

###
### for each unique PN to LHN connection, simulate a single compartment cell
###

import sys
sys.path.append("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model")
from run_local5 import *
#plt.rcParams.update({'font.size': 30})

### directly solving conductance-based differential equation
### c_m dV/dt + g_pas (V - V_rest) = I(t)
### dV/dt = (I(t) - g_pas (V - V_rev)) / c_m
### I(t) = - G_syn (V - V_synrev) = -1 * syn_strength * factor * (exp(-t/tau2) - exp(-t/tau1)) * (V - V_synrev)
### see factor term below
### 
### FIXED:
### tau1 = 0.2 ms
### tau2 = 1.1 ms
### V_synrev = -10 mV
### V_rest = -55 mV
### c_m = ? uF (/cm^2 * area cm^2)
### G_L = ? S (/cm^2 * area cm^2)
### syn_strength = ? S (3.5e-5 uS)

from scipy.integrate import odeint
from math import log, exp
from datetime import datetime

# parameters, default at local5 values (only used if run_sim is run directly)
num_syn_def = 2 # number of synapses
surf_area = 2230 # um^2, average of surf area of all LHNs in dataset, weighted by # of PN connections in dataset
tau1 = 0.2 # ms
tau2 = 1.1 # ms
g_pas_def = 4.4e-5 * (1/10000)**2 * surf_area * 1e9 # nS, note (1cm/10000um)
c_m_def = 1.2 * (1/10000)**2 * surf_area # uF
syn_strength_def = 0.035 # nS

# normalize peak of conductance waveform to 1: 
# normally 2 state kinetic scheme for a synapse has the following expression: 
# g(t) = tau1*tau2 / (tau2-tau1) (-1*exp(-t_p / tau1) + exp(-t_p/tau2))
# but to normalize the peak of the conductance to 1, we apply a factor
t_p = tau1*tau2/(tau2-tau1) * log(tau2/tau1) # peak t, from NEURON Github for Exp2Syn / Wolfram Alpha
peak_g = -1*exp(-t_p / tau1) + exp(-t_p/tau2)
factor = 1/peak_g

# single compartment neuron
def sc_neuron(v, t, g_pas, c_m, syn_strength, syn_count):
	v_rev = -10 # mV
	v_rest = -55 # mV
	# set units for dv/dt to mV/ms using 10^-6 factor
	dvdt = 1e-6 * 1/c_m * (-1*syn_count*syn_strength*factor*(exp(-t/tau2) - exp(-t/tau1))*(v-v_rev) - g_pas*(v-v_rest)) #mV/ms
	#print(dvdt)
	return dvdt

def run_sim(g_pas = g_pas_def, c_m = c_m_def, syn_strength = syn_strength_def, syn_count = num_syn_def):
	v0 = -55
	t = np.linspace(0, 20, 41)
	sol = odeint(sc_neuron, v0, t, args = (g_pas, c_m, syn_strength, syn_count))

	#plt.plot(t, sol[:,0])
	#plt.show()

	return sol[:,0]

###
### plot peak EPSP value vs number of input synapses, local5 params
###
def num_syn_scaling():

	peak_vals = []
	for i in range(200):
		peak_vals.append(max(run_sim(syn_count = i)))

	plt.plot(np.arange(200), peak_vals)
	plt.show()

# 20-09-29 best fits: 
# SCv1: 0.00006	0.6	0.0185
# SCv2: 0.000064 0.6 0.0225
def find_peak_vals_SC(version, gpas = 5e-5, cm = 0.8, gsyn = 0.0175, 
					show_plot = False, show_scatter = False, save_excel = False):

	all_conns = pd.read_csv(conn_file)

	g_pas_i = gpas # S/cm^2
	c_m_i = cm # uF/cm^2
	syn_strength_i = gsyn # nS, peak synaptic conductance

	sim_traces = []
	# iterate through all connections
	for i in range(len(all_conns)):
		curr_trace = {}
		curr_trace.update(lhn = all_conns.lhn[i], pn=all_conns.pn[i])

		num = all_conns.num_syn[i]
		if version==1:
			surf_area = 2367 # weighted average surf area of all LHN's
			# update 20-09-26 after refreshing body IDs
		elif version==2:
			surf_area = all_conns.lhn_SA[i] # unique SA for each LHN body ID

		if num > 0:

			# calculate overall (non-surf-area-normalized) parameters
			g_pas_o = g_pas_i * (1/10000)**2 * surf_area * 1e9 # nS, note (1cm/10000um)
			c_m_o = c_m_i * (1/10000)**2 * surf_area # uF

			t = np.linspace(0, 20, 41)
			v_s = run_sim(g_pas=g_pas_o, c_m=c_m_o, syn_strength=syn_strength_i, syn_count=num)

			all_conns.loc[i, 'epsp_sim'] = float(max(v_s)+55)

			curr_trace.update(t_sim = t, v_sim = v_s, v_sim_peak = max(v_s)+55, 
							  v_exp_peak = all_conns.epsp_exp[i])

		else:
			all_conns.loc[i, 'epsp_sim'] = 0 # EPSP size = 0 if no synapses

			curr_trace.update(t_sim = [0, 20], v_sim = [-55, -55], v_sim_peak = 0, 
							  v_exp_peak = all_conns.epsp_exp[i])

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
		toAppend.update(lhn = curr_lhn, pn = curr_pn, epsp_sim = np.mean(conn_peaks),
						epsp_exp = sim_traces[conn_indices[conn][0]]["v_exp_peak"])
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
		plot_traces_SC(sim_traces, cm, gsyn, gpas, version)
	if save_excel:
		all_conns.to_excel('{}_scsim_v{}_each_inst.xlsx'.format(datetime.today().strftime("%y-%m-%d"), str(version)))
		sim_avgs.to_excel('{}_scsim_v{}_avgs.xlsx'.format(datetime.today().strftime("%y-%m-%d"), str(version)))
	if show_scatter:
		plt.rcParams["figure.figsize"] = (15,15)

		fig, ax = plt.subplots(nrows = 1, ncols = 1)

		#plt.scatter(all_conns.loc[:, 'epsp_exp'], all_conns.loc[:, 'epsp_sim'], color = 'black')
		ax.scatter(sim_avgs.loc[:, 'epsp_exp'], sim_avgs.loc[:, 'epsp_sim'], color = 'black')
		ax.set_xlabel("experimental EPSP peak (mV)")
		ax.set_ylabel("simulated EPSP peak (mV)")
		ax.plot([0, 7], [0, 7], color = 'grey', ls = '--')
		props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(synstrength*1000, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2, R_a = " + str(ra) + 
			" Ohm-cm")
		plt.suptitle(props + " [current params]", 
				 fontsize = 15, y = 0.96)

		draw()

		fig.savefig('{}_scsim_v{}_scatter.svg'.format(date.today().strftime("%y-%m-%d"), str(version)))

		plt.show()

	return sim_traces, sim_avgs, sum_peak_err

def plot_traces_SC(sim_traces, cm, gsyn, gpas, version):

	plt.rcParams["figure.figsize"] = (9, 7)

	# 14 LHN by 17 PN plot
	fig, axs = plt.subplots(nrows = 14, ncols = 17, sharex = True, sharey = True)
	lhn_list = ['CML2', 'L1', 'L11', 'L12', 'L13', 'L15', 'ML3', 'ML8', 'ML9', 'V2', 'V3', 'local2', 'local5', 'local6']
	pn_list = ['DA4l', 'DC1', 'DL4', 'DL5', 'DM1', 'DM3', 'DM4', 'DP1m', 'VA1v', 'VA2', 'VA4', 'VA6', 'VA7l', 'VC1', 'VC2', 'VL2a', 'VL2p']
	[ax.set_xlim(0,21) for subrow in axs for ax in subrow]
	[ax.set_ylim(0,7) for subrow in axs for ax in subrow]
	plt.subplots_adjust(wspace=0, hspace=0)
	[axs[0, i].set_title(pn_list[i], fontsize = 12) for i in range(len(pn_list))]
	[axs[i, 0].set_ylabel(lhn_list[i], fontsize = 12) for i in range(len(lhn_list))]
	[axs[len(lhn_list)-1, i].tick_params('x', labelsize = 10) for i in range(len(pn_list))]
	[axs[i, 0].tick_params('y', labelsize = 10) for i in range(len(lhn_list))]
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
	avg_sim_traces.to_csv('figdata_avg_sim_traces_sc_v{}.csv'.format(str(version)))

	for i in range(len(sim_traces)):

		### plot simulated and experimental traces
		row = lhn_list.index(sim_traces[i]["lhn"])
		col = pn_list.index(sim_traces[i]["pn"])
		# read & plot experimental trace
		trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(sim_traces[i]["lhn"], sim_traces[i]["pn"]), header = None, dtype = np.float64)
		t_exp = trace_exp[0]  # don't adjust experimental rise, since sim rises at t=0
		v_exp = trace_exp[1]
		axs[row, col].plot(t_exp, v_exp, color = 'red', lw = 0.8)

		axs[row, col].plot(sim_traces[i]["t_sim"], [x+55 for x in sim_traces[i]["v_sim"]], color = 'grey', alpha = 0.2, lw = 0.4)

	props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(gsyn, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2")
	plt.suptitle(props + " [current params]", 
				 fontsize = 15, y = 0.96)

	plt.show()

###
### t1, v1 the simulated trace, t2, v2 the experimental trace to fit to
###
def find_error(t1, v1, t2, v2):

	# normalize both traces by peak of experimental trace
	v1_norm = (v1+55) / max(v2+55) 
	v2_norm = (v2+55) / max(v2+55)

	peak_err = np.abs(max(v2_norm) - max(v1_norm))

	return peak_err

###
### search for optimal biophysical parameter set
### vers = 1: same parameters for entire population
### vers = 2: parameters vary based on each LHN's surf area
###
def param_search_SC(vers):

	all_conns = pd.read_csv(conn_file)

	start_time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')

	# 20-12-11 tiny refresh of EPSP amplitudes to max of trace avgs (not avg of max)
	# 18 * 7 * 49 = 6174
	syn_strength_s = np.arange(0.01, 0.1, 0.005)  # nS
	c_m_s = np.arange(0.6, 1.21, 0.1) # uF/cm^2
	g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.1e-5) # S/cm^2, round to 6 places

	# 20-12-10 after refreshing EPSP peaks, local6 morphologies -- hopefully first of final
	# still likely need to tile other part of g_pas
	# 195 * 7 * 49 = 66885
	#syn_strength_s = np.arange(0.0025, 0.1, 0.0005) # nS
	#c_m_s = np.arange(0.6, 1.21, 0.1) # uF/cm^2
	#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.1e-5) # S/cm^2, round to 6 places
	# v1 fit: start time: 20-12-10-11:01:21, end time: 20-12-10 17:54:52
	# v2 fit: start time: 20-12-10-17:54:52, end time: 20-12-11 04:54:17

	# 20-09-26 after refreshing body IDs after Pasha's final revisions, saving all_resids
	#g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.1e-5) # S/cm^2, round to 6 places
	#c_m_s = np.arange(0.6, 1.21, 0.1) # uF/cm^2
	#syn_strength_s = np.arange(0.0025, 0.1, 0.0005) # nS, explore a broad range
	# v1 fit: start time: 20-09-27-21:04:20, end time: 20-09-28 04:21:23
	# v2 fit: start time: 20-09-28-04:21:23, end time: 20-09-28 10:37:06
	# sim_params and all_resids for v1 and v2 saved in '20-09-28_scfits_all_resids.out'

	# different types of errors evaluated (over all connections) per paramater set
	err_per_paramset = []
	# EPSP amplitude and kinetics per connection, for each parameter set
	sim_per_conn_per_paramset = pd.DataFrame()

	# iterate through parameter sets
	# iterate through all biophysical parameter combinations
	for syn_strength_i in syn_strength_s:
		for c_m_i in c_m_s:
			for g_pas_i in g_pas_s:

				### following errors skip local5/vl2a
				abs_norm_skip = 0 # sum of absolute values of normalized residuals
				abs_nonorm_skip = 0 # sum of absolute values of non-normalized residuals
				squ_norm_skip = 0 # sum of squares of normalized residuals
				
				### doesn't skip local5/vl2a
				abs_norm_noskip = 0 # sum of absolute values of normalized residuals
				abs_nonorm_noskip = 0 # sum of absolute values of non-normalized residuals
				squ_norm_noskip = 0 # sum of squares of normalized residuals
					
				sim_traces, sim_avgs, rss_err = find_peak_vals_SC(version = vers, cm = c_m_i, gsyn = syn_strength_i,
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
				params_toAppend.update(g_syn = syn_strength_i, g_pas = g_pas_i, 
								c_m = c_m_i,
								err_abs_norm_skip = abs_norm_skip, err_abs_norm_noskip = abs_norm_noskip,
								err_abs_nonorm_skip = abs_nonorm_skip,
								err_squ_norm_skip = squ_norm_skip, 
								err_squ_norm_noskip = squ_norm_noskip,
								err_abs_nonorm_noskip = abs_nonorm_noskip)

				# save overall statistics AND residuals per connection for this parameter set
				err_per_paramset.append(params_toAppend)
				# save EPSP prediction per connection
				if sim_per_conn_per_paramset.empty:
					sim_per_conn_per_paramset = sim_avgs
				else:
					sim_per_conn_per_paramset = sim_per_conn_per_paramset.append(sim_avgs)

				# likely don't need to update CSV every 20k params, bc fit is so fast
				if len(err_per_paramset) % 20000 == 1:
					pd.DataFrame(err_per_paramset).to_csv('{}_SCv{}_err_per_{}paramsets_temp.csv'.format(datetime.today().strftime("%y-%m-%d"), str(vers), str(len(err_per_paramset))))
					sim_per_conn_per_paramset.to_csv('{}_SCv{}_sim_per_conn_{}paramsets_temp.csv'.format(datetime.today().strftime("%y-%m-%d"), str(vers), str(len(err_per_paramset))))

		print("g_syn: finished with " + str(round(syn_strength_i, 5)) + " nS")

	err_per_paramset = pd.DataFrame(err_per_paramset)

	# plot lowest error trace: 
	#low_err = np.argsort(err_per_paramset["err_abs_nonorm_skip"])[0]
	#find_peak_vals(version = vers, cm = sim_params.loc[low_err, "c_m"], gsyn = sim_params.loc[low_err, "g_syn"],
	#				gpas = sim_params.loc[low_err, "g_pas"], save_excel = True, show_scatter = True, show_plot = True)
	#print("lowest error is {}".format(err_per_paramset.loc[low_err, "error_peak_notnorm"]))

	err_per_paramset.to_csv('{}_SCv{}_err_per_{}paramsets.csv'.format(datetime.today().strftime("%y-%m-%d"), str(vers), str(len(err_per_paramset))))
	sim_per_conn_per_paramset.to_csv('{}_SCv{}_sim_per_conn_{}paramsets.csv'.format(datetime.today().strftime("%y-%m-%d"), str(vers), str(len(err_per_paramset))))

	end_time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
	print("start time: {}, end time: {}".format(start_time, end_time))

	return err_per_paramset, sim_per_conn_per_paramset

# all_resids = second output of the method above, then can execute shelving: 
'''
	### to shelve something: 
	toshelve = ['all_resids']
	shelf_name = '20-09-27_scfit_v{}_all_resids.out'.format(str(vers))
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

	### to reopen:
	my_shelf = shelve.open(shelfname)
	for key in my_shelf:
		globals()[key]=my_shelf[key]
	my_shelf.close()
'''

###
### cross validation
### sim_params: dataframe: (# parameter sets) x (# of biophysical params, # of error types summed over all connections)
### all_resids: [# of parameter sets] -> 
####				each element contains raw residual (sim epsp peak - exp epsp peak) for each connection

### for 20-10-01 loocv v1 and v2 75075:
### myshelf= shelve.open('20-09-28_scfits_all_resids.out')
### cv1, cv2 = cross_val(sim_params_v1, all_resids_v1), cross_val(sim_params_v2, all_resids_v2)

def cross_val(sim_params, all_resids):

	# save best param sets from each cross-validated fold
	cv_top_params = []

	# for each iteration, leave out i'th connection in error term
	for i in range(all_resids[0].shape[0]):
		# track new error for each parameter set
		loocv_errors = []

		# iterate through each parameter set and recalculate error
		for j in range(sim_params.shape[0]):
			cv_error = 0

			# recalculate new error: 
			for k in range(all_resids[j].shape[0]):
				# skip connection that's being left out
				# currently using residual sum of absolute values, non-normalized
				if k != i:
					cv_error += np.abs(all_resids[j].resid[k])

			loocv_errors.append(cv_error)

		lowest_error_params_ind = np.array(loocv_errors).argsort()[0]
		lowest_error_params = sim_params.loc[lowest_error_params_ind]

		cv_top_params.append(dict(lowest_error_params))

	cv_top_params = pd.DataFrame(cv_top_params)

	# do some computation on the cross validated params
	# print out mean and SD of cross validated params
	print("top CV fits: c_m: {} p.m. {}".format(str(mean(cv_top_params.c_m)), str(np.std(cv_top_params.c_m))))
	print("across all params: c_m: {} p.m. {}".format(str(mean(sim_params.c_m)), str(np.std(sim_params.c_m))))

	print("top CV fits: g_pas: {} p.m. {}".format(str(mean(cv_top_params.g_pas)), str(np.std(cv_top_params.g_pas))))
	print("across all params: g_pas: {} p.m. {}".format(str(mean(sim_params.g_pas)), str(np.std(sim_params.g_pas))))

	print("top CV fits: g_syn: {} p.m. {}".format(str(mean(cv_top_params.g_syn)), str(np.std(cv_top_params.g_syn))))
	print("across all params: g_syn: {} p.m. {}".format(str(mean(sim_params.g_syn)), str(np.std(sim_params.g_syn))))

	return cv_top_params

###
### plot spread of optimal params from a LOOCV output

### cv1 and cv2 variables from previous cross validation of 75075 fits
### loaded 20-10-04_scfit_v1_v2_loocv.out
def plot_opt_params(cv):

	fig, axs = plt.subplots(nrows = 2, ncols = 2)

	g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.1e-5) # S/cm^2, round to 6 places
	c_m_s = np.arange(0.6, 1.21, 0.1) # uF/cm^2
	syn_strength_s = np.arange(0.0025, 0.1, 0.0005)

	bins = np.arange(0.0025, 0.1, 0.0005)
	axs[0,0].hist(cv.g_syn, bins)
	axs[0,0].set_xlabel("g_syn (uS)")
	axs[0,0].set_ylabel("freq.")

	bins = np.arange(1.0e-5, 6.5e-5, 0.1e-5) # S/cm^2, round to 6 places
	axs[0,1].hist(cv.g_pas, bins)
	axs[0,1].set_xlabel("g_pas (S/cm^2))")
	axs[0,1].set_ylabel("freq.")	

	bins = np.arange(0.6, 1.21, 0.1) # uF/cm^2
	axs[1,0].hist(cv.c_m, bins)
	axs[1,0].set_xlabel("c_m (uF/cm^2)")
	axs[1,0].set_ylabel("freq.")	

	#axs[1,1].scatter(sim_params["R_a"], sim_params[error])
	#axs[1,1].set_xlabel("R_a (ohm-cm)")
	#axs[1,1].set_ylabel("freq.")	

	plt.suptitle("Optimal parameter spread from L.O.O. Cross Validation")
	plt.show()

# for sc v1: yay, things are actually clustered!!
# top CV fits: c_m: 0.6 p.m. 0.0
#across all params: c_m: 0.8999999999999999 p.m. 0.1999999999999944
#top CV fits: g_pas: 6.13666666666667e-05 p.m. 2.414999424890667e-06
#across all params: g_pas: 3.700000000000002e-05 p.m. 1.5874507866388193e-05
#top CV fits: g_syn: 0.017933333333333332 p.m. 0.0005587684871413408
#across all params: g_syn: 0.051000000000000004 p.m. 0.028145455524235523

# 20-09-19 error includes NON-NORMALIZED absolute value of residual summed per connection, skipping local5/vl2a
#g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.2e-5) # S/cm^2, round to 6 places
#c_m_s = np.arange(0.6, 1.21, 0.1) # uF/cm^2
#syn_strength_s = np.arange(0.0025, 0.1, 0.0005) # nS, explore a broad range

# 20-09-15 error is absolute value of residual summed per connection
#g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.2e-5) # S/cm^2, round to 6 places
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#syn_strength_s = np.arange(0.0025, 0.1, 0.005) # nS, explore a broad range

# 20-09-10 revert local5 to v1.1 parameter search
# 20-09-12 rerun after error changed to average per connection
#g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.4e-5) # S/cm^2, round to 6 places
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#syn_strength_s = np.arange(0.0025, 0.1, 0.005) # nS, explore a broad range

# 20-09-08 initial parameter search
#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
#syn_strength_s = np.arange(0.0025, 0.1, 0.005) # nS, explore a broad range

# 20-09-08 initial runs:
#sim_params1, sim_params2 = param_search_v1(), param_search_v2()
#sim_params1.to_excel("20-09-08_scfit_v1_960.xlsx")
#sim_params2.to_excel("20-09-08_scfit_v2_960.xlsx")

'''
### using NEURON to build single compartment model
soma = h.Section(name='soma')
soma.L = 10 # to fill in
soma.diam = 10 # to fill in 
soma.Ra = 100 # to fill in
soma.nseg = 1 # maybe more? 
soma.cm = 1.2 

# add passive channels
mt = h.MechanismType(0)
mt.select("pas")
for section in h.allsec():
	# insert distributed mechanism into section
	mt.make(sec=section)	

for sec in h.allsec():
		for seg in sec.allseg():
			if hasattr(seg, 'pas'):
				seg.pas.e = epas
				seg.pas.g = gpas
'''