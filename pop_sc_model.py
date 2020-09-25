
###
### for each unique PN to LHN connection, simulate a single compartment cell
###

import sys
sys.path.append("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model")
from run_local5 import *
plt.rcParams.update({'font.size': 30})

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
from datetime import date

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

def find_peak_vals(version, gpas = 5e-5, cm = 0.8, gsyn = 0.0175, 
					show_plot = False, show_scatter = False, save_excel = False):

	all_conns = pd.read_csv("20-08-27_all_conns.csv")

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
			surf_area = 2314 # weighted average surf area of all LHN's
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
		plot_traces(sim_traces, cm, gsyn, gpas)
	if save_excel:
		all_conns.to_excel('{}_scsim_v{}_each_inst.xlsx'.format(date.today().strftime("%y-%m-%d"), str(version)))
		sim_avgs.to_excel('{}_scsim_v{}_avgs.xlsx'.format(date.today().strftime("%y-%m-%d"), str(version)))
	if show_scatter:
		plt.rcParams["figure.figsize"] = (15,15)

		plt.scatter(all_conns.loc[:, 'epsp_exp'], all_conns.loc[:, 'epsp_sim'], color = 'black')
		plt.scatter(sim_avgs.loc[:, 'epsp_exp'], sim_avgs.loc[:, 'epsp_sim'], color = 'red')
		plt.xlabel("experimental EPSP peak (mV)")
		plt.ylabel("simulated EPSP peak (mV)")
		plt.plot([0, 7], [0, 7])
		props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(gsyn, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2")
		plt.suptitle(props + " [current params]", 
				 fontsize = 24, y = 0.96)
		plt.show()

	return sim_traces, sim_avgs, sum_peak_err

def plot_traces(sim_traces, cm, gsyn, gpas):

	plt.rcParams["figure.figsize"] = (40,35)

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

	for i in range(len(sim_traces)):

		### plot simulated and experimental traces
		row = lhn_list.index(sim_traces[i]["lhn"])
		col = pn_list.index(sim_traces[i]["pn"])
		# read & plot experimental trace
		trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(sim_traces[i]["lhn"], sim_traces[i]["pn"]), header = None, dtype = np.float64)
		t_exp = trace_exp[0]+1.25 # slightly adjust to align with rise time of EPSP
		v_exp = trace_exp[1]
		axs[row, col].plot(t_exp, v_exp, color = 'red')

		axs[row, col].plot(sim_traces[i]["t_sim"], [x+55 for x in sim_traces[i]["v_sim"]], color = 'black')

	props = ("g_pas = " + str(gpas) + " S/cm^2, g_syn = " + str(round(gsyn, 4)) + 
			" nS, c_m = " + str(cm) + " uF/cm^2")
	plt.suptitle(props + " [current params]", 
				 fontsize = 24, y = 0.96)

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
def param_search(vers):

	all_conns = pd.read_csv("20-08-27_all_conns.csv")

	# 20-09-19 error includes NON-NORMALIZED absolute value of residual summed per connection, skipping local5/vl2a
	g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.2e-5) # S/cm^2, round to 6 places
	c_m_s = np.arange(0.6, 1.21, 0.1) # uF/cm^2
	syn_strength_s = np.arange(0.0025, 0.1, 0.0005) # nS, explore a broad range

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

	sim_params = []

	# iterate through parameter sets
	# iterate through all biophysical parameter combinations
	for syn_strength_i in syn_strength_s:
		for c_m_i in c_m_s:
			for g_pas_i in g_pas_s:

				### all errors skip local5/vl2a
				sum_peak_err = 0 # sum of absolute values of normalized residuals
				sum_peak_err_notnorm = 0 # sum of absolute values of non-normalized residuals
				sum_peak_err_rss = 0 # sum of squares of normalized residuals
				
				### doesn't skip local5/vl2a
				sum_peak_err_noskip = 0 # sum of absolute values of normalized residuals
				sum_peak_err_notnorm_noskip = 0 # sum of absolute values of non-normalized residuals
					
				sim_traces, sim_avgs, rss_err = find_peak_vals(version = vers, cm = c_m_i, gsyn = syn_strength_i,
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
				params_toAppend.update(g_syn = syn_strength_i, g_pas = g_pas_i, 
								c_m = c_m_i,
								error_peak = sum_peak_err, error_peak_noskip = sum_peak_err_noskip,
								error_peak_notnorm = sum_peak_err_notnorm,
								error_peak_rss = sum_peak_err_rss)

				sim_params.append(params_toAppend)

		print("g_syn: finished with " + str(round(syn_strength_i, 5)) + " nS")

	sim_params = pd.DataFrame(sim_params)

	# plot lowest error trace: 
	low_err = np.argsort(sim_params["error_peak_notnorm"])[0]
	#find_peak_vals(version = vers, cm = sim_params.loc[low_err, "c_m"], gsyn = sim_params.loc[low_err, "g_syn"],
	#				gpas = sim_params.loc[low_err, "g_pas"], save_excel = True, show_scatter = True, show_plot = True)
	print("lowest error is {}".format(sim_params.loc[low_err, "error_peak_notnorm"]))

	sim_params.to_excel('{}_scfit_v{}_{}.xlsx'.format(date.today().strftime("%y-%m-%d"), str(vers), str(len(sim_params))))

	return sim_params

# for the optimal parameter set, run and extract peak values for all connections, plot vs MC model

### DEPRECATED:
### search for optimal biophysical parameter set, parameters vary based on each LHN's surf area
###
def param_search_v2():
	
	all_conns = pd.read_csv("20-08-27_all_conns.csv")

	# 20-09-10 revert local5 to v1.1 parameter search
	g_pas_s = np.arange(1.0e-5, 6.5e-5, 0.4e-5) # S/cm^2, round to 6 places
	c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
	syn_strength_s = np.arange(0.0025, 0.1, 0.005) # nS, explore a broad range

	# 20-09-08 initial parameter search
	#g_pas_s = np.arange(1.0e-5, 5.9e-5, 0.4e-5) # S/cm^2, round to 6 places
	#c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
	#syn_strength_s = np.arange(0.0025, 0.1, 0.005) # nS, explore a broad range

	sim_params = []

	# iterate through parameter sets
	# iterate through all biophysical parameter combinations
	for syn_strength_i in syn_strength_s:
		for c_m_i in c_m_s:
			for g_pas_i in g_pas_s:

				sum_peak_err = 0
				# iterate through all connections
				for i in range(len(all_conns)):

					num = all_conns.num_syn[i]
					surf_area = all_conns.lhn_SA[i] # unique SA for each LHN body ID

					if num > 0:

						# calculate overall (non-surf-area-normalized) parameters
						g_pas_o = g_pas_i * (1/10000)**2 * surf_area * 1e9 # nS, note (1cm/10000um)
						c_m_o = c_m_i * (1/10000)**2 * surf_area # uF

						t_sim = np.linspace(0, 20, 41)
						v_sim = run_sim(g_pas=g_pas_o, c_m=c_m_o, syn_strength=syn_strength_i, syn_count=num)

						# read experimental trace
						trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(all_conns.lhn[i], all_conns.pn[i]), header = None, dtype = np.float64)
						t_exp = trace_exp[0]+1.25 # slightly adjust VA6 to align with rise time of EPSP
						v_exp = trace_exp[1]-55

						# calculate error of v_s to experimental trace
						peak_err = find_error(t_sim, v_sim, t_exp, v_exp)

						# increment sum_peak_err
						sum_peak_err += peak_err
					else:
						continue # if no synapses, don't register error (doesn't vary w/ params)

				# save parameter values, (output trace indices), fit errors
				params_toAppend = {}
				params_toAppend.update(g_syn = syn_strength_i, g_pas = g_pas_i, 
								c_m = c_m_i,
								error_peak = sum_peak_err)

				sim_params.append(params_toAppend)

			print("g_syn: finished with " + str(str(round(syn_strength_i, 5))) + " nS")

	sim_params = pd.DataFrame(sim_params)

	return sim_params

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