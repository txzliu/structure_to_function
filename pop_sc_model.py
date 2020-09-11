
###
### for each unique PN to LHN connection, simulate a single compartment cell
###

import sys
sys.path.append("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model")
from run_local5 import *

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
	dvdt = 1e-6 * 1/c_m * (-1*syn_count*syn_strength*factor*(exp(-t/tau2) - exp(-t/tau1))*(v-v_rev) - g_pas*(v-v_rest))
	#print(dvdt)
	return dvdt

def run_sim(g_pas = g_pas_def, c_m = c_m_def, syn_strength = syn_strength_def, syn_count = num_syn_def):
	v0 = -55
	t = np.linspace(0, 20, 41)
	sol = odeint(sc_neuron, v0, t, args = (g_pas, c_m, syn_strength, syn_count))

	plt.plot(t, sol[:,0])
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

def find_peak_vals(version, gpas = 5e-5, cm = 0.8, gsyn = 0.0175):

	all_conns = pd.read_csv("20-08-27_all_conns.csv")

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

	g_pas_i = gpas # S/cm^2
	c_m_i = cm # uF/cm^2
	syn_strength_i = gsyn # nS, peak synaptic conductance

	# iterate through all connections
	for i in range(len(all_conns)):

		num = all_conns.num_syn[i]
		if version==1:
			surf_area = 2230 # weighted average surf area of all LHN's
		elif version==2:
			surf_area = all_conns.lhn_SA[i] # unique SA for each LHN body ID

		### plot experimental traces
		row = lhn_list.index(all_conns.lhn[i])
		col = pn_list.index(all_conns.pn[i])
		# read & plot experimental trace
		trace_exp = pd.read_csv('exp_traces\\{}_{}.csv'.format(all_conns.lhn[i], all_conns.pn[i]), header = None, dtype = np.float64)
		t_exp = trace_exp[0]+1.25 # slightly adjust to align with rise time of EPSP
		v_exp = trace_exp[1]
		axs[row, col].plot(t_exp, v_exp, color = 'red')

		if num > 0:

			# calculate overall (non-surf-area-normalized) parameters
			g_pas_o = g_pas_i * (1/10000)**2 * surf_area * 1e9 # nS, note (1cm/10000um)
			c_m_o = c_m_i * (1/10000)**2 * surf_area # uF

			t_sim = np.linspace(0, 20, 41)
			v_sim = run_sim(g_pas=g_pas_o, c_m=c_m_o, syn_strength=syn_strength_i, syn_count=num)

			all_conns.loc[i, 'epsp_sim'] = float(max(v_sim)+55)

			axs[row, col].plot(t_sim, v_sim + 55, color = 'black')

		else:
			all_conns.loc[i, 'epsp_sim'] = 0 # EPSP size = 0 if no synapses

	props = ("g_pas = " + str(g_pas_i) + " S/cm^2, g_syn = " + str(round(syn_strength_i, 5)) + 
			" nS, c_m = " + str(c_m_i) + " uF/cm^2")
	plt.suptitle(props + " [current params]", 
				 fontsize = 24, y = 0.96)

	plt.show()

	all_conns.to_excel('{}_scsim_v{}.xlsx'.format(date.today().strftime("%Y-%m-%d"), str(version)))

###
### t1, v1 the simulated trace, t2, v2 the experimental trace to fit to
### weight error from VL2a x3 to normalize amplitudes
###
def find_error(t1, v1, t2, v2):

	# normalize both traces by peak of experimental trace
	v1_norm = (v1+55) / max(v2+55) 
	v2_norm = (v2+55) / max(v2+55)

	peak_err = np.abs(max(v2_norm) - max(v1_norm))

	return peak_err

###
### search for optimal biophysical parameter set, same parameters for entire population
###
def param_search_v1():

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
					surf_area = 2349 # weighted average surf area of all LHN's, 2349 after local5 to v1.1

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

			print("g_syn: finished with " + str(round(syn_strength_i, 5)) + " nS")

	sim_params = pd.DataFrame(sim_params)

	return sim_params

# for the optimal parameter set, run and extract peak values for all connections, plot vs MC model

###
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