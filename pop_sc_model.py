
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

# parameters:
num_syn = 2 # number of synapses
surf_area = 2230 # um^2, average of surf area of all LHNs in dataset, weighted by # of PN connections in dataset
tau1 = 0.2 # ms
tau2 = 1.1 # ms
g_pas = 4.4e-5 * (1/10000)**2 * surf_area * 1e9 # nS, note (1cm/10000um)
c_m = 1.2 * (1/10000)**2 * surf_area # uF
syn_strength = 0.035 # nS

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

def run_sim(syn_count = num_syn):
	v0 = -55
	t = np.linspace(0, 20, 40)

	sol = odeint(sc_neuron, v0, t, args = (g_pas, c_m, syn_strength, syn_count))

	plt.plot(t, sol[:,0])
	plt.show()

	return [t, sol[:,0]]

###
### search for optimal biophysical parameter set, same parameters for entire population
###
def param_search_v1():

	all_conns = pd.read_csv("20-08-27_all_conns.csv")

	# set parameter ranges
	g_pas_s = np.arange(1.0e-5, 5.8e-5, 0.4e-5) # S/cm^2, round to 6 places
	c_m_s = np.arange(0.6, 1.21, 0.2) # uF/cm^2
	syn_strength_s = np.arange(0.0000325, 0.00005, 0.000005) # uS

	# iterate through parameter sets

	# iterate through each connection

	# calculate total error

###
### search for optimal biophysical parameter set, parameters vary based on 
###
def param_search_v2():


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