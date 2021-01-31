# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:40:29 2020

@author: Tony
"""

from brian2 import *
from brian2 import Morphology
from collections import OrderedDict, defaultdict, namedtuple
import matplotlib.pyplot as plt
#import mayavi.mlab as mayavi
import sympy
# brian2.test()

#import neurom as nm 
#from neurom import viewer
#nrn = nm.load_neuron("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\swc_files\\5813105722-local5unhealed-corrected3.swc")
#fig, ax = viewer.draw(nrn, mode = '3d')
#fig.show()

def plot_morphology2D(morpho, axis, color_switch=False):
    if isinstance(morpho, Soma):
        ax.plot(morpho.x/um, morpho.y/um, 'o', color='darkred',
                ms=morpho.diameter/um, mec='none')
    else:
        coords = morpho.coordinates
        if color_switch:
            color = 'darkblue'
        else:
            color = 'darkred'
        ax.plot(coords[:, 0]/um, coords[:, 1]/um, color='black',
                lw=2)
        # dots at the center of the compartments
        ax.plot(morpho.x/um, morpho.y/um, 'o', color=color,
                mec='none', alpha=0.75)

    for child in morpho.children:
        plot_morphology2D(child, axis=axis, color_switch= not color_switch)

def from_swc_file(filename, spherical_soma=True):
        '''
        Load a morphology from a ``SWC`` file. A large database of morphologies
        in this format can be found at http://neuromorpho.org
        The format consists of an optional header of lines starting with ``#``
        (ignored), followed by a sequence of points, each described in a line
        following the format::
            index type x y z radius parent
        ``index`` is an integer label (starting at 1) that identifies the
        current point and increases by one each line. ``type`` is an integer
        representing the type of the neural segment. The only type that changes
        the interpretation by Brian is the type ``1`` which signals a soma.
        Types ``2`` (axon), ``3`` (dendrite), and ``4`` (apical dendrite) are
        used to give corresponding names to the respective sections. All other
        types are ignored. ``x``, ``y``, and ``z`` are the cartesian coordinates
        at each point and ``r`` is its radius. ``parent`` refers to the index
        of the parent point or is ``-1`` for the root point.
        Parameters
        ----------
        filename : str
            The name of the ``SWC`` file.
        spherical_soma : bool, optional
            Whether to model the soma as a sphere.
        Returns
        -------
        morpho : `Morphology`
            The morphology stored in the given file.
        '''
        swc_types = defaultdict(lambda: None)
        # The following names will be translated into names, all other will be
        # ignored
        swc_types.update({'1': 'soma', '2': 'axon', '3': 'dend', '4': 'apic'})

        with open(filename, 'r') as f:
            points = []
            for line_no, line in enumerate(f):
                line = line.strip()
                if line.startswith('#') or len(line) == 0:
                    # Ignore comments or empty lines
                    continue
                splitted = line.split()
                if len(splitted) != 7:
                    raise ValueError('Each line of an SWC file has to contain '
                                     '7 space-separated entries, but line %d '
                                     'contains %d.' % (line_no + 1,
                                                       len(splitted)))
                index, comp_type, x, y, z, radius, parent = splitted
                points.append((int(index),
                               swc_types[comp_type],
                               float(x)/12,
                               float(y)/125,
                               float(z)/125,
                               2*float(radius)/125,
                               int(parent)))

        return Morphology.from_points(points, spherical_soma=spherical_soma)

local5 = from_swc_file("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\swc_files\\5813105722-local5unhealed-corrected3.swc")
# total_compartments = 6857, return self.n + sum(c.total_compartments for c in self.children)

# todo: perhaps modify the from_points function to output all the compartments,
# so that I can index compartments without recursing through all parents
# this'll be critical to assigning synapses

fig, ax = plt.subplots()
plot_morphology2D(local5, ax)

gL = 1e-4*siemens/cm**2 # membrane conductance (?) 
EL = -70*mV             # leak reversal potential
eqs='''
Im = gL * (EL - v) : amp/meter**2
Is = gs * (Es - v) : amp (point current)
gs : siemens
'''
neuron = SpatialNeuron(morphology=local5, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
S = Synapses(stimulation, neuron, model='''dg/dt = -g/taus : siemens
                                           gs_post = g : siemens (summed)''',
             on_pre='g += w')
#
S.connect(i=0, j=50)
S.connect(i=1, j=100)
S.connect(i=0, j=morpho[25*um])
S.connect(i=1, j=morpho.axon[30*um])

# to inject current into the soma;
neuron.I[0] = 1*nA

"""
Created on Sun May 17 18:06:00 2020

@author: jamesjeanne
"""

seed(11922)  # to get identical figures for repeated runs


################################################################################
# Model parameters
################################################################################
### General parameters
duration = 0.2*second  # Total simulation time
sim_dt = 0.1*ms        # Integrator/sampling step
N_e = 3200             # Number of excitatory neurons
N_i = 800              # Number of inhibitory neurons

### Neuron parameters
E_l = -60*mV           # Leak reversal potential
g_l = 1.5*nS          # Leak conductance: 9.99*nS
E_e = 0*mV             # Excitatory synaptic reversal potential
E_i = -80*mV           # Inhibitory synaptic reversal potential

C_m = 15*pF           # Membrane capacitance 198*pF
tau_e = 10*ms           # Excitatory synaptic time constant

tau_r = 5*ms           # Refractory period
I_ex = 0*pA          # External current
V_th = -40*mV          # Firing threshold
V_r = E_l              # Reset potential

### Synapse parameters
w_e = 0.27*nS          # Excitatory synaptic conductance
w_i = 1.0*nS           # Inhibitory synaptic conductance
U_0 = 1              # Synaptic release probability at rest
Omega_d = 2.0/second   # Synaptic depression rate
Omega_f = 3.33/second  # Synaptic facilitation rate

################################################################################
# Model definition
################################################################################
# Set the integration time (in this case not strictly necessary, since we are
# using the default value)

start_scope()
defaultclock.dt = sim_dt

### Neurons
neuron_eq = '''
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + I_ex)/C_m    : volt (unless refractory)
dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance
'''
neurons = NeuronGroup(1, model=neuron_eq,
                      threshold='v>V_th', reset='v=V_r',
                      refractory='tau_r', method='euler')


indices = array([0])
times = array([100])*ms
G = SpikeGeneratorGroup(1, indices, times)


#Define conductance-based synapse model:
synapses_eqs = '''
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S     : 1 (event-driven)
# Fraction of synaptic neurotransmitter resources available:
dx_S/dt = Omega_d *(1 - x_S) : 1 (event-driven)
'''

#r_S is the release of 

synapses_action = '''
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
'''

#connect spike generator to neuron:
#syn = Synapses(G, neurons, model=synapses_eqs, on_pre=synapses_action+'g_e_post += w_e*r_S')
syn = Synapses(G, neurons, model=synapses_eqs, on_pre='g_e_post += w_e')

#Now connect the synapses...

syn.connect(i=0, j = 0)
# Start from "resting" condition: all synapses have fully-replenished
# neurotransmitter resources
syn.x_S = 1
neurons.v = -60*mV


spike_mon = SpikeMonitor(G)
state_mon = StateMonitor(neurons, ['v', 'g_e'], record=0)
synapse_mon = StateMonitor(syn, ['u_S', 'x_S'], record=syn[0, :], when='after_synapses')

# ##############################################################################
# # Simulation run
# ##############################################################################

run(duration, report='text')
plot(state_mon.t/ms, state_mon.v[0]/mV, color='black')

plot(state_mon.t/ms, state_mon.g_e[0]/nS, color='black')

plot(spike_mon.t[exc_mon.i <= N_e//4]/ms,
           exc_mon.i[exc_mon.i <= N_e//4], '|', color='C0')


# I think that the below is unused code I copied from a tutorial from the Brian2 website.
# JMJ 2020-06-05.


################################################################################
# Analysis and plotting
################################################################################
plt.style.use('figures.mplstyle')


plot(exc_mon.t[exc_mon.i <= N_e//4]/ms,
           exc_mon.i[exc_mon.i <= N_e//4], '|', color='C0')
plot(inh_mon.t[inh_mon.i <= N_i//4]/ms,
           inh_mon.i[inh_mon.i <= N_i//4]+N_e//4, '|', color='C1')



### Spiking activity (w/ rate)
fig1, ax = plt.subplots(nrows=2, ncols=1, sharex=False,
                        gridspec_kw={'height_ratios': [3, 1],
                                     'left': 0.18, 'bottom': 0.18, 'top': 0.95,
                                     'hspace': 0.1},
                        figsize=(3.07, 3.07))
ax[0].plot(exc_mon.t[exc_mon.i <= N_e//4]/ms,
           exc_mon.i[exc_mon.i <= N_e//4], '|', color='C0')
ax[0].plot(inh_mon.t[inh_mon.i <= N_i//4]/ms,
           inh_mon.i[inh_mon.i <= N_i//4]+N_e//4, '|', color='C1')
pu.adjust_spines(ax[0], ['left'])
ax[0].set(xlim=(0.,duration/ms), ylim=(0,(N_e+N_i)//4), ylabel='neuron index')

# Generate frequencies
bin_size = 1*ms
spk_count, bin_edges = np.histogram(np.r_[exc_mon.t/ms, inh_mon.t/ms],
                                    int(duration/ms))
rate = double(spk_count)/(N_e + N_i)/bin_size/Hz
ax[1].plot(bin_edges[:-1], rate, '-', color='k')
pu.adjust_spines(ax[1], ['left', 'bottom'])
ax[1].set(xlim=(0.,duration/ms), ylim=(0, 10.),
          xlabel='time (ms)', ylabel='rate (Hz)')
pu.adjust_ylabels(ax, x_offset=-0.18)

### Dynamics of a single neuron
fig2, ax = plt.subplots(4, sharex=False,
                       gridspec_kw={'left': 0.27, 'bottom': 0.18, 'top': 0.95,
                                    'hspace': 0.2},
                       figsize=(3.07, 3.07))
### Postsynaptic conductances
ax[0].plot(state_mon.t/ms, state_mon.g_e[0]/nS, color='C0')
ax[0].plot(state_mon.t/ms, -state_mon.g_i[0]/nS, color='C1')
ax[0].plot([state_mon.t[0]/ms, state_mon.t[-1]/ms], [0, 0], color='grey',
           linestyle=':')
# Adjust axis
pu.adjust_spines(ax[0], ['left'])
ax[0].set(xlim=(0., duration/ms), ylim=(-5.0,0.25),
          ylabel='postsyn.\nconduct.\n(${0}$)'.format(sympy.latex(nS)))

### Membrane potential
ax[1].axhline(V_th/mV, color='C2', linestyle=':')  # Threshold
# Artificially insert spikes
ax[1].plot(state_mon.t/ms, state_mon.v[0]/mV, color='black')
ax[1].vlines(exc_mon.t[exc_mon.i == ni]/ms, V_th/mV, 0, color='black')
pu.adjust_spines(ax[1], ['left'])
ax[1].set(xlim=(0., duration/ms), ylim=(-1+V_r/mV,0.),
          ylabel='membrane\npotential\n(${0}$)'.format(sympy.latex(mV)))

### Synaptic variables
# Retrieves indexes of spikes in the synaptic monitor using the fact that we
# are sampling spikes and synaptic variables by the same dt
spk_index = np.in1d(synapse_mon.t, exc_mon.t[exc_mon.i == ni])
ax[2].plot(synapse_mon.t[spk_index]/ms, synapse_mon.x_S[0][spk_index], '.',
           ms=4, color='C3')
ax[2].plot(synapse_mon.t[spk_index]/ms, synapse_mon.u_S[0][spk_index], '.',
           ms=4, color='C4')
# Super-impose reconstructed solutions
time = synapse_mon.t  # time vector
tspk = Quantity(synapse_mon.t, copy=True)  # Spike times
for ts in exc_mon.t[exc_mon.i == ni]:
    tspk[time >= ts] = ts
ax[2].plot(synapse_mon.t/ms, 1 + (synapse_mon.x_S[0]-1)*exp(-(time-tspk)*Omega_d),
           '-', color='C3')
ax[2].plot(synapse_mon.t/ms, synapse_mon.u_S[0]*exp(-(time-tspk)*Omega_f),
           '-', color='C4')
# Adjust axis
pu.adjust_spines(ax[2], ['left'])
ax[2].set(xlim=(0., duration/ms), ylim=(-0.05, 1.05),
          ylabel='synaptic\nvariables\n$u_S,\,x_S$')

nspikes = np.sum(spk_index)
x_S_spike = synapse_mon.x_S[0][spk_index]
u_S_spike = synapse_mon.u_S[0][spk_index]
ax[3].vlines(synapse_mon.t[spk_index]/ms, np.zeros(nspikes),
             x_S_spike*u_S_spike/(1-u_S_spike))
pu.adjust_spines(ax[3], ['left', 'bottom'])
ax[3].set(xlim=(0., duration/ms), ylim=(-0.01, 0.62),
          yticks=np.arange(0, 0.62, 0.2), xlabel='time (ms)', ylabel='$r_S$')

pu.adjust_ylabels(ax, x_offset=-0.20)


plt.show()