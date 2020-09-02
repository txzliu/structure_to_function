
###
### for each unique PN to LHN connection, simulate a single compartment cell
###

import sys
sys.path.append("C:\\Users\\Tony\\Documents\\TonyThings\\Research\\Jeanne Lab\\code\\EManalysis\\LH dendritic computation\\mc_model")
from run_local5 import *

soma = h.Section(name='soma')
soma.L = 10 # to fill in
soma.diam = 10 # to fill in 
soma.Ra = 100 # to fill in
soma.nseg = 1 # maybe more? 
soma.cm = 1.2 
'''
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