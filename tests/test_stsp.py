import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

b2.start_scope()


eqs_nrn = '''
dv/dt = -v/tau_m : volt (unless refractory)
tau_m = 10*ms : second (shared)
'''

eqs_syn = '''
dg/dt = (-g)/tau_s : radian (clock-driven)
tau_s = 5*ms : second (shared)
w: 1
'''
spikes = [0, 5, 10, 25, 50, 100, 150, 200]*b2.ms
spikes += transient  # allow for some initial transient
P = b2.SpikeGeneratorGroup(1, np.zeros(len(spikes)), spikes)

G = b2.NeuronGroup(2, eqs_nrn, 'euler', 
                   threshold='v>15*mV', 
                   refractory=2*b2.ms, 
                   reset='v=0*mV')

H = b2.NeuronGroup(2, eqs_nrn, 'euler', 
                   threshold='v>15*mV', 
                   refractory=2*b2.ms, 
                   reset='v=0*mV')


S = b2.Synapses(G,H, eqs_syn, method='exact')

S.connect(i=[0,0], j=[1,0])

S.g = 'rand()'
S.w = 'rand()'
print(S.g)
print(S.w)
