import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2


eqs = '''
dv/dt = -v/tau + stimuli(t,i)/C : volt
tau : second (shared)
C: farad (shared)
#stimuli: amp
'''

N = 50
stimuli = b2.TimedArray(np.vstack([[i, i, i, 0, 0] for i in range(N)]).T* b2.nA, dt=10*b2.ms)

pop = b2.NeuronGroup(N, eqs, 'euler', namespace={'stimuli': stimuli})
pop.v = 0*b2.mV
pop.C = 100*b2.pF
pop.tau = 30*b2.ms

mon= b2.StateMonitor(pop, ['v'], True)

b2.start_scope()
net = b2.Network()
net.add([pop])
net.add([mon])

net.run(200*b2.ms)

fig, axs = plt.subplots(2,1)
for i in range(5):
    axs[0].plot(mon.t, mon.v[i,:], label=str(i))
    axs[1].plot(mon.t, stimuli(mon.t,i), label=str(i))

for ax in axs:
    ax.legend()
