import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

b2.start_scope()

taupre = taupost = 20*b2.ms
Apre = 0.9
Apost = -0.7

#Apre*taupre/taupost*1.5

# idxs = np.array([0, 0, 1, 1, 0, 1 ,0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
# spikes = np.array([0, 2, 5, 10, 20, 30])*b2.ms
# spikes = np.arange(len(idxs))*2*b2.ms

# P = b2.SpikeGeneratorGroup(2, idxs, spikes)
P = b2.PoissonGroup(2, 50*b2.Hz)
G = b2.NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0')

PG = b2.Synapses(P, G, on_pre='''v_post += 2''', method='linear')
PG.connect(j='i')

S = b2.Synapses(G, G,
             '''
             w : 1
             dapre/dt = -apre/taupre : 1 (clock-driven)
             dapost/dt = -apost/taupost : 1 (clock-driven)
             ''',
             on_pre='''
             apre = Apre
             w += w*apost
             ''',
             on_post='''
             apost = Apost
             w += (1-w)*apre
             ''', 
             method='exact')

S.connect('j!=i')
S.w = 0.5

M = b2.StateMonitor(S, ['w', 'apre', 'apost'], record=True)

b2.run(500*b2.ms)

fig, axs = plt.subplots(2,1, sharex=True)

for id in range(len(G)):
    axs[0].plot(M.t/b2.ms, M.apre[id], label=f'apre {id}', color=f'C{id}')
    axs[0].plot(M.t/b2.ms, M.apost[id],'--' , label=f'apost {id}', color=f'C{id}')
    axs[1].plot(M.t/b2.ms, M.w[id], label=f'w {id}->{(id+1)%2}', color=f'C{id}')
for ax in axs:
    ax.legend(loc='best', ncol=4)
    ax.grid()
    # ylims = ax.get_ylim()
    # ax.vlines(spikes[idxs==0]/b2.ms,*ylims, color='k', alpha=0.5)
    # ax.vlines(spikes[idxs==1]/b2.ms,*ylims, color='r', alpha=0.5)

ax.set_xlabel('Time (ms)');
