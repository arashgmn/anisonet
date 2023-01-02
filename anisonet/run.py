#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:58:11 2022

@author: arash
"""

from simulate import *


sim = Simulate('I_net', scalar=2, 
               load_connectivity=0, 
               to_event_driven=1,)

sim.setup_net(init_cell='ss', )#init_syn='het')
sim.warmup()
sim.start(duration=2500*b2.ms, batch_dur=1000*b2.ms, 
            restore=False, profile=False, plot_snapshots=True)
sim.post_process(overlay=True)

#sim.warmup()

# idxs, _ = utils.get_line_idx(x0=30, y0=10, c=1, pop=sim.pops['I'])
# for post_idx in idxs:
#     pres = sim.syns['II'].j == post_idx
#     sim.syns['II'].U.__array__()[pres] = 0.1
    
# for i in range(1):
#     sim.set_protocol()
#     sim.start(duration=50*b2.ms, batch_dur=1000*b2.ms, 
#                restore=False, profile=False, plot_snapshots=False)
#     for pop in sim.pops.values():
#         pop.I_stim = 0*b2.pA
#         #pop.mu = 700*b2.pA
#         #pop.sigma=500*b2.pA
    
#     sim.start(duration=50*b2.ms, batch_dur=1000*b2.ms, 
#                 restore=False, profile=False, plot_snapshots=True)

#sim.start(duration=3000*b2.ms, batch_dur=1000*b2.ms, 
#            restore=False, profile=False, plot_snapshots=True)

# sim.start(duration=2000*b2.ms, batch_dur=1000*b2.ms, 
#            restore=False, profile=False, plot_snapshots=True)
    
# sim.net.run(3900*b2.ms, )
# viz.plot_firing_rates(sim, suffix='_all')
# sim.save_monitors(suffix ='all')

#sim.post_process(overlay=True)
# import matplotlib.pyplot as plt

# m = sim.mons[1]
# s = sim.syns[0]

# print(s.g[:11])
# for i in range(10):
#     plt.plot(m.t, m.x[i], 'r', alpha=1-i/10.)
#     plt.plot(m.t, m.u[i], 'b', alpha=1-i/10.)
#     plt.plot(m.t, m.g[i], 'g', alpha=1-i/10.)
# print(m.g.max(), m.g.min())

# plt.plot([],[], 'r', label='x',)
# plt.plot([],[], 'b', label='u',)
# plt.plot([],[], 'g', label='g',)
# plt.legend()
# EI_net
# sim = Simulate('EI_net', scalar=2.5, load_connectivity=False,
#                 voltage_base_syn=1)
# sim.setup_net()
# sim.warmup()
# sim.start(duration=2000*b2.ms, batch_dur=1000*b2.ms, 
#           restore=False, profile=False)
# sim.post_process()

# import matplotlib.pyplot as plt

# ms = sim.mons[1]; mp = sim.mons[2]
# plt.plot(mp.t[:100], mp.I_syn_I[0,:100]/mp.I_syn_I.max()/30)
# for i in np.where(sim.syns[0].j==0)[0]:
#     plt.plot(ms.t[:100], ms.g[i,:100])
