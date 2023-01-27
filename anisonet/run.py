#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:58:11 2022

@author: arash
"""

from anisonet.simulate import Simulate
import brian2 as b2

sim = Simulate('I_net_synaniso', scalar=2.5, 
               load_connectivity=1, 
               to_event_driven=1,)

sim.setup_net()

sim.start(duration=3000*b2.ms, batch_dur=500*b2.ms, 
            restore=False, profile=False, plot_snapshots=True)
sim.post_process()

# 
# for n in range(6):
#     sim.pops['I'].I_stim = 0*b2.nA
#     sim.start(duration=500*b2.ms, batch_dur=500*b2.ms, 
#             restore=False, profile=False, plot_snapshots=True)
#     sim.set_protocol()
#     sim.start(duration=500*b2.ms, batch_dur=500*b2.ms, 
#             restore=False, profile=False, plot_snapshots=True)

# # # sim.start(duration=5000*b2.ms, batch_dur=500*b2.ms, 
# # #         restore=False, profile=False, plot_snapshots=True)
# # sim.start(duration=3000*b2.ms, batch_dur=500*b2.ms, 
# #             restore=False, profile=False, plot_snapshots=True)
# sim.post_process(overlay=True)