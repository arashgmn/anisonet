#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:58:11 2022

@author: arash
"""

from anisonet.simulate import Simulate
import brian2 as b2



sim = Simulate('I_net', scalar=2, load_connectivity=1, to_event_driven=1, name='')
sim.setup_net(init_cell='ss', init_syn='ss') # ss = steady state initialization
sim.set_protocol() # adding stimulation to the network

sim.syns['II'].delay = 1 * b2.ms # Turns out that this is very critical!

sim.start(duration=1000*b2.ms, batch_dur=200*b2.ms, 
          restore=False, profile=False, plot_snapshots=True)
sim.post_process()
            