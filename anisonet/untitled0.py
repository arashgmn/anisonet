#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 20:28:20 2023

@author: arash
"""
from brian2.units import pA, mV, ms, pF, nA
scalar = 3

pops_cfg = {
    'I': {'gs': 10, 
          'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
          'cell': {'type': 'LIF', 
                   'params': {
                        'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                        'tau': 10*ms, 'C': 250*pF
                        }
                   }
          }
    }

conn_cfg = {
    'II': {'ncons': 9, 'self_link':False, 
           'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4}},
           #'profile': {'type':'Gaussian', 'params': {'std':3, 'gap': max(2, 6./scalar) }},
           #'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
           #'synapse': {'type':'alpha_jump', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
           #'synapse': {'type':'exp_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
           #'synapse': {'type':'exp_jump', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
           'synapse': {'type':'biexp_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau_r': 5*ms, 'tau_d': 5*ms}},
           #'synapse': {'type':'biexp_jump', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau_d': 5*ms, 'tau_r': 5*ms}},
           #'synapse': {'type':'tsodyks-markram_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms,'tau_d': 5*ms, 'tau_f': 5*ms}},
           #'synapse': {'type':'tsodyks-markram_jump', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau_d': 5*ms, 'tau_f': 5*ms}},
           }
    }


stims_cfg = {},

nonuniformity = {
    'II': {
        'synaptic': 'sin', 
        'connectivity': 'shift-rotate', 
        }
    }


lscps_cfg ={
    'II': {'r'  : 1, 
            'phi': {'type': 'perlin', 'args': {'scale':3, 'vmin': 0., 'vmax':3.14} }, 
            'U': {'type': 'random', 'args': {'vmin': 0.01, 'vmax':0.3}},
            'q': {'type': 'perlin', 'args': {'vmin': 0., 'vmax':3.14} }, 
            }
         }  

stim_cfg = {}