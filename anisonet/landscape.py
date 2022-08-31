#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a library of different landscapes. By `landscape`, I mean
the anisotropy angle $\phi$ and shift $r$ defined in `[1]`_. The followings are
merely a reimplementation of ``connectivity_landscape.py`` file of `[1]`_, except
a few assertion.

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432. 
"""

import numpy as np
from noise import pnoise2 as perlin    

def make_landscape(gs, ls_type='random', ls_params={}):
    
    if ls_type=='homogeneous':
        assert 'phi' in ls_params.keys(), "phi value is needed for landscape "+ls_type
        assert 'r' in ls_params.keys(), "r value is needed for landscape "+ls_type
        
        phis = np.ones(gs**2, dtype=float)*ls_params['phi']
        rs = np.ones(gs**2, dtype=int)*ls_params['r']
    
    elif ls_type=='perlin':
        assert 'scale' in ls_params.keys(),"scale value is needed for landscape "+ls_type
        assert 'r' in ls_params.keys(), "r value is needed for landscape "+ls_type
        
        x = y = np.linspace(0, ls_params['scale'], gs)
        phis = [[perlin(i, j, repeatx=ls_params['scale'], repeaty=ls_params['scale']) 
                 for j in y] for i in x]
        phis = np.concatenate(phis)
        
        # to flatten the histogram of phis, one has to adjust the 
        # phi values
        sorted_idx = np.argsort(phis)
        max_val = gs * 2
        idx = len(phis) // max_val
        for ii, val in enumerate(range(max_val)):
            phis[sorted_idx[ii * idx:(ii + 1) * idx]] = val
        phis = (phis - gs) / gs
        
        # to push between -pi and pi
        phis -= np.min(phis)
        phis *= 2*np.pi/(np.max(phis)+1e-12)
        phis -= np.pi
        
        rs = np.ones(gs**2, dtype=int)*ls_params['r']
   
    elif ls_type=='random':
        phis = np.random.uniform(-np.pi, np.pi, size=gs**2)
        rs = np.ones(gs**2, dtype=int)*ls_params['r']
   
    elif ls_type=='symmetric':
        phis = np.random.uniform(-np.pi, np.pi, size=gs**2)
        rs = np.zeros(gs**2, dtype=int)
        
    else:
        raise 'Landscape type not recognized!'
    
    return rs, phis
