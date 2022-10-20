#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a library of different landscapes. By `landscape`, I mean
the anisotropy angle $\phi$ and shift $r$ defined in `[1]`_. The followings are
merely a reimplementation of ``connectivity_landscape.py`` file of `[1]`_, except
a few assertion.

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
"""

import numpy as np
from noise import pnoise3 as perlin    

def make_landscape(gs, config, digitize=False, dim=2) :
    """
    Makes a landscape according for neural network placed on a square grid of 
    size ``gs``. The landscape has two components `rs` (the radial 
    displacement) and `phis` (the anisotropic bias angle). `phis` are forced to
    have a flat histogram. In other words, even if `phis` are not equal, their
    frequency is the same over the grid. `phis` are always between -pi to pi.
    
    :param gs: grid size
    :type gs: int
    :param ls_type: landscape name. Only ['random', 'perlin', 'symmetric', 'homogeneous'] 
        are allowed., defaults to 'random'
    :type ls_type: str, optional
    :param ls_params: A dictionary specifying the parameters of the desired 
        landsacepe., defaults to {}
    :type ls_params: dict, optional
    
    :return: (rs, phis) as flattened arrays 
    :rtype: tuple of np.array of floats

    .. note::
        Perlin noise package, by default doesn't provide a balanced landscape. i.e., some
        angles are more frequent than others. To have a landscape comparable with the 
        symmetric and random cases, similar to `[1]`_ we flatten the disribution before 
        passing the landscape field to the network.  
    """
    if config==None:
    	ls_type = 'iso'
    else:	
        ls_type = config['type']
        ls_params = config['params']
    
    
    if ls_type=='homogeneous':
        assert 'phi' in ls_params.keys(), "phi value is needed for landscape "+ls_type
        assert 'r' in ls_params.keys(), "r value is needed for landscape "+ls_type
        
        phis = np.ones(gs**dim, dtype=float)*ls_params['phi']
        rs = np.ones(gs**dim, dtype=int)*ls_params['r']
    
    elif ls_type=='perlin':
        assert 'scale' in ls_params.keys(),"scale value is needed for landscape "+ls_type
        assert 'r' in ls_params.keys(), "r value is needed for landscape "+ls_type
        
        
        x = y = z = np.linspace(0, ls_params['scale'], gs)
        phis = [[[perlin(i, j, k, 
                        repeatx=ls_params['scale'], 
                        repeaty=ls_params['scale'],
                        repeatz=ls_params['scale']) 
                 for k in z] for j in y] for i in x]
        
        phis = np.array(phis).reshape((gs,gs,gs))
        if dim==1:
            phis = phis[0,0,:]
        elif dim==2:
            phis = phis[0,:,:]
        else:
            pass
        phis = np.ravel(phis)
        
        # to flatten the histogram of phis, one has to adjust the phi values
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
        
        rs = np.ones(gs**dim, dtype=int)*ls_params['r']
   
    elif ls_type=='random':
        phis = np.random.uniform(-np.pi, np.pi, size=gs**dim)
        rs = np.ones(gs**dim, dtype=int)*ls_params['r']
   
    elif ls_type=='iso':
        phis = np.random.uniform(-np.pi, np.pi, size=gs**dim)
        rs = np.zeros(gs**dim, dtype=int)
        
    else:
        raise 'Landscape type not recognized!'
    
    if digitize:
        _, bins = np.histogram(phis, bins=8)
        phis = bins[np.digitize(phis, bins, True)]
        
    return rs, phis
