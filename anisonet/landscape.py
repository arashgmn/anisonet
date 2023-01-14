#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates different spatial landscapes for parameter pools used
in the `anisofy` module. There's nothing special about it except the fact that
any inhomogeneity in the landscape is forced to be balanced. i.e., the
distribution of all possible values over the entire field, is more or less, 
uniform, despite (possible) spatial correlations.

The following landscape types are supported:
    
    #. **None**: a uniformly random landscape
    #. **constant**: a homogenous landscape
    #. **Perlin-noise**: a landscape based on perlin distribution
    #. **numpy**: a landscape based on supported distribution by `[numpy]`_

Examples below illustrate valid landscape configurations and their structures:
    

.. code-block:: python
    
    None, # a uniformly random landscape between (0,1)
    .314,  # a homogenous landscape between of value 0.314
    {'type': 'perlin', 'args': {'scale': ...} }, # Perlin-based
    {'type': 'beta', 'args': {'a': ..., 'b': ...} }, # numpy beta
    {'type': 'binomial', 'args': {'n': ..., 'p': ...} }, # numpy binomial
    {'type': 'logistic', 'args': {'loc': ..., 'scale': ...} }, # numpy logisitc
    {'type': 'lognormal', 'args': {'mean': ..., 'sigma': ...} }, # numpy lognormal
    {'type': 'normal', 'args': {'mean': ..., 'sigma': ...} }, # numpy logisitc
    {'type': 'poisson', 'args': {'lam': ...} }, # numpy poissonian
    ... # other numpy distributions


.. _[numpy]: https://numpy.org/doc/stable/reference/random/generator.html#distributions

"""

import numpy as np
from noise import pnoise2 as perlin    
from pdb import set_trace
#TODO: emphasize in the docs that lanscape is reserved for anisotropy

def make_landscape(gs, lscp_cfg, vmin=-np.pi, vmax=np.pi, balance=False,
                   n_levels =None):
    """
    Makes a landscape according for a square grid of size ``gs`` using the 
    landscape configuration ``lscp_cfg`` which be either a constant (float) or
    a dictionary with ``'type'`` and ``'args'`` as key.
    
    For distibution with bounded range, it is possible to provide the minimum 
    and maximum values as well. The landscape, will be then linearly scaled to
    the desired range. 
    
    :param gs: grid size
    :type gs: int
    :param lscp_cfg: landscape configuration
    :type lscp_cfg: float or dictionary
    :param vmin: minimum value in the landscape, float, defaults to -np.pi
    :type vmin: float, optional
    :param vmax: maximum value in the landscape, float, defaults to -np.pi
    :type vmax: float, optional
    
    :return: flattened array
    :rtype: np.array of float

    """
    if type(lscp_cfg) != type({}):
        lscp = np.ones(gs**2)* lscp_cfg
    else:

        # extracting range: not applicable for numpy distros
        if ('vmin' in lscp_cfg) and ('vmax' in lscp_cfg):
            vmin = lscp_cfg['vmin']
            vmax = lscp_cfg['vmax']
        
        
        if lscp_cfg['type'] == 'random':
            lscp = np.random.uniform(vmin, vmax, size=gs**2)
        
        elif lscp_cfg['type'] =='perlin':
            assert 'scale' in lscp_cfg['args'],"scale value is needed for perlin landscape"
            
            s = lscp_cfg['args']['scale']
            x = y = np.linspace(0, s, gs)
            lscp = [[perlin(i, j, repeatx=s, repeaty=s) for j in y] for i in x]
            lscp = np.concatenate(lscp)
            
            # normalize to the range and balance the histogram
            lscp = balance_landscape(lscp)
            lscp -= lscp.min()
            lscp *= (vmax-vmin)/lscp.max()
            lscp += vmin
#            set_trace()

        else:
            distro = eval('np.'+lscp_cfg['type'])
            lscp = distro(**lscp_cfg['args'], size=gs**2)
        
            
    if n_levels != None:
        assert type(n_levels)==int
        _, lscp = np.histogram(lscp, bins=n_levels)
        
    return lscp

    # lscp = {}
    
    # if lscp_cfg == None:
    #     return lscp
    
    # else:
    #     for key, key_cfg in lscp_cfg['params'].items():
    #         if type(key_cfg) != type({}): # a single value (equivalent to homogeneous)
    #             lscp[key] = np.ones(gs**2) * key_cfg
            
    #         elif key_cfg['type']=='homogeneous':
    #             lscp[key] = np.ones(gs**2) * key_cfg['value']
            
    #         else:
    #             assert key!='r', "I don't know how to intialized r randomly for now."
    #             # TODO: Needs clarification of what does random or perlin mean for r
    #             # here the assumption is only phi is provided as random
                
    #             if key_cfg['type']=='random':
    #                 phis = np.random.uniform(-np.pi, np.pi, size=gs**2)
                    
    #             elif key_cfg['type']=='perlin':
    #                 assert 'scale' in key_cfg.keys(),"scale value is needed for perlin landscape"
    #                 scale = key_cfg['scale']
    #                 x = y = np.linspace(0, scale, gs)
    #                 phis = [[perlin(i, j, repeatx=scale, repeaty=scale) 
    #                          for j in y] for i in x]
    #                 phis = np.concatenate(phis)
                    
    #                 # to flatten the histogram of phis, one has to adjust the phi values
    #                 sorted_idx = np.argsort(phis)
    #                 max_val = gs * 2
    #                 idx = len(phis) // max_val
    #                 for ii, val in enumerate(range(max_val)):
    #                     phis[sorted_idx[ii * idx:(ii + 1) * idx]] = val
    #                 phis = (phis - gs) / gs
                    
    #                 # to push between -pi and pi
    #                 phis -= np.min(phis)
    #                 phis *= 2*np.pi/(np.max(phis)+1e-12)
    #                 phis -= np.pi
                
    #             lscp[key] = phis
        
    #     return lscp
    
    #     phis = bins[np.digitize(phis, bins, True)]
        
    
    # if config==None:
    # 	ls_type = 'iso'
    # else:	
    #     ls_type = config['type']
    #     ls_params = config['params']
    
    # if ls_type=='homogeneous':
    #     assert 'phi' in ls_params.keys(), "phi value is needed for landscape "+ls_type
    #     assert 'r' in ls_params.keys(), "r value is needed for landscape "+ls_type
        
    #     phis = np.ones(gs**2, dtype=float)*ls_params['phi']
    #     rs = np.ones(gs**2, dtype=int)*ls_params['r']
    
    # elif ls_type=='perlin':
    #     assert 'scale' in ls_params.keys(),"scale value is needed for landscape "+ls_type
    #     assert 'r' in ls_params.keys(), "r value is needed for landscape "+ls_type
        
    #     x = y = np.linspace(0, ls_params['scale'], gs)
    #     phis = [[perlin(i, j, repeatx=ls_params['scale'], repeaty=ls_params['scale']) 
    #              for j in y] for i in x]
    #     phis = np.concatenate(phis)
        
    #     # to flatten the histogram of phis, one has to adjust the 
    #     # phi values
    #     sorted_idx = np.argsort(phis)
    #     max_val = gs * 2
    #     idx = len(phis) // max_val
    #     for ii, val in enumerate(range(max_val)):
    #         phis[sorted_idx[ii * idx:(ii + 1) * idx]] = val
    #     phis = (phis - gs) / gs
        
    #     # to push between -pi and pi
    #     phis -= np.min(phis)
    #     phis *= 2*np.pi/(np.max(phis)+1e-12)
    #     phis -= np.pi
        
    #     rs = np.ones(gs**2, dtype=int)*ls_params['r']
   
    # elif ls_type=='random':
    #     phis = np.random.uniform(-np.pi, np.pi, size=gs**2)
    #     rs = np.ones(gs**2, dtype=int)*ls_params['r']
   
    # elif ls_type=='iso':
    #     phis = np.random.uniform(-np.pi, np.pi, size=gs**2)
    #     rs = np.zeros(gs**2, dtype=int)
        
    # else:
    #     raise 'Landscape type not recognized!'
    
    # if digitize:
    #     _, bins = np.histogram(phis, bins=8)
    #     phis = bins[np.digitize(phis, bins, True)]
        
    # return rs, phis

def balance_landscape(array):
    sorted_idx = np.argsort(array)
    gs = int(np.sqrt(len(array)))
    max_val = gs * 2
    idx = len(array) // max_val
    for i, val in enumerate(range(max_val)):
        array[sorted_idx[i * idx:(i + 1) * idx]] = val
    
    return (array- gs) / gs
    