#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module `anisotropifies` the network by sampling the post-synapses in an 
specific manner defined in `[1]`_. I have seperated this utility, in case we 
decided to develope other structural connectivities.

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
"""

import numpy as np

def get_post_syns(s_coord, ncons,
                  srow, scol, trow, tcol, 
                  profile, shift={}, 
                  self_link= False, recurrent=True):
    """Samples indices of the source neuron `s_coord` in the target network 
    according to a location-dependent profile. The profile by default is 
    localized on the location of the source neuron. Yet, it is possible to 
    shift to any location relative to the source neuron on the target network. 
    The location of the source neuron on the target network will be 
    interpolated if the size of source and target networks don't match.

    Periodic boundary condition is applied.

    :param s_coord: source coordinate index as a tuple of ints
    :type s_coord: tuple
    :type ncons: int
    :param srow: number of rows in source network
    :type srow: int
    :param scol: number of columns in source network
    :type scol: int
    :param trow: number of rows in target network
    :type trow: int
    :param tcol: number of columns in target network
    :type tcol: int
    :param ncons: number of out-links
    :param profile: connectivity configuration. Read below for more info.
    :type profile: dict
    :param shift: shift configuration Read below for more info.
    :type shift: dict
    :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
    :return: source and target coordinate indices (in target network coordinates) 
    :rtype: tuple

    .. note:
        Connectivity dictionary must have the following structure:
        `profile = {'type':'Gaussian', 'params':{'std': 10}, 'self_link':False}`
        or 
        `profile = {'type':'Gamma', 'params':{'k': 3, 'theta':4}, 'self_link':False}`
        Types, other than `Gaussian` and `Gamma` are not accepted.

    .. note:
        A shift can be prescribed by a dictionary that follows this structure:
        `shift_cfg = {'r':5, 'phi':np.pi/4}`
        This will move the targets location accordingly.        
    """
    
    wparams = profile['params']
    wtype = profile['type']
    
    
    shift_x = shift['r'] * np.cos(shift['phi']) 
    shift_y = shift['r'] * np.sin(shift['phi'])     
    
    # source neuron location on the target net
    scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
    s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
    # here we draw (source-centered) components of targets. we assume 
    # self-links are not permitted, and draw as many links as needed
    # until no self-link exist anymore. 
    x = np.zeros(ncons, dtype=int)
    y = np.zeros(ncons, dtype=int)
    
    redraw = (x==0) & (y==0)
    while sum(redraw)>0:
        ncon = sum(redraw)
        #print('Sampling', ncon, 'targets.')
        
        phi = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
        #phi=shift['phi']
        if wtype=='Gaussian':
            radius = wparams['std'] * np.random.randn(ncon)
        elif wtype=='Gamma':
            radius = np.concatenate(
                (-np.random.gamma(shape= wparams['kappa'], 
                                  scale= wparams['theta'], 
                                  size= int(ncon // 2)),
                +np.random.gamma(shape=wparams['kappa'], 
                                 scale=wparams['theta'], 
                                 size=ncon -int(ncon // 2)))
                )
        else:
            raise NotImplementedError
        
        #print(redraw, s_coord, ncons)
        #if len(redraw)==4:
        #    set_trace()
        x_ = np.round(radius*np.cos(phi) + shift_x).astype(int)
        y_ = np.round(radius*np.sin(phi) + shift_y).astype(int)
        
        x[redraw] = x_ % tcol
        y[redraw] = y_ % trow
        
        if (recurrent) and (not self_link):
            redraw = (x==0) & (y==0)
        else:
            redraw = np.zeros_like(x, dtype=bool)
            
    # adding the source index
    x = (x + s_coord[0]) % tcol
    y = (y + s_coord[1]) % trow
    t_coords = np.array([x,y]).T 
    
    return s_coord, t_coords.astype(int)



def get_post_syns_spreizer(s_coord, ncons,
                  srow, scol, trow, tcol, 
                  profile, shift={}, 
                  self_link= False, recurrent=True):
    
    wparams = profile['params']
    wtype = profile['type']
    
    shift_x = shift['r'] * np.cos(shift['phi']) 
    shift_y = shift['r'] * np.sin(shift['phi'])     
    
    # source neuron location on the target net
    scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
    s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
    # here we draw (source-centered) components of targets. we assume 
    # self-links are not permitted, and draw as many links as needed
    # until no self-link exist anymore. 
    x = np.zeros(ncons, dtype=int)
    y = np.zeros(ncons, dtype=int)
    
    redraw = (x==0) & (y==0)
    while sum(redraw)>0:
        ncon = sum(redraw)
        #print('Sampling', ncon, 'targets.')
        
        phi = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
        #phi=shift['phi']
        if wtype=='Gaussian':
            radius = wparams['std'] * np.random.randn(ncon)
        elif wtype=='Gamma':
            radius = np.concatenate(
                (-np.random.gamma(shape= wparams['kappa'], 
                                  scale= wparams['theta'], 
                                  size= int(ncon // 2)),
                +np.random.gamma(shape=wparams['kappa'], 
                                 scale=wparams['theta'], 
                                 size=ncon -int(ncon // 2)))
                )
        else:
            raise NotImplementedError
        
        if not self_link:
            radius[radius>=0] += 1
            radius[radius<0] -= 1
            
        #print(redraw, s_coord, ncons)
        #if len(redraw)==4:
        #    set_trace()
        x_ = np.round(radius*np.cos(phi) + shift_x).astype(int)
        y_ = np.round(radius*np.sin(phi) + shift_y).astype(int)
        
        x[redraw] = x_ % tcol
        y[redraw] = y_ % trow
        
        if (recurrent) and (not self_link):
            redraw = (x==0) & (y==0)
        else:
            redraw = np.zeros_like(x, dtype=bool)
            
    # adding the source index
    x = (x + s_coord[0]) % tcol
    y = (y + s_coord[1]) % trow
    t_coords = np.array([x,y]).T 
    
    return s_coord, t_coords.astype(int)
