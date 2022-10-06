#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module `anisotropifies` the network by sampling the post-synapses in an 
specific manner defined in `[1]`_. Here, we provide two approaches which differ
in connection-free enforcement mechanism (reak below). Please note that these 
two functions lead to different connecitivty profiles due to their inherent 
difference, even with the exact same arguments, although their relative 
overhead, is practically negligible. Other approaches might be added later on.


In locally connected random networks (LCRNs), nearest neighbors have a
higher connection probability. However, if no autosynapse is allowed, 
one can enforce a connection-free region around each neuron--what we
call a gap. Existence of such connection-free zone, ensures 
non-monotonicity in the connectivity profile, which in turn, leads to 
stable bumps of activation in random networks, for instance `[3]`_. 
In genral, the interplay between the gap and the connectivity profile 
(Gaussian or Gamma) determines to spatio-temporal pattern, its movement
, and robustnesss `[4]`_.


There are several ways to enforce such a gap.
One simple method is draw potential post-synapses according to the 
connectivity profile, and only permit those whose distance from the
neuron is larger than the gap. Those who fail this criterion must be 
redrawn. 

Approach of `[1]`_, however is slightly different. They radially 
displace those neurons which fall in the connection-free zone to the 
outside of this region by increasing their distance from the pre-
synaptic neuron. Therefore, this approach does not need any resampling.
However, it alters the pdf of the connectivity profile by sliding the 
near-zero events forward. Importantly, since the anisotropy is 
introduced with a displacing the post-synaptic cloud, when the 
displacement vector is zero, this approach may cause unintended 
self-connections. `[1]`_ prevented this by explicitly excluding those 
connections (look `here`_) which in turn, may change the total number 
of out-degrees. 

Following `[1]`_, we considered a minimum distance of 1 between the 
source and targets when initializing the network. 
    
    
.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
.. _[3]: https://doi.org/10.1523/ENEURO.0348-16.2017
.. _[4]: http://epubs.siam.org/doi/10.1137/030600040
.. _here: https://github.com/babsey/spatio-temporal-activity-sequence/blob/bb82c7d4d3b2f85e9d2635a5479d0478868d33bb/scripts/lib/connection_matrix.py#L28

"""

import numpy as np

def draw_post_syns(s_coord, ncons,
                  srow, scol, trow, tcol, 
                  profile, shift={}, 
                  self_link= False, recurrent=True, gap=1.1):
    """Samples indices of the source neuron `s_coord` in the target network 
    according to a location-dependent profile. The profile by default is 
    localized on the location of the source neuron. Yet, it is possible to 
    shift to any location relative to the source neuron on the target network. 
    The location of the source neuron on the target network will be 
    interpolated if the size of source and target networks don't match.

    Periodic boundary condition is applied.

    :param s_coord: source coordinate index as a tuple of ints
    :type s_coord: tuple
    :param ncons: number of out-links
    :type ncons: int
    :param srow: number of rows in source network
    :type srow: int
    :param scol: number of columns in source network
    :type scol: int
    :param trow: number of rows in target network
    :type trow: int
    :param tcol: number of columns in target network
    :type tcol: int
    :param profile: connectivity configuration. Please refer to 
        :ref:`configs:Profile` for details.
    :type profile: dict
    :param shift: shift configuration Please refer to 
        :ref:`configs:Anisotropy` for details.
    :type shift: dict
    :param self_link: Is self-link permitted; Matter only if source and target
        populations are the same.
    :type self_link: bool
    :param recurrent: Whether or not the pathway is a recurent one
    :type recurrent: bool
    :param gap: The gap between the pre and post neurons. Look below for 
        details.
    :type gap: int
    
    :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
    :return: source and target coordinate indices (in target network coordinates) 
    :rtype: tuple   
    """
    
    # just for convenience
    if profile== None:
        w_type = None        
        shift = {'r':0, 'phi':0}
    else:
        w_type = profile['type']
        p_param = profile['params']
        
    
    # dispalcement can be anything, but it will be rounded to the grid points
    shift_x = shift['r'] * np.cos(shift['phi']) 
    shift_y = shift['r'] * np.sin(shift['phi'])     
    
    # source neuron's location on the target population
    scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
    s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
    # here we draw source-centeric components of target cells, and shift them
    # according to the desired displacement vector. If the source and target 
    # populations are the same, i.e., connection is recurrent, and some of the
    # drawn targets land on the source, we draw ignore them, redraw again and 
    # again until none of the targets terminate on the source. If self-link is
    # permitted, we don't perform this iterative step and all initially drawn
    # (and displaced) targets will be accepted.
    
    x = np.zeros(ncons, dtype=int)
    y = np.zeros(ncons, dtype=int)
    redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    
    while sum(redraw)>0:
        ncon = sum(redraw)
        
        if w_type==None:
            x_ = np.random.randint(-tcol//2, tcol-tcol//2, ncon)
            y_ = np.random.randint(-(trow-trow//2), trow//2, ncon)
            
        else:
            # angle        
            alpha = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
            
            # radius
            if w_type=='Gaussian':
                radius = p_param['std'] * np.random.randn(ncon)
            elif w_type=='Gamma':
                radius = np.concatenate(
                    (-np.random.gamma(shape= p_param['kappa'], 
                                      scale= p_param['theta'], 
                                      size= int(ncon // 2)),
                    +np.random.gamma(shape=p_param['kappa'], 
                                     scale=p_param['theta'], 
                                     size=ncon -int(ncon // 2)))
                    )
            else:
                raise NotImplementedError
    
            # LCRN network has a minimum gap of 1 between source and targets.
            # If these lines don't exist, the bumps won't be pronounced. Also look
            # at our refernce [1].
            radius[radius<0] -= gap
            radius[radius>=0]+= gap
            
            # draw coordinates
            x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
            y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
        # make periodic
        # x[redraw] = x_ % tcol
        # y[redraw] = y_ % trow
        # make periodic around the presynapse
        x[redraw] = (x_ + tcol/2) % tcol - tcol/2
        y[redraw] = (y_ + trow/2) % trow - trow/2
        
        
        # mark those self links for redraw, if necessary
        if (recurrent) and (not self_link):
            redraw = (x==0) & (y==0)
        else:
            redraw = np.zeros_like(x, dtype=bool)
            
    # translating the coordinates w.r.t. source coordinates
    x = (x + s_coord[0]) % tcol
    y = (y + s_coord[1]) % trow
    t_coords = np.array([x,y]).T 
    
    return s_coord, t_coords.astype(int)


def draw_post_syns_simple(s_coord, ncons,
                  srow, scol, trow, tcol, 
                  profile, shift={}, 
                  self_link= False, recurrent=True, gap=1.1):
    """Samples indices of the source neuron `s_coord` in the target network 
    according to a location-dependent profile. The profile by default is 
    localized on the location of the source neuron. Yet, it is possible to 
    shift to any location relative to the source neuron on the target network. 
    The location of the source neuron on the target network will be 
    interpolated if the size of source and target networks don't match.

    Periodic boundary condition is applied.

    :param s_coord: source coordinate index as a tuple of ints
    :type s_coord: tuple
    :param ncons: number of out-links
    :type ncons: int
    :param srow: number of rows in source network
    :type srow: int
    :param scol: number of columns in source network
    :type scol: int
    :param trow: number of rows in target network
    :type trow: int
    :param tcol: number of columns in target network
    :type tcol: int
    :param profile: connectivity configuration. Please refer to 
        :ref:`configs:Profile` for details.
    :type profile: dict
    :param shift: shift configuration Please refer to 
        :ref:`configs:Anisotropy` for details.
    :type shift: dict
    :param self_link: Is self-link permitted; Matter only if source and target
        populations are the same.
    :type self_link: bool
    :param recurrent: Whether or not the pathway is a recurent one
    :type recurrent: bool
    :param gap: The gap between the pre and post neurons. Look below for 
        details.
    :type gap: int
    
    :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
    :return: source and target coordinate indices (in target network coordinates) 
    :rtype: tuple   
    """
    
    # just for convenience
    p_param = profile['params']
    w_type = profile['type']
    
    # dispalcement can be anything, but it will be rounded to the grid points
    shift_x = shift['r'] * np.cos(shift['phi']) 
    shift_y = shift['r'] * np.sin(shift['phi'])     
    
    # source neuron's location on the target population
    scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
    s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
    # here we draw source-centeric components of target cells, and shift them
    # according to the desired displacement vector. If the source and target 
    # populations are the same, i.e., connection is recurrent, and some of the
    # drawn targets land on the source, we draw ignore them, redraw again and 
    # again until none of the targets terminate on the source. If self-link is
    # permitted, we don't perform this iterative step and all initially drawn
    # (and displaced) targets will be accepted.
    
    
    x = np.zeros(ncons, dtype=int)
    y = np.zeros(ncons, dtype=int)
    redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    
    while sum(redraw)>0:
        ncon = sum(redraw)

        # angle        
        alpha = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
        
        # radius
        if w_type=='Gaussian':
            radius = p_param['std'] * np.random.randn(ncon)
        elif w_type=='Gamma':
            radius = np.concatenate(
                (-np.random.gamma(shape= p_param['kappa'], 
                                  scale= p_param['theta'], 
                                  size= int(ncon // 2)),
                +np.random.gamma(shape=p_param['kappa'], 
                                 scale=p_param['theta'], 
                                 size=ncon -int(ncon // 2)))
                )
        else:
            raise NotImplementedError

        # draw coordinates
        x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
        y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
        # make periodic around the presynapse
        x_ = (x_ + tcol/2) % tcol - tcol/2
        y_ = (y_ + trow/2) % trow - trow/2
        
        x[redraw] = x_ 
        y[redraw] = y_
        
        # mark those self links for redraw, if necessary
        if (recurrent) and (not self_link):
            dist = np.sqrt(x**2+y**2)
            redraw = dist<=gap
        else:
            redraw = np.zeros_like(x, dtype=bool)
            
    # translating the coordinates w.r.t. source coordinates
    x = (x + s_coord[0]) % tcol
    y = (y + s_coord[1]) % trow
    t_coords = np.array([x,y]).T 
    
    return s_coord, t_coords.astype(int)
