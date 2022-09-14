#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module `anisotropifies` the network by sampling the post-synapses in an 
specific manner defined in `[1]`_. I have seperated this utility, in case we 
decided to develope other structural connectivities.

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
"""

import numpy as np

def draw_post_syns(s_coord, ncons,
                  srow, scol, trow, tcol, 
                  profile, shift={}, 
                  self_link= False, recurrent=True, gap=2):
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

    .. note::
        Locally connected random networks (LCRNs) exhibit a minimum distance of
        1 between the pre- and post-neuron. In other words, no self-connection
        is permitted by default in the LCRN (`[3]`_). However, anisotropic 
        nets, due to the manual displacement of the targets' cloud center, are
        capable of exhiiting a self-link.
        
        Following `[1]`_, we considered a minimum distance of 2 between the 
        source and targets when initializing the network. Existence of such a 
        gap, ensures non-monotonicity in the connectivity profile, which in 
        turn leads to stable bumps of activation in random networks, for 
        instance (`[3]`_). In genral, the interplay between the gap and the 
        connectivity profile (Gaussian or Gamma) determines to bump's pattern,
        robustness, and movement (`[4]`_).
        
    .. _[3]: https://doi.org/10.1523/ENEURO.0348-16.2017
    .. _[4]: http://epubs.siam.org/doi/10.1137/030600040
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

        # LCRN network has a minimum gap of 1 between source and targets.
        # If these lines don't exist, the bumps won't be pronounced. Also look
        # at our refernce [1].
        radius[radius<0] -= gap
        radius[radius>=0]+= gap
        
        # draw coordinates
        x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
        y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
        # make periodic
        x[redraw] = x_ % tcol
        y[redraw] = y_ % trow
        
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
