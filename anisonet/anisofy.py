#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What is anisotropy, really?
===========================
We assume every neuron is somehow different from its surrounding, in that it is
biased in one form or other toward on direction. Such angular bias, thus, makes
up so-called anisotropy landscape, which is nothing but a visualization of the
extend of this bias for each neuron in the spatial field.

For instance, `[1]`_ introduced a directional bias (anisotropy) in the 
connectivity of locally connected neurons. Similarly, we can introduce an
angular heterogeneity in the synaptic parameters (think of the parameters of 
the `kernel` dicussed in :ref:`equations:Kernel`). Probably other forms are 
possible too, but in this package, we stick to these two only. So in what 
follows, we discuss these two form of anisotropy and how can we produce them.


.. note::
    In this madule we only explain the imposition of anisotropy. For how such 
    neuron-specific biases are spread in space, please refer to 
    :ref:`landscape`.


Connectivity heterogeneity
==========================

Isotropic profile
-----------------
It is common to assume that the probability of establishing a link between two
neurons depends on their distance. As such, the connecitivity profile from any
presynapse will be an isometric sphere centered on the locaiton of presynapse.
Such isotropy even holds for situations in which neurons are connected 
uniformly regardless of their distance. 

We refer this radial connecitivty pattern simply as ``profile``. The following
profiles are supported: 

    #. uniform (homogeneous)
    #. Gaussian distribution
    #. Gamma distribution
    
Not specifying any profile will be fallen back to the ``uniform`` case
automatically.

.. note::
    One particularly interesting connectivity pattern is Mexcan hat profile (aka
    difference of Gaussians) which implements proximal excitation and lateral
    inhibition (or vice versa, depending on which one is subtracted from the 
    other). This profile is known to give rise to activity bumps in networks.
    
    In case of networks with single synaptic types (only inhibitory or 
    excitatory) which are driven by a background input of the opposite polarity
    it is possible to emulate the maxican hot profile by introducing a central
    ``gap`` in the connectivity profile. As such, ``gap`` ensures that there is
    an effective connecitivty difference between nearby neurons and distals 
    ones. By default, it is set to zero, but use it if your network does not 
    opposite synaptic types.
    
    
Anisotropic connectivity
------------------------
The connecitivty pattern introduced earlier is isotropic. We can break this 
isotropy in the following ways:
    
    #. ``shift``: Shifts the postsynaptic cloud, as introduced by `[1]`_. In this case,
       the connections are simply shifted in space by a displacement vector, which
       is specific to each neuron. Thus, the anisotropy landscape is characterized
       by displacement length :math:`r(\\textbf x)` and anisotropy angle 
       :math:`\phi(\\textbf x)`, in which :math:`\\textbf x` shows the location 
       dependence.
    #. ``squeeze-rotate``: Squeezes the postsynaptic cloud and then rotate it. 
       This method, too, introduces anisotropy using two factors. Axis squeezing 
       ratio :math:`r(\\textbf x)` and the anisotropy angle :math:`\phi(\\textbf x)`.
       Note that in this method, the outcomes of rotation angles :math:`\\theta` and 
       :math:`2\pi - \\theta` are indentical. 
    #. ``positive-rotate``: Mirros the connecitions with negative x-component to 
       the positive-x, and then rotate them. This method only requires the roation 
       angle :math:`\phi(\\textbf x)`.
    #. ``positive-squeeze-rotate``: Mixes the last two methods by squeezing the 
       reflected-to-positive connections before rotating. Thus, it requires both 
       the angle :math:`\phi(\\textbf x)` and the axis squeezing ratio 
       :math:`r(\\textbf x)` for full characterization.


The figure below displays the effect of these method:
    
    
    

Synaptic heterogeneity
======================

Isotropic synaptic parameterization
-----------------------------------
Synapses are often initialized either indentically or completely randomly. 
Either case are isotropic and are refered here as:
    
    #. ``rand``: uniformly initializes the parameters between 0 and 1 or 
       between the minimum and maximum acceptable values for non-normalized
       quantities.
    #. ``ss``:  initialized all the varaibles in accord to their steady-state
       equilibrated value. In case of plasticitiy dependent values, steady-state
       is defined under the `no-usage` condition.
    

Anisotropic synaptic parameterization
-------------------------------------
The anisotropy in the synapses in imposed based on their relative direction 
w.r.t. the presynapse (which is computed for each neuron individually). Given
the direction of each postsynapse (:math:`\\theta`), and neuron-specific angular
bias (:math:`\phi(\\textbf x)`) the anisotropy is applied according to a 
trianglumertic transformation:

    #. ``sin``: modulates the variable between its minium and maximum range 
       accodring to :math:`\\frac{1+\sin (\\theta-\phi(\\textbf x) )}{2}`
    #. ``cos``: modulates the variable between its minium and maximum range 
       accodring to :math:`\\frac{1+\cos (\\theta-\phi(\\textbf x) )}{2}`
       
.. note::
    Other anisotropic transformations are under development.


    
.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
.. _[3]: https://doi.org/10.1523/ENEURO.0348-16.2017
.. _[4]: http://epubs.siam.org/doi/10.1137/030600040
.. _here: https://github.com/babsey/spatio-temporal-activity-sequence/blob/bb82c7d4d3b2f85e9d2635a5479d0478868d33bb/scripts/lib/connection_matrix.py#L28

"""

from pdb import set_trace
import numpy as np
from anisonet.utils import pre_loc2post_loc_rel 

# def draw_post_syns(s_coord, ncons,
#                   srow, scol, trow, tcol, 
#                   profile, shift={}, 
#                   self_link= False, recurrent=True, gap=5,
#                   p_WS = 0.37):
#     """Samples indices of the source neuron `s_coord` in the target network 
#     according to a location-dependent profile. The profile by default is 
#     localized on the location of the source neuron. Yet, it is possible to 
#     shift to any location relative to the source neuron on the target network. 
#     The location of the source neuron on the target network will be 
#     interpolated if the size of source and target networks don't match.

#     Periodic boundary condition is applied.

#     :param s_coord: source coordinate index as a tuple of ints
#     :type s_coord: tuple
#     :param ncons: number of out-links
#     :type ncons: int
#     :param srow: number of rows in source network
#     :type srow: int
#     :param scol: number of columns in source network
#     :type scol: int
#     :param trow: number of rows in target network
#     :type trow: int
#     :param tcol: number of columns in target network
#     :type tcol: int
#     :param profile: connectivity configuration. Please refer to 
#         :ref:`configs:Profile` for details.
#     :type profile: dict
#     :param shift: shift configuration Please refer to 
#         :ref:`configs:Anisotropy` for details.
#     :type shift: dict
#     :param self_link: Is self-link permitted; Matter only if source and target
#         populations are the same.
#     :type self_link: bool
#     :param recurrent: Whether or not the pathway is a recurent one
#     :type recurrent: bool
#     :param gap: The gap between the pre and post neurons. Look below for 
#         details.
#     :type gap: int
    
#     :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
#     :return: source and target coordinate indices (in target network coordinates) 
#     :rtype: tuple   
#     """
    
#     # just for convenience
#     if profile== None:
#         w_type = None        
#         shift = {'r':0, 'phi':0}
#     else:
#         w_type = profile['type']
#         p_param = profile['params']
        
    
#     # the Watts-Strogatz p modulator
#     if np.random.rand()>p_WS:
#     	 shift['r'] = 0 # only shift a fraction p_WS of cells 
    
#     # dispalcement can be anything, but it will be rounded to the grid points
#     shift_x = shift['r'] * np.cos(shift['phi']) 
#     shift_y = shift['r'] * np.sin(shift['phi'])     
    
#     # source neuron's location on the target population
#     scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
#     s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
#     # here we draw source-centeric components of target cells, and shift them
#     # according to the desired displacement vector. If the source and target 
#     # populations are the same, i.e., connection is recurrent, and some of the
#     # drawn targets land on the source, we draw ignore them, redraw again and 
#     # again until none of the targets terminate on the source. If self-link is
#     # permitted, we don't perform this iterative step and all initially drawn
#     # (and displaced) targets will be accepted.
    
#     x = np.zeros(ncons, dtype=int)
#     y = np.zeros(ncons, dtype=int)
#     redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    
#     while sum(redraw)>0:
#         ncon = sum(redraw)
        
#         if w_type==None:
#             x_ = np.random.randint(-tcol//2, tcol-tcol//2, ncon)
#             y_ = np.random.randint(-(trow-trow//2), trow//2, ncon)
            
#         else:
#             # angle        
#             alpha = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
            
#             # radius
#             if w_type=='Gaussian':
#                 radius = p_param['std'] * np.random.randn(ncon)
#             elif w_type=='Gamma':
#                 radius = np.concatenate(
#                     (-np.random.gamma(shape= p_param['kappa'], 
#                                       scale= p_param['theta'], 
#                                       size= int(ncon // 2)),
#                     +np.random.gamma(shape=p_param['kappa'], 
#                                      scale=p_param['theta'], 
#                                      size=ncon -int(ncon // 2)))
#                     )
#             else:
#                 raise NotImplementedError
    
#             # LCRN network has a minimum gap of 1 between source and targets.
#             # If these lines don't exist, the bumps won't be pronounced. Also look
#             # at our refernce [1].
#             radius[radius<0] -= gap
#             radius[radius>=0]+= gap
            
#             # draw coordinates
#             x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
#             y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
#         # make periodic
#         # x[redraw] = x_ % tcol
#         # y[redraw] = y_ % trow
#         # make periodic around the presynapse
#         x[redraw] = (x_ + tcol/2) % tcol - tcol/2
#         y[redraw] = (y_ + trow/2) % trow - trow/2
        
        
#         # mark those self links for redraw, if necessary
#         if (recurrent) and (not self_link):
#             redraw = (x==0) & (y==0)
#         else:
#             redraw = np.zeros_like(x, dtype=bool)
            
#     # translating the coordinates w.r.t. source coordinates
#     x = (x + s_coord[0]) % tcol
#     y = (y + s_coord[1]) % trow
#     t_coords = np.array([x,y]).T 
    
#     return s_coord, t_coords.astype(int)


# def draw_post_syns_simple(s_coord, ncons,
#                   srow, scol, trow, tcol, 
#                   profile, shift={}, 
#                   self_link= False, recurrent=True, gap=1.1):
#     """Samples indices of the source neuron `s_coord` in the target network 
#     according to a location-dependent profile. The profile by default is 
#     localized on the location of the source neuron. Yet, it is possible to 
#     shift to any location relative to the source neuron on the target network. 
#     The location of the source neuron on the target network will be 
#     interpolated if the size of source and target networks don't match.

#     Periodic boundary condition is applied.

#     :param s_coord: source coordinate index as a tuple of ints
#     :type s_coord: tuple
#     :param ncons: number of out-links
#     :type ncons: int
#     :param srow: number of rows in source network
#     :type srow: int
#     :param scol: number of columns in source network
#     :type scol: int
#     :param trow: number of rows in target network
#     :type trow: int
#     :param tcol: number of columns in target network
#     :type tcol: int
#     :param profile: connectivity configuration. Please refer to 
#         :ref:`configs:Profile` for details.
#     :type profile: dict
#     :param shift: shift configuration Please refer to 
#         :ref:`configs:Anisotropy` for details.
#     :type shift: dict
#     :param self_link: Is self-link permitted; Matter only if source and target
#         populations are the same.
#     :type self_link: bool
#     :param recurrent: Whether or not the pathway is a recurent one
#     :type recurrent: bool
#     :param gap: The gap between the pre and post neurons. Look below for 
#         details.
#     :type gap: int
    
#     :raises NotImplementedError: If the weigh profiles is not Gaussian or Gamma
#     :return: source and target coordinate indices (in target network coordinates) 
#     :rtype: tuple   
#     """
    
#     # just for convenience
#     p_param = profile['params']
#     w_type = profile['type']
    
#     # dispalcement can be anything, but it will be rounded to the grid points
#     shift_x = shift['r'] * np.cos(shift['phi']) 
#     shift_y = shift['r'] * np.sin(shift['phi'])     
    
#     # source neuron's location on the target population
#     scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
#     s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
#     # here we draw source-centeric components of target cells, and shift them
#     # according to the desired displacement vector. If the source and target 
#     # populations are the same, i.e., connection is recurrent, and some of the
#     # drawn targets land on the source, we draw ignore them, redraw again and 
#     # again until none of the targets terminate on the source. If self-link is
#     # permitted, we don't perform this iterative step and all initially drawn
#     # (and displaced) targets will be accepted.
#     x = np.zeros(ncons, dtype=int)
#     y = np.zeros(ncons, dtype=int)
#     redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    
#     while sum(redraw)>0:
#         ncon = sum(redraw)

#         # angle        
#         alpha = np.random.uniform(low=-np.pi, high=np.pi, size=ncon)
        
#         # radius
#         if w_type=='Gaussian':
#             radius = p_param['std'] * np.random.randn(ncon)
#         elif w_type=='Gamma':
#             radius = np.concatenate(
#                 (-np.random.gamma(shape= p_param['kappa'], 
#                                   scale= p_param['theta'], 
#                                   size= int(ncon // 2)),
#                 +np.random.gamma(shape=p_param['kappa'], 
#                                  scale=p_param['theta'], 
#                                  size=ncon -int(ncon // 2)))
#                 )
#         else:
#             raise NotImplementedError

#         # draw coordinates
#         x_ = np.round(radius*np.cos(alpha) + shift_x).astype(int)
#         y_ = np.round(radius*np.sin(alpha) + shift_y).astype(int)
        
#         # make periodic around the presynapse
#         x_ = (x_ + tcol/2) % tcol - tcol/2
#         y_ = (y_ + trow/2) % trow - trow/2
        
#         x[redraw] = x_ 
#         y[redraw] = y_
        
#         # mark those self links for redraw, if necessary
#         if (recurrent) and (not self_link):
#             dist = np.sqrt(x**2+y**2)
#             redraw = dist<=gap
#         else:
#             redraw = np.zeros_like(x, dtype=bool)
            
#     # translating the coordinates w.r.t. source coordinates
#     x = (x + s_coord[0]) % tcol
#     y = (y + s_coord[1]) % trow
#     t_coords = np.array([x,y]).T 
    
#     return s_coord, t_coords.astype(int)


def draw_posts(s_coord, ncons,
              srow, scol, trow, tcol, 
              profile,
              nonuniformity, 
              local_landscape,
              # anisotropy, aniso_methods,
              recurrent=True, self_link= False, 
              ):
    """
    Draws the coordinates of ``ncons`` postsynaptic neurons 
    according to the connectivity profile ``profile`` and the
    anisotropy pattern ``landscape`` using the given ``method``
    (``rotate`` or ``shift``). In case of recurrent pathways,
    if self-link is not allowed, autospes will be ignored and
    extra postsynapses are drawn untill ``ncons`` post that 
    differ from pre is obtained.
    """
    #set_trace()
    # source neuron's location on the target population
    scale_x, scale_y = 1.*trow/srow, 1.*tcol/scol
    s_coord = np.round([s_coord[0]*scale_x, s_coord[1]*scale_y]).astype(int) 
    
    # initialzing containers for postsynapse coordiantes
    x = np.zeros(ncons, dtype=int)
    y = np.zeros(ncons, dtype=int)
    
    # intialzing synaptic parameters
    delays = np.zeros(ncons, dtype=float)
    
    redraw = (x==0) & (y==0)  # index of those who must be redrawn. Now, all.
    while sum(redraw)>0:
        ncon = sum(redraw)
        
        # making a (for now isotropic) point cloud around the presynapse
        if profile['type']=='homog':
            x_= np.random.randint(0,tcol, size=ncon)-s_coord[0]
            y_= np.random.randint(0,trow, size=ncon)-s_coord[1]
        
        else:
            alpha = np.random.uniform(-np.pi, np.pi, ncon)
            radius = get_radial_profile(ncon, profile)
            x_, y_ = radius*np.cos(alpha), radius*np.sin(alpha) 
        
        # making anisotropic connectivity
        x_, y_ = make_anisotropic_profile(x_, y_, 
                                          local_landscape, 
                                          nonuniformity)    
        
        # make coordinates periodic around the presynapse
        x[redraw] = (x_ + tcol/2) % tcol - tcol/2
        y[redraw] = (y_ + trow/2) % trow - trow/2
        delays[redraw] = np.sqrt(x_**2 + y_**2)/2.5 # this division is arbitrary
        
        # mark self-links for redraw, if necessary
        if (recurrent) and (not self_link):
            redraw = (x==0) & (y==0)
        else:
            # source and target populations are different.
            # No self-link can possibly occur. 
            redraw = np.zeros_like(x, dtype=bool)
            
    # translating the coordinates w.r.t. source coordinates
    x = (x + s_coord[0]) % tcol
    y = (y + s_coord[1]) % trow
    t_coords = np.array([x,y]).T 
    
    #set_trace()
    # make anisotropic parameters
    synaptic_params = make_anisotropic_syn(s_coord, t_coords, tcol, 
                                           local_landscape = local_landscape,
                                           nonuniformity = nonuniformity)
    
    
    return s_coord, t_coords.astype(int), synaptic_params


def get_radial_profile(nconn, profile):
    """generate the radial profile for the inhomog. networks."""
    
    wtype = profile['type']
    param = profile['params']
    
    if wtype =='Gaussian':
        radius = param['std'] * np.random.randn(nconn)
    
    elif wtype =='Gamma':
        radius = np.concatenate(
            (-np.random.gamma(shape= param['kappa'], 
                              scale= param['theta'], 
                              size= int(nconn // 2)),
            +np.random.gamma(shape=param['kappa'], 
                             scale=param['theta'], 
                             size=nconn -int(nconn // 2)))
            )
        
    else:
        raise NotImplementedError
    
    # In case of solely inhibitory networks, neurons are excited via excitatory
    # background. In such cases, to emulate the mexian-hat profile, it is 
    # common to prevent establishment of inhibitory links for nearest neighbors.
    # The parameter `gap` ensures that the radial profile, doesn't sample a 
    # post-synapse which is closer than the specified gap. Note that this gap
    # may be squeezed/sheared during the anisotrofy step. But, it is ok, as the
    # mexican hat profile is introduced only for the radial profile. 
    if 'gap' in profile['params']:
        radius[radius< 0] -= profile['params']['gap']
        radius[radius>=0] += profile['params']['gap']
    
    return radius


def make_anisotropic_profile(x,y, local_landscape, nonuniformity):
    """
    takes an isotropic set of coordinates and transforms them
    into anisotropic ones according to the provided ``method``.
    """
    from scipy.spatial.transform import Rotation as R
    
    if 'connectivity' in nonuniformity:
        method = nonuniformity['connectivity']
    else:
        method = None
    #set_trace()
        
    if method!= None:
        # for convenience
        # TODO: This might not be applicable for later forms of profile anisotorpy
        r = local_landscape['r']
        phi = local_landscape['phi']
        r0 = np.array([x, y])
     
        
        if method=='shift':
            x = x + r*np.cos(phi)
            y = y + r*np.sin(phi)
            
    
        elif method=='shift-rotate': # identical to shift. It's written for testing
            r0[0,:] += r
            rot = R.from_euler('z', phi).as_matrix()[:2,:2]
            
            x, y = rot @ r0
        
        elif method=='squeeze-rotate':
            sqz = np.diag([1+r, 1/(1+r)])
            rot = R.from_euler('z', phi).as_matrix()[:2,:2]
            
            x, y = rot @ sqz @ r0
    
        elif method=='positive-rotate':
            r0[0,:] = np.abs(r0[0,:]) 
            rot = R.from_euler('z', phi).as_matrix()[:2,:2]
            
            x, y = rot @ r0
    
        elif method=='positive-squeeze-rotate':
            r0[0,:] = np.abs(r0[0,:]) 
            sqz = np.diag([1+r, 1/(1+r)])
            rot = R.from_euler('z', phi).as_matrix()[:2,:2]
            
            x, y = rot @ sqz @ r0
        else:
            raise
        
    
    return np.round(x).astype(int), np.round(y).astype(int)
    

def make_anisotropic_syn(s_loc, t_locs, gs, local_landscape, nonuniformity):
    
    syn_pars = {}
    
    # delays
    rel_locs = pre_loc2post_loc_rel(s_loc, t_locs, gs)
    delays = np.linalg.norm(rel_locs, axis=1)
    syn_pars['delays'] = delays
    
    for param, method in nonuniformity.items():
        if param!= 'connectivity':
            
            phis = np.arctan2(rel_locs[:,1], rel_locs[:,0])
            phis -= local_landscape['phi']
            
            pmin = local_landscape[param+'min']
            pmax = local_landscape[param+'max']
            if 's' not in local_landscape: 
                s = 1
            else:
                s = local_landscape['s'] # scale factor 
                
            
            if method == 'cos':
                transform = (1+ np.cos(phis/s))/2
            elif method =='sin':
                transform = (1+ np.sin(phis/s))/2
            elif method =='arctan':
                transform = np.arctan(phis/s)
            elif method == 'normal':
                transform = np.exp(-(phis**2)/(2*s**2))
            else:
                raise NotImplementedError('Method not recognized.')
            
            transform -= transform.min()
            transform *= (pmax-pmin)/transform.max()
            transform += pmin
            
            # if param =='U':
            #     set_trace()
                
            syn_pars[param] = transform
            # elif method == 'sin':
            #     phis = np.arctan2(rel_locs[:,1], rel_locs[:,0])
            #     phis -= local_landscape['phi']
            #     Us = local_landscape['Umin'] + (local_landscape['Umax']-local_landscape['Umin'])*(1+ np.sin(phis))/2# * anisotropy['U'] 
                
            #     syn_pars['Us'] = Us
        
    # else:
    #     pass
    
    return syn_pars
