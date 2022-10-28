#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configs is a utility module that passes the default configurations according to
the `[1]`_. Configurations are set-up in form of nested dictionaries whose 
structure is explained below. If you'd  like to get the exact configurations as 
in `[1]`_, use the ``get_config`` utility below.


==========================
Population's configuration
==========================

Populations are identified by a single character name (like ``"P"``) and must 
have parameters in the following form:

..  code-block:: python
    
    {'gs': ..., 
     'noise': {'mu': ..., 'sigma': ..., 'noise_dt': ...},
     'cell': {'type': 'LIF', 
              'thr': ..., 'ref': ..., 'rest': ...,
              'tau': ..., 'C': ...}
     }
    

with 

#. ``gs``: grid size (int)
#. ``mu`` and ``sigma``: the background current amplitude in standard deviation as Brian quantities with current unit
#. ``noise_dt``: the time scaling of the Wiener process. Please refer to :ref:`equations:Noise scaling` for details.
#. ``type``: fixed -- for the moment only LIF neuron is possible
#. ``thr`` and ``rest``: threhold and resting potentials as Brian quantities with voltage unit
#. ``ref``: refractory period as a Brian quantity with time unit
#. ``tau``: membrane time scale as a Brian quantity with time unit
#. ``C``: membrane capacitance as a Brian quantity with capacitance unit


=======================
Pathways' configuration
=======================

Pathways are identified by two-character names (like ``"PQ"``) that specifies 
the source (P) and target (Q) populations. Each pathway must have be configured
similar to the following form:

..  code-block:: python

    {'ncons': ..., 'self_link': False, 
     'profile': {...}
     'anisotropy': {...}
    }


#. ``ncons``: number of connections from each source neuron to the target population (int)
#. ``self_link``: if self-link is allowed; only important if source and target are the same object. In other words, pathway is recurrent. (bool)

~~~~~~~
Profile
~~~~~~~
Use either of the following structures for the value of ``profile`` key:
    
..  code-block:: python

    {
     'profile': {'type':'Gamma', 'params': {'theta': ..., 'kappa': ...} },  # this
     'profile': {'type':'Gaussian', 'params': {'std': ...} }, # or this can be used
    }
    
    
#. ``type``: either ``"Gamma"`` or ``"Gaussian"`` (str)
#. ``params``: distribution parameters (float)

~~~~~~~
Synapse
~~~~~~~
Use either of the following structures for the value of ``profile`` key:
    
..  code-block:: python

    {
     'synapse': {'type':'alpha_current', 'params': {'J': ..., 'delay': ..., 'tau': ...}}, # usually we use this
     'synapse': {'type':'alpha_voltage', 'params': {'J': ..., 'delay':..., 'tau': ...}},
     'synapse': {'type':'alpha_conductance', 'params': {'J': ..., 'delay': ..., 'tau': ..., 'Erev': ...}},
    }
    
    
#. ``type``: encodes both synpatic *kernel* and *model* in form of 
   ``<kernel>_<method>``. Please refer to  :ref:`equations:Synapse equations` 
   for possible values of kernels and models.
#. ``params``: 

   * ``tau``: synaptic timescales as a Brian time quantity (for exp, and alpha kernels)
   * ``tau_r`` and ``tau_d``: rise and decay timescales as a Brian time quantity (for biexp kernel)
   * ``delay``: synaptic delay  as a Brian time quantity
   * ``J``: synaptic qunatal with unit volt, ampere, or siemens for synapse
     models ``jump``, ``current`` or ``conductance`` (c.f. :ref:`equations:Synapse equations` ).
     Note that the sign will determine the polarity of the projection (inhibitory or excitatory).
   * ``Erev``: the reversal potential for conductance-based synapse as a Brian quantity of unit volt


~~~~~~~~~~
Anisotropy
~~~~~~~~~~
Use either of the following structures for the value of ``profile`` key:
    
..  code-block:: python

    {
     'anisotropy': {'type': 'perlin', 'params': {'r': ..., 'scale': ...}}
     'anisotropy': {'type': 'homogeneous', 'params': {'r': ..., 'phi': ...}}
     'anisotropy': {'type': 'random', 'params': {'r':  ...,}}
     'anisotropy': {'type': 'symmetric', 'params': {}}
    }
    
#. ``type``: name of anisotropy profile. For now only one of the ``['perlin, homogeneous, random, symmetric']`` are possible. (str)
#. ``params``:

   * ``r``: displacement (float) -- not important for symmetric type,
   * ``scale``: perlin scale if type is ``"perlin"`` (int)
   * ``phi``: uniform anisotropic angle if type is ``"homogeneous"`` (float)

    

.. _[1]: https://doi.org/10.1371/journal.pcbi.1007432
"""

from brian2.units import pA, mV, ms, pF
import numpy as np

np.random.seed(18)


def round_to_even(gs, scaler):
    """
    We better round things to even number for better visualization
    """
    rounded = round(gs/scaler)
    if rounded%2:
        rounded+=1
    return int(rounded)

def get_config(name='EI_net', scalar=3):
    """
    Generates the population and pathways config dictuinary only by providing 
    the name of the desired network.
    
    .. note::
        One should differetiate between homogeneity/randomness in angle and 
        location. `[1]`_ used these terms somewhat loosely. We use the following 
        terms for different setups:
            
            * ``homiso_net``: Homogenous and isotropic netowrk, equivalent to the fully
              random graph of Erdos-Renyi.
            * ``iso_net``: Isotropic but spatially inhomogeneous (in a locally 
              conneted manner, although with a long-tailed radial profile one can
              generate few long-range connection -- thus produce a small-world net).
            * ``homo_net``: Connections are formed without dependence on the distance,
              but angle.
            * ``I_net``: the recurrent inhibitory network with radial and angular
              profiles according to to `[1]`_.
            * ``EI_net``: the recurrent inhibitory network with radial and angular
              profiles according to to `[1]`_.

        Also note that these structures are independent from how anisotropy is 
        imposed.


    :param name: nework name. Either "I_net" or "EI_net", defaults to 'EI_net'
    :type name: str, optional
    :param scalar: scales down the network by a factor. The network must be 
        divisble to the factor. Number of connections, their strenghts, and the
        connectivity profile will be scaled accordingly. defaults to 3
    :type scalar: int, optional
    :return: pops_cfg, conn_cfg 
    :rtype: tuple of dicts

    .. _[1]: https://doi.org/10.1371/journal.pcbi.1007432

    """
    if name=='I_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   #'profile': {'type':'Gaussian', 'params': {'std': 3} },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'perlin', 'params': {'r': np.sqrt(2), 'scale':3}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/6.}}
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   #'anisotropy': {'type': 'symmetric', 'params': {}}
                   },
        }    
    elif name=='E_net':
        pops_cfg = {
            'E': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 275*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'EE': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   #'profile': {'type':'Gaussian', 'params': {'std': 3} },
                   'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   #'anisotropy': {'type': 'perlin', 'params': {'r': np.sqrt(2), 'scale':3}}
                   'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/6.}}
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   #'anisotropy': {'type': 'symmetric', 'params': {}}
                   },
        }    
    
    elif name=='EI_net':
        noiser = 1.
        meaner = 1
        pops_cfg = {
            'I': {'gs': round_to_even(60, scalar), 
                  'noise': {'mu': 350*pA/meaner, 'sigma': 100*pA/noiser, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  },
            
            'E': {'gs': round_to_even(120, scalar), 
                  'noise': {'mu': 350*pA/meaner, 'sigma': 100*pA/noiser, 'noise_dt': 1*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  }
        }

        conn_cfg = {
            'EE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                  #'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                  'profile': {'type':'Gaussian', 'params': {'std': 9/scalar} },
                  'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms} },
                  #'anisotropy': {'type': 'perlin', 'params': {'scale': 3, 'r':np.sqrt(2)}},
                  #'anisotropy': {'type': 'symmetric', 'params': {}}
                  'anisotropy': {'type': 'homogeneous', 'params': {'r': np.sqrt(2), 'phi':np.pi/6.}}
                  },
            
            'EI': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 4.5/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'random', 'params': {'r': 1,}},
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   },
            
            'IE': {'ncons': round_to_even(720, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 12/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'random', 'params': {'r': 1,}},
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   },

            'II': {'ncons': round_to_even(180, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 6/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': -80*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'random', 'params': {'r': 1,}},
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   },
        }
        
    elif name=='homo_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': None,
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/6.}}
                   'anisotropy': {'type': 'perlin', 'params': {'r': np.sqrt(2), 'scale':3}}
                   },
        }    
        
    elif name=='iso_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': None
                   },
        }    

    elif name=='homiso_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': None,
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': None
                   },
        }    
    elif name=='STSP_TM_I_net':
        pops_cfg = {
            'I': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'II': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   #'profile': {'type':'Gaussian', 'params': {'std': 3} },
                   #'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                    # 'synapse': {'type':'tsodysk-markram_jump', 
                    #             'params': {'J': -0.221*mV*(scalar**2), 'delay':1*ms, 
                    #                        'tau_f': 1500*ms, 'tau_d': 200*ms, 'U':0.5}},
                    'synapse': {'type':'tsodysk-markram_current', 
                                'params': {'J': -10*pA*(scalar**2), 'delay':1*ms, 
                                           'tau_f': 1500*ms, 'tau_d': 200*ms, 'U':1}},
                   
                   
                   'anisotropy': {'type': 'perlin', 'params': {'r': np.sqrt(2), 'scale':3}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/6.}}
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   #'anisotropy': {'type': 'symmetric', 'params': {}}
                   },
        }
        
    elif name=='STSP_TM_E_net':
        pops_cfg = {
            'E': {'gs': round_to_even(100, scalar), 
                  'noise': {'mu': 325*pA, 'sigma': 100*pA, 'noise_dt': 1.*ms},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau':10*ms, 'C': 250*pF}
                          }
            }

        conn_cfg = {
            'EE': {'ncons': round_to_even(1000, scalar**2), 'self_link':False, 
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   #'profile': {'type':'Gaussian', 'params': {'std': 3} },
                   #'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'synapse': {'type':'tsodysk-markram_jump', 
                               'params': {'J': .4*mV*(scalar**2), 'delay':1*ms, 
                                          'tau_f': 1500*ms, 'tau_d': 600*ms, 'U':0.5}},
                   'anisotropy': {'type': 'perlin', 'params': {'r': np.sqrt(2), 'scale':3}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/6.}}
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   #'anisotropy': {'type': 'symmetric', 'params': {}}
                   },
            }
    else:
        raise
    
    return pops_cfg, conn_cfg


