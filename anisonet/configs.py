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
     'noise': {'mu': ..., 'sigma': ...},
     'cell': {'type': 'LIF', 
              'thr': ..., 'ref': ..., 'rest': ...,
              'tau': ..., 'C': ...}
     }
    

with 

#. ``gs``: grid size (int)
#. ``mu`` and ``sigma``: the background current amplitude in standard deviation as Brian quantities with current unit
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
     'synapse': {'type':'alpha_current', 'params': {'J': ..., 'delay': ..., 'tau': ...}},
     'synapse': {'type':'alpha_conductance', 'params': {'J': ..., 'delay': ..., 'tau': ..., 'Erev': ...}},
     'synapse': {'type':'alpha_voltage', 'params': {'J': ..., 'delay':..., 'tau': ...}},
    }
    
    
#. ``type``: either ``"alpha_voltage", "alpha_current"`` or ``"alpha_conductance"`` (str) -- for now, only alpha profile is available
#. ``params``: 
    * ``tau``: rise and decay time scales as a Brian time quantity
    * ``delay``: synaptic delay  as a Brian time quantity
    * ``J``: synaptic qunatal with unit volt, ampere, or siemens for synapses of type ``voltage, current`` or ``conductance``. Note that the sign will determine the polarity of the projection (inhibitory or excitatory).
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

np.random.seed(8)


def get_config(name='EI_net', scalar=3):
    """
    Generates the population and pathways config dictuinary only by providing 
    the name of the desired network.
    
    :param name: nework name. Either "I_net" or "EI_net", defaults to 'EI_net'
    :type name: str, optional
    :param scalar: scales down the network by a factor. The network must be 
        divisble to the factor. Number of connections, their strenghts, and the
        connectivity profile will be scaled accordingly. defaults to 3
    :type scalar: int, optional
    :return: pops_cfg, conn_cfg 
    :rtype: tuple of dicts

    """
    if name=='I_net':
        pops_cfg = {
            'I': {'gs':100//scalar, 
                  'noise': {'mu': 700*pA, 'sigma': 100*pA},
                  'cell': {'type': 'LIF', 
                           #'thr': -55*mV, 'ref': 10*ms, 'rest': -70*mV,
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  }
        }

        conn_cfg = {
            'II': {'ncons': 1000//(scalar**2), 'self_link':False, 
                   #'profile': {'type':'Gamma', 'params': {'theta': 0.5, 'kappa': 8} },
                   'profile': {'type':'Gamma', 'params': {'theta': 3/scalar, 'kappa': 4} },
                   #'profile': {'type':'Gaussian', 'params': {'std': 3} },
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
#                   'synapse': {'type':'alpha_voltage', 'params': {'J': -0.221*mV, 'delay':1*ms, 'tau': 5*ms}},
                   #'anisotropy': {'type': 'perlin', 'params': {'r': 1, 'scale':3}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/4.}}
                   #'anisotropy': {'type': 'random', 'params': {'r': 1,}}
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   },
        }    
    elif name=='EI_net':
        # pops_cfg = {
        #     'I': {'gs':60, 'mu': 350*pA, 'sigma': 100*pA},
        #     'E': {'gs':120, 'mu': 350*pA, 'sigma': 100*pA}
        # }

        # lscp_cfg = {
        #     'I': {'type': 'perlin', 'params': {'scale': 4, 'r':1} }
        # }

        # conn_cfg = {
        #     'EE': {'type':'Gaussian', 'self_link':False, 'params': {'std': 9}},
        #     'EI': {'type':'Gaussian', 'self_link':False, 'params': {'std': 12}},# EI means E -> I
        #     'IE': {'type':'Gaussian', 'self_link':False, 'params': {'std': 4.5}}, # IE means I -> E
        #     'II': {'type':'Gaussian', 'self_link':False, 'params': {'std': 6}}, 
        # }

        # syns_cfg = { 
        #     'EE': {'synapse_dyn': 'alpha', 'ncons':720, 'J': 10*pA}, # note the sign
        #     'EI': {'synapse_dyn': 'alpha', 'ncons':180, 'J': 10*pA}, # note the sign
        #     'IE': {'synapse_dyn': 'alpha', 'ncons':720, 'J': -1.76*mV}, # note the sign
        #     'II': {'synapse_dyn': 'alpha', 'ncons':180, 'J': -1.76*mV}, # note the sign
        # }
        pops_cfg = {
            'I': {'gs':60//scalar, 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  },
            
            'E': {'gs':120//scalar, 
                  'noise': {'mu': 350*pA, 'sigma': 100*pA},
                  'cell': {'type': 'LIF', 
                           'thr': -55*mV, 'ref': 2*ms, 'rest': -70*mV,
                           'tau': 10*ms, 'C': 250*pF}
                  }
        }

        conn_cfg = {
            'EE': {'ncons': 720//(scalar**2), 'self_link':False, 
                  'profile': {'type':'Gaussian', 'params': {'std': 9/scalar} },
                  'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms} },
                  'anisotropy': {'type': 'perlin', 'params': {'scale': 3, 'r':1}}
                  #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/4.}}
                  },
            
            'EI': {'ncons': 180//(scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 12/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': 10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/4.}}
                   },
            
            'IE': {'ncons': 720//(scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 4.5/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi}}
                   },

            'II': {'ncons': 180//(scalar**2), 'self_link':False, 
                   'profile': {'type':'Gaussian', 'params': {'std': 6/scalar}},
                   'synapse': {'type':'alpha_current', 'params': {'J': -10*(scalar**2)*pA, 'delay':1*ms, 'tau': 5*ms}},
                   'anisotropy': {'type': 'symmetric', 'params': {}}
                   #'anisotropy': {'type': 'homogeneous', 'params': {'r': 1, 'phi':np.pi/2.}}
                   },
        }    
    else:
        raise
    
    return pops_cfg, conn_cfg


