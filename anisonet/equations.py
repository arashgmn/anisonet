#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The equation module provides proper equation strings for both neuron and 
synapses. Note however, due to the existence of the stochastic background 
current, extra care must be paid when translating the code of NEST to Brian 
and vice versa.


=============
Noise scaling
=============

According to this `notebook`_ in NEST a Gaussian ``noise_generator`` adds a 
constant current :math:`I_j` for the inteval :math:`[t_j, t_j + \Delta t]` 
using a normal distribution:

.. math::
	I_j = \mu + \sigma N(0,1)
	
With such an input, the total charge injected until time :math:`t` is

.. math::
	q(t)_{NEST} = \int_0^t I dt' = \mu t + \sigma \sum_{j=0}^n I_j \Delta t

where we have used the Riemann sum. The mean and variance of total charge are

.. math::
	E[q(t)_{NEST}] = \mu t  \\\\
	Var[q(t)_{NEST}] = 0 + \sigma^2 \Delta t^2 * n = (n\Delta t) \sigma^2 \Delta t =  t \sigma^2 \Delta t

However, in Brian, the same stochastic current is modelled as Wiener processes (:math:`dW \~ N(0, dt)`) whose
variance depends on the timestep `dt` (`[2]`_). Wrting the stochastic current in its `Langevin form`, we get:

.. math::
	I_j = \mu + \sigma \zeta(t_j)
	
which gives rise to

.. math::
	q(t)_{Brian} = \int_0^t I dt' = \mu t + \sigma \int_0^t \zeta(t') dt' = \mu t + \sigma W(t) 

with :math:`W(t)` being a Brownian trajectory. Thus, the mean and variance are

.. math:: 
	E[q(t)_{Brian}] = \mu t  \\\\
	Var[q(t)_{Brian}] = 0 + t \sigma^2

As a result, the variance of Brian and NEST are related in the following way:

.. math:: 
	\sigma^2_{Brian} = \sigma^2_{NEST} \Delta t_{NEST}
 
 
=================
Synapse equations
=================
Synaptic inputs can be added to the neuron either by a current, or a prescribed jump
in the voltage. By default, all synaptic states must be updated in each timestamp -- 
so-called ``clock-driven`` -- which is very time consuming. However, if synapses 
induces discret voltage jumps, for certain synaptic models, it is possible to update
the synaptic inputs only when a spike arrives -- thus, ``event-driven``. This enhances
the performance substatially. Also look at (`Brian documentation`_) for more detail.


We use the following numenclature to explain possible synaptic models:

* **model**: specifies how neuron incorporates the synaptic input. Possible 
  options are ``conductance, current``, and ``jump``.
* **kernel**: determines the dynamics of synaptic input. In other words, how
  the strength of synaptic input changes over time. Possible options are 
  ``alpha``, ``biexp`` (for bi-exponential), ``exp`` (for exponential), and 
  ``const`` (for constant).


Model
~~~~~
From mathematical point of view, different models mean the following term in the 
neuron equation

#. **conductance-based**: :math:`I_{syn}(t) = J_{max} g(t) (V_{post} - V_{pre})`
#. **current-based**: :math:`I_{syn}(t) = J_{max} g(t)`
#. **increment-based (jump)** : :math:`I_{syn}(t) = J_{max} g(t) \delta(t)`

	
in which :math:`g(t)` is the kernel (look below), and :math:`\delta(t)` is unitary 
Dirac's delta function that picks a specific strength for the synaptic input whenever
a spike arrives. Note that the :math:`J_{max}` quantities have different unit in each 
model (respectively simense, ampre, and volt/s in SI units). 

.. note::
    We do not include a synaptic current term for ``jump`` model. Instead, we use the ``on_pre``
    syntax of Brian to directly apply this jump. Consequently, the strength of the jump,  
    :math:`J_{max}`, must have a unit of volts.
    
    
Kernel
~~~~~~
Kernels, are unitless functions that control the temporal characteristics of the synaptic 
input. **It is important to remember that the kernels, by design are NOT bounded**. The 
following kernels are available (look below to see which ones are available):

#. **constant** (``const``): :math:`g(t) = 1`
#. **exponential** (``exp``): :math:`g(t) = \exp(-t/\\tau)`
#. **alpha** (``alpha``): :math:`g(t) = A \\frac{t}{\\tau} \exp(-t/\\tau)`
#. **bi-exponential** (``biexp``): :math:`g(t) = A \\frac{\\tau_2}{\\tau_2 - \\tau_1} [ \exp(-t/\\tau_1) - \exp(-t/\\tau_2) ]`

    
In which the coefficient :math:`A` max-normalizes the kernel to 1. In brian, these kernels
are implemented as first-order ordinary differential equations. For instance, for ``alpha``
the kernel is modeled as 

..  code-block:: python

    '''
    dg/dt = -(g+h)/tau : 1
    dh/dt = -h/tau :1
    '''
     
    ...
    
    on_pre = '''h += exp(1)'''  # takes care of the max-normalizing to 1
    

.. note::
    ``biexp`` kernel is still under development and is not tested yet.
    
    
    
========================
Conversion to jump model
========================
To avoide long run-times, one can convert the ``clock-driven`` models to
``event-driven`` voltage-increment ones by equating the total injected 
charge to the membrane upon arrival of a spike. For instance, for a 
current-based synapse model we can write

.. math:: 
	q = \int I_{max}g(t) dt \equiv C\ V_{max}


where :math:`V_{max}` is the equivalent increment in the membrane potential. 
Performing the integration we get

#. ``exp``: :math:`V_{max} = \\frac{I_{max} \\tau_s}{C}`.
#. ``alpha``: :math:`V_{max} = \\frac{I_{max} \\tau_s}{C} e`. (Factor e is due 
   to normalization of alpha kernel).
#. ``biexp``: :math:`V_{max} = ???` (TODO :D ).


This a utility function that transforms current-based models to increment-based
ones is implemented as a method of ``Simulate`` class. 

.. note::
   Such equivalence is only possible for decaying kernels. Therefore, constant
   kernel cannot be converted. Furthermore, conductance-based models, due to
   their dependence on both pre and post voltages are not amenable to this 
   equivalence. 

.. warning::
    This equivalence is not mathematically rigorous. A general non-jumping 
    model computes the voltage increment due to synaptic inputs according to
    
    .. math::
        Cv(t) = \int (I_{syn}(t) + ...) dt = \int (\sum_k I_k(t) + ... ) dt 

    with $k$ enumerating the afferent fibers to a neuron. Yet, increment-based 
    model, assumes that each spike is independt from the others. Therefore, 
    every spike induces the same amount of potentiation (or depression). 
    Mathematically, this translates to membrane voltage increment according to
    
    .. math::
    	Cv(t) = \int (I_{syn}(t) + ...) dt = \sum_k \int I_k(t) dt + \int (...) dt 
        
    Therefore, such an equivalence is true if spiking patterns is sparse, 
    or when :math:`\\tau_s \ll \\tau_m`, meaning that the synaptic dynamic is 
    so rapid that the membrane essentially doesn't resolve it. In other words,
    one when timescale seperation is possible.
    
    
.. _Brian documentation: https://brian2.readthedocs.io/en/stable/user/synapses.html?highlight=event-driven#event-driven-updates
.. _notebook: https://nest-simulator.readthedocs.io/en/v3.3/model_details/noise_generator.html
.. _[2]: https://www.researchgate.net/publication/260254488_Equation-oriented_specification_of_neural_models_for_simulations
"""

import brian2 as b2
import copy

def get_nrn_eqs(pop_name, pops_cfg, syn_base):    
    """
    It matters how the synapse inputs are modeled. Seemingly, the one used in
    NEST and the paper are using a "current-alpha", as opposed to "voltage-" or
    "conductance-". Look at `source code`_ for more details.
    
    :param pop_name: population name
    :type pop_name: str
    :param pops_cfg: population configuration
    :type pops_cfg: dict
    :param syn_base: synapse model and kernel (inferred from the config dicts)
    :type syn_base: str
    :return: Brian Equation object for neuron
    :rtype: Brian Equation object

    
    .. _source code: https://github.com/nest/nest-simulator/blob/d7b3c7671c5fcd9acf211b58012db9729607a1db/models/iaf_psc_alpha.h#L48
    """
    tmp = '''
        mu: amp (shared)
        sigma: amp (shared)
        noise_pop = mu + sigma*sqrt(noise_dt)*xi_pop: amp 
        '''
        
    noise_dt = pops_cfg[pop_name]['noise']['noise_dt']
    tmp = tmp.replace('noise_dt', str(noise_dt/b2.ms)+'*ms')
    
    kernel, model = syn_base.split('_') # identify kernel and model
    if model in ['conductance', 'current']:
        eqs_str= tmp + '''dv/dt = (E-v)/tau + (noise_pop + I_syn)/C : volt (unless refractory)\n'''
        I_syn_components = []
        for src_name in pops_cfg.keys():
            eqs_str+= '''I_syn_{}: amp \n'''.format(src_name)
            I_syn_components.append('I_syn_'+src_name)
        eqs_str+= 'I_syn = '+ '+'.join(I_syn_components) +': amp \n'''
        
    else:
        eqs_str= tmp + '''dv/dt = (E-v)/tau + (noise_pop)/C : volt (unless refractory)\n'''
        
    eqs_str = eqs_str.replace('_pop', '_'+pop_name)
    eqs = b2.Equations(eqs_str, 
                    C = pops_cfg[pop_name]['cell']['C'],
                    E = pops_cfg[pop_name]['cell']['rest'],
                    tau = pops_cfg[pop_name]['cell']['tau'])
    
    return eqs

def get_syn_eqs(conn_name, conn_cfg, syn_base):
    """
    It matters how the synapse inputs are modeled. Seemingly, the one used in
    NEST and the paper are using a "current-alpha", as opposed to "voltage-" or
    "conductance-": `nest repo`_.
    
    Nest normalizes its models according the the following rule (`nest docs`_):
        
        - `iaf_psc_alpha`: max current of unitary excitation/inhiition = 1 pA
        - `iaf_cond_alpha`: max conductance of unitary excitation/inhiition = 1 nS
        
    In order to get the same behaviour in brian, we need to ascribe a weight to
    our unitary events. (This weight happen to be exp(1).) In addition, the 
    parameter ``h`` is unbounded, and can be well beyond 1. 
    
    #TODO: check if that's also the case in nest.
     

    .. _nest repo: https://github.com/nest/nest-simulator/blob/d7b3c7671c5fcd9acf211b58012db9729607a1db/models/iaf_psc_alpha.h#L48
    .. _nest docs: https://nest-simulator.readthedocs.io/en/v2.20.1/models/neurons.html#classnest_1_1iaf__psc__alpha


    :param conn_name: pathway name
    :type conn_name: str
    :param conn_cfg: population connections configurations
    :type conn_cfg: dict
    :param syn_base: synapse model base (infered from the config dicts)
    :type syn_base: str
    :return: tuple of (synapse equation, on_pre, on_post)
    :rtype: (Brian Equation object, str, str)

    """
    
    # equation templates
    tmp_alpha = '''
        dg/dt = (-g+h) / tau : 1 (clock-driven)
        dh/dt = -h / tau : 1 (clock-driven)
        '''
    
    tmp_biexp = '''''' # TODO    
    
    tmp_exp = '''
        dg/dt = -g / tau : 1 (clock-driven)
        '''
    
    tmp_tsodysk_markram = '''
        dx/dt = (1-x)/tau_d: 1 (clock-driven)
        du/dt = (U-u)/tau_f: 1 (clock-driven)
        g: 1
    '''
    
    # Constructing equations
    kernel, model = syn_base.split('_') # identify kernel and model
    print(kernel, model)
    # kernel related components (only traces updates are given here)
    if kernel == 'alpha':
    	eqs_str = tmp_alpha
    	on_pre = 'h += exp(1)' 
    
    elif kernel== 'biexp':
    	eqs_str = tmp_biexp
    	on_pre = '' # TODO: we have to decide how do we normalize exp kernel
    
    elif kernel== 'exp':
    	eqs_str = tmp_exp
    	on_pre = 'g += 1' # TODO: we have to decide how do we normalize exp kernel
        
    elif kernel=='const' : # constant kernel
    	eqs_str = 'g = 1 : 1'
    	on_pre = '' # no 
    
    elif kernel=='tsodysk-markram':
        eqs_str = tmp_tsodysk_markram
        on_pre = '''
            u += U * (1 - u)
            g = u * x
            x = x-g
        '''
    
    else:
        raise NotImplementedError('synaptic kernel type "{}" is not recognized!'.format(kernel))
    
    
    # model related components (pre/post updates are prescribed here)
    if model=='conductance':
        eqs_str += '''J : siemens (shared)\n'''
        eqs_str += '''Erev : volt (shared)\n'''
        eqs_str += '''I_syn_{}_post = J*g*(v_post-Erev): amp (summed)\n'''.format(conn_name[0])
        
    elif model=='current':
        eqs_str += '''J : amp (shared)\n'''
        eqs_str += '''I_syn_{}_post = J*g: amp  (summed)\n'''.format(conn_name[0])
    
    elif model=='jump':
    	eqs_str = eqs_str#.replace('clock-driven', 'event-driven')
    	eqs_str += '''\nJ: volt (shared)'''
    	on_pre += '''\nv_post += J*g'''        
    
    else:
        raise NotImplementedError('synaptic model type "{}" is not recognized!'.format(model))
    
    on_post = ''

    
    
    # if syn_base=='alpha_conductance':
    #     # J is the maximum conductance
    #     eqs_str = tmp_alpha
    #     eqs_str += '''J : siemens (shared)\n'''
    #     eqs_str += '''Erev : volt (shared)\n'''
    #     eqs_str += '''I_syn_{}_post = J*g*(v_post-Erev): amp \n'''.format(conn_name[0])
        
    #     on_pre  = '''h+=1''' # have not tested yet
        
    # elif syn_base=='alpha_current':
    #     # J is the post-synaptic current injection 
    #     eqs_str = tmp_alpha
    #     eqs_str += '''J : amp (shared)\n'''
    #     eqs_str += '''I_syn_{}_post = J*g: amp (summed)\n'''.format(conn_name[0])
        
    #     on_pre  = '''h+=exp(1)''' # to scale the max to 1
        
        
    # elif syn_base=='alpha_voltage':
    #     # J is the post-synaptic voltage increment
    #     eqs_str = tmp_alpha.replace('clock-driven', 'event-driven')
    #     eqs_str += '''J: volt (shared)\n'''  
        
    #     on_pre  = '''h+=exp(1)''' # don't know why this works but exp(1) not
    #     on_pre += '''\nv_post += J*g'''
        
    # elif syn_base=='exp_voltage':
    #     # J is the post-synaptic voltage increment
    #     eqs_str = tmp_exp.replace('clock-driven', 'event-driven')
    #     eqs_str += '''J: volt (shared)'''  
        
    #     on_pre = '''g+=1''' # have not tested yet
    #     on_pre+= '''\nv_post += J'''


    # elif syn_base=='delta_voltage':
    #     eqs_str = '''J: volt (shared)'''  
        
    #     on_pre = '''\nv_post += J*exp(1)''' # to match it with the alpha_current
    
    # else:
    #     print(syn_base)
    #     raise NotImplementedError
        
    eqs = b2.Equations(eqs_str)
    namspace = copy.deepcopy(conn_cfg[conn_name]['synapse']['params'])
    namspace.pop('J')

    return eqs, on_pre, on_post, namspace
    
