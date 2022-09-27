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
 
==============
Synaptic model
==============
There are three different conventional models for synapses:
    
    #. conductance-based,
    #. current-based, and
    #. voltage-based
    
which, from the both mathematical and implementation perspective have 
differences. From mathematical point of view, a conductance based synapse 
is governed by:
    
.. math:: 
	I_{syn} = g_{max} g(t) (V_{post} - V_{pre})\n
    g(t) = \int_{-\infty}^t \sum_k f(t-t') \delta(t'-t_k) dt' = \sum_k f(t-t_k)

A current-based synapse, on the other hand, describes the synaptic current as:

.. math::
	I_{syn} = I_{max} \int_{-\infty}^t \sum_k f(t-t') \delta(t'-t_k) dt' = I_{max} \sum_k f(t-t_k)
    
Finally, in voltage-based formulation we don't describe the injected current, 
but directly the increment in the membrane voltage:
    
.. math::
    \Delta V_{syn} = V_{max} \int_{-\infty}^t \sum_k f(t-t') \delta(t'-t_k) dt' =  V_{max} \sum_k f(t-t_k)

In all these formulations :math:`f(t-t')` is a filter, often replaced by an 
exponential, bi-exponential (with rise and decay time :math:`\\tau_r` and 
:math:`\\tau_d`), and alpha function (equal rise and decay time), scaled by the
:math:`_{max}` qunatites. :math:`\sum \delta(.)` is the incoming spike train. 
Here we use the alpha function -- scaled to a maximum of one -- with the 
following system:

..  code-block:: python

    '''
    dg/dt = -(g+h)/tau : 1
    dh/dt = -h/tau :1
    '''
     
    ...
    
    on_pre = '''h += exp(1)'''  # takes care of the max-normalizing to 1
    


Computationally, in Brian the first two require a ``I_syn`` term in the neuronal
equations that needs to be evaluated on every single timestep. One the 
contrary, voltage-based synapses only manipulate the potential of the 
post-synapse upon arrival of the presynaptic spike. Therefore, voltage-based 
synapse is computationally cheaper, immensely. In Brian, current/conductance-based 
synapses must be handled as ``(clock-driven)`` states whereas the voltage-based 
one can be computed in a ``(event-driven)`` fashion. Also look at 
(`Brian documentation`_) for more detail.


It's however possible to transfer the current-based synapse to the voltage-based
one with the following assumption:
    
.. warning::
    **Asumption**: The total current injected to the cell via a single spike 
    leads to a  certain voltage increament. Setting :math:`V_{max}` equal to 
    this increment, we can make an voltage-based synapse from the current-based 
    one.
    
This assumption leads to the scaling :math:`V_{max} = \\frac{I_{max} \\tau}{C} e`.


.. _Brian documentation: https://brian2.readthedocs.io/en/stable/user/synapses.html?highlight=event-driven#event-driven-updates
.. _notebook: https://nest-simulator.readthedocs.io/en/v3.3/model_details/noise_generator.html
.. _[2]: https://www.researchgate.net/publication/260254488_Equation-oriented_specification_of_neural_models_for_simulations
"""

import brian2 as b2


def get_nrn_eqs(pop_name, pops_cfg, syn_base):    
    """
    It matters how the synapse inputs are modeled. Seemingly, the one used in
    NEST and the paper are using a "current-alpha", as opposed to "voltage-" or
    "conductance-". Look at `source code`_ for more details.
    
    :param pop_name: population name
    :type pop_name: str
    :param pops_cfg: population configuration
    :type pops_cfg: dict
    :param syn_base: synapse model base (infered from the config dicts)
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
    
    if syn_base!='voltage':
        eqs_str= tmp + '''dv/dt = (E-v)/tau + (noise_pop + I_syn)/C : volt (unless refractory)\n'''
        I_syn_components = []
        for src_name in pops_cfg.keys():
            eqs_str+= '''I_syn_{}: amp \n'''.format(src_name)
            I_syn_components.append('I_syn_'+src_name)
        eqs_str+= 'I_syn = '+ '+'.join(I_syn_components) +': amp \n'''
        
    else:
        eqs_str= tmp + '''dv/dt = (E-v)/tau + noise_pop/C : volt (unless refractory)\n'''
        
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
    
    tmp = '''
        dg/dt = (-g+h) / tau : 1 (clock-driven)
        dh/dt = -h / tau : 1 (clock-driven)
        '''
        
    on_pre ='''h+=exp(1)'''
    on_post = ''
    
    if syn_base=='voltage':
        # J is the post-synaptic voltage increment
        eqs_str = tmp.replace('clock-driven', 'event-driven')
        eqs_str += '''w = J*g: volt \n'''
        #eqs_str += '''J = {}*mV: volt \n'''.format(J/mV)  
        eqs_str += '''J: volt (shared)\n'''  
        on_pre += '''\nv_post += w'''
        
    if syn_base=='delta_voltage':
        # J is the post-synaptic voltage increment
        eqs_str = '''w = J: volt \n'''
        #eqs_str += '''J = {}*mV: volt \n'''.format(J/mV)  
        eqs_str += '''J: volt (shared)\n'''  
        on_pre += '''\nv_post += w'''
    
    elif syn_base=='current':
        # J is the post-synaptic current injection 
        eqs_str = tmp
        eqs_str += '''I_syn_{}_post = J*g: amp (summed)\n'''.format(conn_name[0])
        eqs_str += '''J : amp (shared)\n'''
        
    elif syn_base=='conductance':
        # J is the maximum conductance
        eqs_str = tmp
        eqs_str += '''I_syn_{}_post = J*g*(v_post-Erev): amp \n'''.format(conn_name[0])
        eqs_str += '''Erev : volt (shared)\n'''
        eqs_str += '''J : siemens (shared)\n'''
        #eqs_str += '''J = {}*nS: siemens \n'''.format(J/nS)  
        #E = conn_cfg[conn_name]['synapse']['params']['E']/mV
        
    eqs = b2.Equations(eqs_str, 
                    tau = conn_cfg[conn_name]['synapse']['params']['tau'])
    
    return eqs, on_pre, on_post
