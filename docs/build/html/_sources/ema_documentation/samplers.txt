***************
:mod:`samplers`
***************

.. automodule:: samplers

:class:`Sampler`
================
.. autoclass:: Sampler
   :show-inheritance:
   
   .. autoattribute:: distributions
   
   .. automethod:: sample
   .. automethod:: generate_design
   

:class:`LHSSampler`
===================
.. autoclass:: LHSSampler
   :show-inheritance:
   
   .. automethod:: sample
   .. automethod:: _lhs

:class:`MonteCarloSampler`
==========================
.. autoclass:: MonteCarloSampler
   :show-inheritance:
   
   .. automethod:: sample

:class:`FullFactorialSampler`
=============================
.. autoclass:: FullFactorialSampler
   :show-inheritance:
   
   .. autoattribute:: max_designs
   
   .. automethod:: generate_design


