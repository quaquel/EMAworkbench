********************
:mod:`uncertainties`
********************

.. automodule:: uncertainties

:class:`AbstractUncertainty`
============================
.. autoclass:: AbstractUncertainty
   :show-inheritance:

   .. autoattribute:: values
   .. autoattribute:: name
   .. autoattribute:: type
   .. autoattribute:: dtype
   .. autoattribute:: dist

   .. automethod:: __init__
   .. automethod:: get_values
   
   
:class:`ParameterUncertainty`
=============================
.. autoclass:: ParameterUncertainty
   :show-inheritance:

   .. autoattribute:: default
   
   .. automethod:: __init__
   .. automethod:: get_default_value

:class:`CategoricalUncertainty`
===============================
.. autoclass:: CategoricalUncertainty
   :show-inheritance:
   
   .. autoattribute:: categories

   .. automethod:: __init__
   .. automethod:: transform
   .. automethod:: invert
